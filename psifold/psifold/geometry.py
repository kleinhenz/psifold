import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def pdist(x):
    return torch.norm((x - x[:, None]), p=2.0, dim=2)

@torch.jit.script
def dRMSD(x_hat, x):
    """Compute dRMSD loss

    Args:
        x_hat: (L, B, D)
        x: (L, B, D)

    Returns:
        dRMSD: (scalar) batch averaged root mean square error
               of pairwise distance matrices of x and x_hat
    """

    L, B, D = x_hat.size()
    assert D == 3

    # (B, L, D)
    x = x.permute(1, 0, 2)
    x_hat = x_hat.permute(1, 0, 2)

    # loop over each batch
    drmsd = torch.zeros(1, dtype=x.dtype, device=x.device)
    for i in range(B):
        delta = pdist(x_hat[i]) - pdist(x[i])
        drmsd += delta.pow(2.0).sum().div(L * (L - 1.0)).pow(0.5)

    return drmsd / B

@torch.jit.script
def dRMSD_masked(x_hat, x, mask):
    """Compute dRMSD loss

    Args:
        x_hat: (L, B, D)
        x: (L, B, D)
        mask: (L, B)

    Returns:
        dRMSD: (scalar) batch averaged root mean square error
               of pairwise distance matrices of x and x_hat
    """
    L, B, D = x_hat.size()
    assert D == 3

    # (B, L, D)
    x = x.permute(1, 0, 2)
    x_hat = x_hat.permute(1, 0, 2)
    mask = mask.permute(1, 0)

    # loop over each batch
    drmsd = torch.zeros(1, dtype=x.dtype, device=x.device)
    for i in range(B):
        mask_i = mask[i].view(-1, 1)
        L_i = torch.sum(mask_i)

        x_hat_i = torch.masked_select(x_hat[i], mask_i).view(-1, D)
        x_i = torch.masked_select(x[i], mask_i).view(-1, D)

        # use custom pdist function (see https://github.com/pytorch/pytorch/issues/25045)
        delta = pdist(x_hat_i) - pdist(x_i)
        drmsd += delta.pow(2.0).sum().div(L_i * (L_i - 1.0)).pow(0.5)

    return drmsd / B

def internal_coords(coords, pad=False):
    """Convert from cartesian to internal coordinates
    Output shape depends on pad parameter

    For a set of points ABCD:
        * the bond lengths (r) are the lengths of the vectors AB, BC, CD
        * the bond angle (theta) are the angles between the vectors (AB, BC) and (BC, CD)
        * the torsions (phi) are the angles between the planes spaned by ABC and BCD

    Args:
        coords: (N, *, 3)

    Returns:
        r: (N|(N-1), *)
        theta: (N|(N-2), *)
        phi: (N|(N-3), *)
    """

    assert coords.size(-1) == 3

    delta = coords[1:] - coords[:-1]
    r = delta.norm(dim=-1)
    theta = torch.acos(torch.sum(delta[:-1] * delta[1:], dim=-1) / (r[:-1] * r[1:]))

    a = delta[:-2]
    b = delta[1:-1]
    c = delta[2:]

    # torsion angle is angle from axb to bxc counterclockwise around b
    # see https://www.math.fsu.edu/~quine/MB_10/6_torsion.pdf

    axb = torch.cross(a, b, dim=-1)
    bxc = torch.cross(b, c, dim=-1)

    # orthogonal basis in plane perpendicular to b
    u1 = axb
    u2 = torch.cross(b / b.norm(dim=-1).unsqueeze(-1), axb, dim=-1)

    x = torch.sum(bxc * u1, dim=-1)
    y = torch.sum(bxc * u2, dim=-1)
    phi = torch.atan2(y, x)

    # internal coords are invariant under total translation/rotation (6 dof)
    # pad so that we can reconstruct modulo symmetries from arbitrary initialization coordinates
    if pad:
        r = torch.cat((r[0].repeat(1, 1), r), dim=0)
        theta = torch.cat((theta[0].repeat(2, 1), theta), dim=0)
        phi = torch.cat((phi[0].repeat(3, 1), phi), dim=0)

    return r, theta, phi

def internal_to_srf(r, theta, phi):
    """Compte SRF coordinates from internal coordinates (r, theta, phi)

    Args:
        r: (L, *)
        theta: (L, *)
        phi: (L, *)

    Returns:
        c_tilde: (L, *, 3)
    """
    assert r.size() == theta.size() == phi.size()

    c_tilde = torch.stack([r * torch.ones_like(phi) * torch.cos(theta),
                           r * torch.cos(phi) * torch.sin(theta),
                           r * torch.sin(phi) * torch.sin(theta)], dim=-1)
    return c_tilde

def torsion_to_srf(r, theta, phi):
    """Compute SRF coordinates from internal coordinates (r, theta, phi)
    where (r, theta) repeat every three bonds

    Args:
        r: (3,)
        theta: (3,)
        phi: (L, B, 3)
    """

    L, B, _ = phi.size()

    # (3,)
    r_cos_theta = r * torch.cos(theta)
    r_sin_theta = r * torch.sin(theta)

    # (D, L, B, 3)
    c_tilde = torch.stack([r_cos_theta.view(1, 1, -1).repeat(L, B, 1),
                           r_sin_theta * torch.cos(phi),
                           r_sin_theta * torch.sin(phi)])

    #(L, 3, B, D)
    c_tilde = c_tilde.permute(1, 3, 2, 0)

    # (3L, B, D)
    c_tilde = c_tilde.contiguous().view(3*L,B,3)
    return c_tilde

@torch.jit.script
def nerf_rot(A, B, C):
    """Compute nerf rotation matrix from previous three coordinates

    Args:
        A, B, C: (N, *, 3) cartesian coordinates of previous three points in chain

    Returns:
        R: (N, *, 3, 3)
    """

    BC = C - B
    bc = BC / BC.norm(dim=-1, p=2, keepdim=True)

    AB = B - A

    N = torch.cross(AB, bc, dim=-1)
    n = N / N.norm(dim=-1, p=2, keepdim=True)

    # (N, *, 3, 3)
    R = torch.stack([bc, torch.cross(n, bc, dim=-1), n], dim=-1)

    return R

@torch.jit.script
def nerf_extend(init_coords, c_tilde):
    """Compute cartesian coordinates from SRF coordinates

    Args:
        init_coords: (N, *, 3) initialization coordinates
        c_tilde: (L, *, 3)

    Returns:
        coords: (L, *, 3)
    """

    assert init_coords.size(0) >= 3

    L = c_tilde.size(0)

    coords : List[Tensor] = []

    A, B, C = init_coords[-3], init_coords[-2], init_coords[-1]
    for i in range(L):
        R = nerf_rot(A, B, C)
        D = C + torch.matmul(R, c_tilde[i].unsqueeze(-1)).squeeze()
        coords += [D]
        A, B, C = B, C, D

    return torch.stack(coords, dim=0)


def nerf(c_tilde):
    """Compute cartesian coordinates from SRF coordinates

    Args:
        c_tilde: (L, B, 3)

    Returns:
        coords: (L, B, 3)
    """
    L, B, _ = c_tilde.size()
    init_coords = torch.eye(3, dtype=c_tilde.dtype, device=c_tilde.device).unsqueeze(1).repeat(1, B, 1)

    coords = nerf_extend(init_coords, c_tilde)

    return coords

def pnerf(c_tilde, nfrag):
    """Compute cartesian coordinates from SRF coordinates

    Args:
        c_tilde: (L, B, 3)
        nfrags: number of fragments to processes in parallel

    Returns:
        coords: (L, B, 3)
    """
    L, B, _ = c_tilde.size()

    # frag_len = ceil(L / nfrags)
    frag_len = (L + nfrag - 1) // nfrag
    padding = frag_len * nfrag - L
    c_tilde = F.pad(c_tilde, [0, 0, 0, 0, 0, padding])
    assert c_tilde.size(0) % nfrag == 0

    #(L',F, B, D)
    c_tilde = c_tilde.view(nfrag, -1, B, 3)
    c_tilde = c_tilde.permute(1, 0, 2, 3)

    #initialization coordinates
    c0 = torch.tensor([-math.sqrt(1.0/2.0), math.sqrt(3.0/2.0), 0.0], dtype=c_tilde.dtype, device=c_tilde.device)
    c1 = torch.tensor([-math.sqrt(2.0), 0.0, 0.0], dtype=c_tilde.dtype, device=c_tilde.device)
    c2 = torch.tensor([0.0, 0.0, 0.0], dtype=c_tilde.dtype, device=c_tilde.device)

    #(3, D)
    init_coords = torch.stack([c0, c1, c2])

    #(3, F, B, D)
    init_coords = init_coords.view(3, 1, 1, 3).repeat(1, nfrag, B, 1)

    # (L', F, B, D)
    coords = nerf_extend(init_coords, c_tilde)

    # (F, L', B, D)
    coords = coords.permute(1, 0, 2, 3)

    # (L', B, D)
    coords_align = coords[-1]

    for i in reversed(range(nfrag-1)):
        A, B, C = coords[i, -3], coords[i, -2], coords[i, -1]

        # (1, B, D, D)
        R = nerf_rot(A, B, C).unsqueeze(0)

        # (L'', B, D, 1)
        coords_align = coords_align.unsqueeze(-1)

        # (L'', B, D)
        coords_align = torch.matmul(R, coords_align).squeeze() + C

        coords_align = torch.cat((coords[i], coords_align), dim=0)

    return coords_align[:L]

class GeometricUnit(nn.Module):
    """input -> torsion angles -> cartesian coords"""

    def __init__(self, input_size, linear_units = 20):
        super(GeometricUnit, self).__init__()

        self.linear = nn.Linear(input_size, linear_units)

        #initialize alphabet to random values between -pi and pi
        u = torch.distributions.Uniform(-math.pi, math.pi)
        self.alphabet = nn.Parameter(u.rsample(torch.Size([linear_units, 3])))

        # [C-N, N-CA, CA-C]
        self.bond_lengths = nn.Parameter(torch.tensor([132.868, 145.801, 152.326]))
        # [CA-C-N, C-N-CA, N-CA-C]
        self.bond_angles = nn.Parameter(torch.tensor([2.028, 2.124, 1.941]))

    def forward(self, inp):
        L, B = inp.size(0), inp.size(1)

        # (L, B, linear_units)
        x = F.softmax(self.linear(inp), dim=2)

        # (L, B, 3)
        sin = torch.matmul(x, torch.sin(self.alphabet))
        cos = torch.matmul(x, torch.cos(self.alphabet))

        # (L, B, 3)
        phi = torch.atan2(sin, cos)

        # (3L, B, 3)
        c_tilde = torsion_to_srf(self.bond_lengths, self.bond_angles, phi)

        # (3L, B, 3)
        coords = pnerf(c_tilde, nfrag=6)

        return coords
