import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _procrustes_batch_first(X, Y):
    """
    align Y with X

    Args:
        X: (B, L, D)
        Y: (B, L, D)

    Returns:
        Z: (B, L, D)
    """

    mu_X = torch.mean(X, dim=1, keepdim=True)
    mu_Y = torch.mean(Y, dim=1, keepdim=True)

    X0 = X - mu_X
    Y0 = Y - mu_Y

    # (B, D, D)
    M = torch.matmul(X0.transpose(1,2), Y0)
    U, S, V = torch.svd(M)

    # (B, D, D)
    R = torch.matmul(V, U.transpose(1, 2))

    Z = torch.matmul(Y0, R) + mu_X

    return Z

def procrustes(X, Y):
    """
    align Y with X

    Args:
        X: (L, B/0, D)
        Y: (L, B/0, D)

    Returns:
        Z: (L, B/0, D)
    """

    assert X.ndim == Y.ndim
    assert X.ndim == 2 or X.ndim == 3

    if X.ndim == 2:
        X = X.unsqueeze(0)
        Y = Y.unsqueeze(0)
        Z = _procrustes_batch_first(X, Y)
        return Z.squeeze(0)

    elif X.ndim == 3:
        # (B, L, D)
        X = X.permute(1, 0, 2)
        Y = Y.permute(1, 0, 2)
        Z = _procrustes_batch_first(X, Y)

        return Z.permute(1, 0, 2)

def pdist(x):
    return torch.norm((x - x[:, None]), p=2.0, dim=2)

@torch.jit.script
def dRMSD(X, Y):
    """Compute dRMSD loss

    Args:
        X: (L, D)
        Y: (L, D)
    """
    assert X.ndim == Y.ndim == 2
    assert X.size() == Y.size()

    L, _ = X.size()

    delta = pdist(X) - pdist(Y)
    return delta.pow(2.0).sum().div(L * (L - 1.0)).pow(0.5)

@torch.jit.script
def dRMSD_masked(X, Y, mask):
    """Compute dRMSD loss

    Args:
        X: (L, B, D)
        Y: (L, B, D)
        mask: (L, B)
    """
    L, B, D = X.size()
    assert D == 3

    # (B, L, D)
    X = X.permute(1, 0, 2)
    Y = Y.permute(1, 0, 2)
    mask = mask.permute(1, 0)

    # loop over each batch
    drmsd = torch.zeros(1, dtype=X.dtype, device=X.device)
    for i in range(B):
        Xi = torch.masked_select(X[i], mask[i].unsqueeze(1)).view(-1, D)
        Yi = torch.masked_select(Y[i], mask[i].unsqueeze(1)).view(-1, D)
        drmsd += dRMSD(Xi, Yi)

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

    sz = coords.size()
    assert sz[-1] == 3

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
        # NOTE can't pad r with zeros otherwise nerf_rot matrix becomes undefined
        r_sz = (1,) + sz[1:-1]
        r_pad = torch.tensor([1], dtype=coords.dtype, device=coords.device).repeat(*r_sz)
        r = torch.cat((r_pad, r), dim=0)

        # NOTE can't pad theta with zeros otherwise nerf_rot matrix becomes undefined (can't have AB || BC)
        theta_sz = (2,) + sz[1:-1]
        theta_pad = torch.tensor([math.pi/2], dtype=coords.dtype, device=coords.device).repeat(*theta_sz)
        theta = torch.cat((theta_pad, theta), dim=0)

        phi_sz = (3,) + sz[1:-1]
        phi_pad = torch.tensor([0], dtype=coords.dtype, device=coords.device).repeat(*phi_sz)
        phi = torch.cat((phi_pad, phi), dim=0)

    return r, theta, phi

def internal_to_srf(r, theta, phi):
    """Compte SRF coordinates from internal coordinates (r, theta, phi)

    Args:
        r: (L, *)
        theta: (L, *)
        phi: (L, *)

    Returns:
        srf: (L, *, 3)
    """
    assert r.size() == theta.size() == phi.size()

    srf = torch.stack([r * torch.ones_like(phi) * torch.cos(theta),
                       r * torch.cos(phi) * torch.sin(theta),
                       r * torch.sin(phi) * torch.sin(theta)], dim=-1)
    return srf

def torsion_to_srf(r, theta, phi):
    """Compute SRF coordinates from internal coordinates (r, theta, phi)
    where (r, theta) repeat every three bonds

    Args:
        r: (3,)
        theta: (3,)
        phi: (L, B, 3)

    Returns:
        srf: (3L, B, 3)
    """

    L, B, _ = phi.size()

    # (3,)
    r_cos_theta = r * torch.cos(theta)
    r_sin_theta = r * torch.sin(theta)

    # (D, L, B, 3)
    srf = torch.stack([r_cos_theta.view(1, 1, -1).repeat(L, B, 1),
                       r_sin_theta * torch.cos(phi),
                       r_sin_theta * torch.sin(phi)])

    #(L, 3, B, D)
    srf = srf.permute(1, 3, 2, 0)

    # (3L, B, D)
    srf = srf.contiguous().view(3*L,B,3)
    return srf

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
def nerf_extend(init_coords, srf):
    """Compute cartesian coordinates from SRF coordinates

    Args:
        init_coords: (N, *, 3) initialization coordinates
        srf: (L, *, 3)

    Returns:
        coords: (L, *, 3)
    """

    assert init_coords.size(0) >= 3

    L = srf.size(0)

    coords : List[Tensor] = []

    A, B, C = init_coords[-3], init_coords[-2], init_coords[-1]
    for i in range(L):
        R = nerf_rot(A, B, C)
        D = C + torch.matmul(R, srf[i].unsqueeze(-1)).squeeze(-1)
        coords += [D]
        A, B, C = B, C, D

    return torch.stack(coords, dim=0)


def nerf(srf):
    """Compute cartesian coordinates from SRF coordinates

    Args:
        srf: (L, B, 3)

    Returns:
        coords: (L, B, 3)
    """
    L, B, _ = srf.size()
    #initialization coordinates
    c0 = torch.tensor([-math.sqrt(1.0/2.0), math.sqrt(3.0/2.0), 0.0], dtype=srf.dtype, device=srf.device)
    c1 = torch.tensor([-math.sqrt(2.0), 0.0, 0.0], dtype=srf.dtype, device=srf.device)
    c2 = torch.tensor([0.0, 0.0, 0.0], dtype=srf.dtype, device=srf.device)

    #(3, D)
    init_coords = torch.stack([c0, c1, c2])

    #(3, B, D)
    init_coords = init_coords.view(3, 1, 3).repeat(1, B, 1)

    coords = nerf_extend(init_coords, srf)

    return coords

def pnerf(srf, nfrag):
    """Compute cartesian coordinates from SRF coordinates

    Args:
        srf: (L, B, 3)
        nfrags: number of fragments to processes in parallel

    Returns:
        coords: (L, B, 3)
    """
    L, B, _ = srf.size()

    # frag_len = ceil(L / nfrags)
    frag_len = (L + nfrag - 1) // nfrag
    padding = frag_len * nfrag - L

    # NOTE we can't pad with zero otherwise backward gives nan
    # also see https://github.com/pytorch/pytorch/issues/31734 (fixed in pytorch 1.5)
    srf = F.pad(srf, [0, 0, 0, 0, 0, padding], value=0.1)
    assert srf.size(0) % nfrag == 0

    #(L',F, B, D)
    srf = srf.view(nfrag, -1, B, 3)
    srf = srf.permute(1, 0, 2, 3)

    #initialization coordinates
    c0 = torch.tensor([-math.sqrt(1.0/2.0), math.sqrt(3.0/2.0), 0.0], dtype=srf.dtype, device=srf.device)
    c1 = torch.tensor([-math.sqrt(2.0), 0.0, 0.0], dtype=srf.dtype, device=srf.device)
    c2 = torch.tensor([0.0, 0.0, 0.0], dtype=srf.dtype, device=srf.device)

    #(3, D)
    init_coords = torch.stack([c0, c1, c2])

    #(3, F, B, D)
    init_coords = init_coords.view(3, 1, 1, 3).repeat(1, nfrag, B, 1)

    # (L', F, B, D)
    coords = nerf_extend(init_coords, srf)

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
        coords_align = torch.matmul(R, coords_align).squeeze(-1) + C

        coords_align = torch.cat((coords[i], coords_align), dim=0)

    return coords_align[:L]
