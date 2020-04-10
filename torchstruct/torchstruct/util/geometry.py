import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def dRMSD(x_hat, x, mask, length_norm=False):
    """Compute dRMSD loss

    Args:
        x_hat: (L, B, D)
        x: (L, B, D)
        mask: (L, B)
        length_norm: (bool) divide dRMSD by L(L - 1)

    Returns:
        loss: (scalar)
    """

    L, B, D = x_hat.size()
    assert D == 3

    loss = []
    for i in range(B):
        mask_i = mask[:, i].view(-1, 1)
        Li = mask_i.sum()

        x_hat_i = torch.masked_select(x_hat[:, i, :], mask_i).view(-1, D)
        x_i = torch.masked_select(x[:, i, :], mask_i).view(-1, D)

        d_hat_i = F.pdist(x_hat_i)
        d_i = F.pdist(x_i)

        loss_i = torch.norm(d_hat_i - d_i)
        if length_norm:
            loss_i = loss_i / (Li.pow(2) - Li)

        loss.append(loss_i)

    return sum(loss) / B

def internal_coords(coords):
    """Convert from cartesian to internal coordinates

    For a set of points ABCD:
        * the bond lengths (r) are the lengths of the vectors AB, BC, CD
        * the bond angle (theta) are the angles between the vectors (AB, BC) and (BC, CD)
        * the torsions (phi) are the angles between the planes spaned by ABC and BCD

    Args:
        coords: (N, batch, 3)

    Returns:
        r: (N-1, batch)
        theta: (N-2, batch)
        phi: (N-3, batch)
    """

    assert coords.ndim == 3

    delta = coords[1:] - coords[:-1]
    bond_length = delta.norm(dim=2)
    bond_angle = torch.acos(torch.sum(delta[:-1] * delta[1:], dim=2) / (bond_length[:-1] * bond_length[1:]))

    a = delta[:-2]
    b = delta[1:-1]
    c = delta[2:]

    # torsion angle is angle from axb to bxc counterclockwise around b
    # see https://www.math.fsu.edu/~quine/MB_10/6_torsion.pdf

    axb = torch.cross(a, b, dim=2)
    bxc = torch.cross(b, c, dim=2)

    # orthogonal basis in plane perpendicular to b
    u1 = axb
    u2 = torch.cross(b / b.norm(dim=2).unsqueeze(2), axb, dim=2)

    x = torch.sum(bxc * u1, dim=2)
    y = torch.sum(bxc * u2, dim=2)
    torsion = torch.atan2(y, x)

    return bond_length, bond_angle, torsion

def nerf_extend_single(A, B, C, c_tilde):
    """Compute next cartesian coordinate in chain (ABC -> ABCD)
    using natural extension reference frame (nerf)

    Args:
        A, B, C: [batch, 3] cartesian coordinates of previous three points in chain
        c_tilde: [batch, 3] SRF (special reference frame) coordinates of next point in chain

    Returns:
        D: [batch, 3] cartesian coordinates of next point in chain
    """

    BC = C - B
    bc = BC / BC.norm(dim=1, keepdim=True)

    AB = B - A
    ab = AB / AB.norm(dim=1, keepdim=True)

    N = torch.cross(AB, bc, dim=1)
    n = N / N.norm(dim=1, keepdim=True)

    R = torch.stack([bc, torch.cross(n, bc), n], dim=2)

    D = C + torch.bmm(R, c_tilde.view(-1, 3, 1)).squeeze()

    return D

def nerf_extend_multi(coords, r, theta, phi):
    """Extend chain of cartesian coordinates
    using natural extension references frame (nerf)

    For a set of points ABCD:
        * the bond lengths (r) are the lengths of the vectors AB, BC, CD
        * the bond angle (theta) are the angles between the vectors (AB, BC) and (BC, CD)
        * the torsions (phi) are the angles between the planes spaned by ABC and BCD

    Args:
        coords: (N, batch, 3) cartesian coordinates of current chain (N >= 3)
        r, theta, phi: (M, batch) internal coordinates of points to be added to chain

    Returns:
        coords: (N+M, batch, 3) cartesian coordinates of extended chain
    """

    assert r.size() == theta.size() == phi.size()
    N = r.size(0)

    # compute SRF (special reference frame) coordinates
    c_tilde = torch.stack([r * torch.ones(phi.size()) * torch.cos(theta),
                           r * torch.cos(phi) * torch.sin(theta),
                           r * torch.sin(phi) * torch.sin(theta)], dim=2)

    # extend chain
    for i in range(N):
        A, B, C = coords[-3], coords[-2], coords[-1]
        D = nerf_extend_single(A, B, C, c_tilde[i])
        coords = torch.cat([coords, D.view(1, -1, 3)])

    return coords

class GeometricUnit(nn.Module):
    """input -> torsion angles -> cartesian coords"""

    def __init__(self, input_size, linear_units = 20):
        super(GeometricUnit, self).__init__()

        self.linear = nn.Linear(input_size, linear_units)

        #initialize alphabet to random values between -pi and pi
        u = torch.distributions.Uniform(-math.pi, math.pi)
        self.alphabet = nn.Parameter(u.rsample(torch.Size([linear_units, 3])))

        # TODO make bond lengths/angles parameters instead of constant?

        # [C-N, N-CA, CA-C]
        self.bond_lengths = torch.tensor([132.868, 145.801, 152.326])
        # [CA-C-N, C-N-CA, N-CA-C]
        self.bond_angles = torch.tensor([2.028, 2.124, 1.941])

    def forward(self, inp):
        L, B = inp.size(0), inp.size(1)

        # (L x B x linear_units)
        x = F.softmax(self.linear(inp), dim=2)

        # (L x B x 3)
        sin = torch.matmul(x, torch.sin(self.alphabet))
        cos = torch.matmul(x, torch.cos(self.alphabet))

        # (L x B x 3)
        phi = torch.atan2(sin, cos)

        # initial coords
        # (3 x B x 3)
        coords = torch.eye(3).unsqueeze(1).repeat(1, B, 1)

        # (3 x B)
        r = self.bond_lengths.unsqueeze(1).repeat(1, B)
        theta = self.bond_angles.unsqueeze(1).repeat(1, B)

        for i in range(L):
            coords = nerf_extend_multi(coords, r, theta, phi[i].transpose(0, 1))

        # ignore first 3 initialization coordinates
        return coords[3:, :, :]
