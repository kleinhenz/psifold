import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def internal_coords(coords):
    """Convert from cartesian to internal coordinates

    For a set of points ABCD:
        * the bond lengths (r) are the lengths of the vectors AB, BC, CD
        * the bond angle (theta) are the angles between the vectors (AB, BC) and (BC, CD)
        * the torsions (phi) are the angles between the planes spaned by ABC and BCD

    Args:
        coords: [N, batch, 3]

    Returns:
        r: [N-1, batch]
        theta: [N-2, batch]
        phi: [N-3, batch]
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

def extend(A, B, C, c_tilde):
    """Compute next cartesian coordinate in chain (ABC -> ABCD)

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

def geometric_unit(coords, r, theta, phi):
    """Extend chain of cartesian coordinates

    For a set of points ABCD:
        * the bond lengths (r) are the lengths of the vectors AB, BC, CD
        * the bond angle (theta) are the angles between the vectors (AB, BC) and (BC, CD)
        * the torsions (phi) are the angles between the planes spaned by ABC and BCD

    Args:
        coords: [N, batch, 3] cartesian coordinates of current chain (N >= 3)
        r, theta, phi: [M, batch] internal coordinates of points to be added to chain

    Returns:
        coords: [N+M, batch, 3] cartesian coordinates of extended chain
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
        D = extend(A, B, C, c_tilde[i])
        coords = torch.cat([coords, D.view(1, -1, 3)])

    return coords
