import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def internal_coords(coords):
    """Convert from cartesian to internal coordinates

    For a set of points ABCD:
        * the bond lengths are the lengths of the vectors AB, BC, CD
        * the bond angle are the angles between the vectors (AB, BC) and (BC, CD)
        * the torsions are the angles between the planes spaned by ABC and BCD

    Args:
        coords: [N, 3]

    Returns:
        bond_length: [N-1]
        bond_angle: [N-2]
        torsion: [N-3]
    """

    delta = coords[1:] - coords[:-1]
    bond_length = delta.norm(dim=1)
    bond_angle = torch.acos(torch.sum(delta[:-1] * delta[1:], dim=1) / (bond_length[:-1] * bond_length[1:]))

    a = delta[:-2]
    b = delta[1:-1]
    c = delta[2:]

    # torsion angle is angle from axb to bxc counterclockwise around b
    # see https://www.math.fsu.edu/~quine/MB_10/6_torsion.pdf

    axb = torch.cross(a, b, dim=1)
    bxc = torch.cross(b, c, dim=1)

    # orthogonal basis in plane perpendicular to b
    u1 = axb
    u2 = torch.cross(b / b.norm(dim=1).unsqueeze(1), axb, dim=1)

    x = torch.sum(bxc * u1, dim=1)
    y = torch.sum(bxc * u2, dim=1)
    torsion = torch.atan2(y, x)

    return bond_length, bond_angle, torsion

def extend(A, B, C, c_tilde):
    """Compute next cartesian coordinate in chain (ABC -> ABCD)

    Args:
        A, B, C: cartesian coordinates of previous three points in chain
        c_tilde: SRF (special reference frame) coordinates of next point in chain

    Returns:
        D: cartesian coordinates of next point in chain
    """

    BC = C - B
    bc = BC / BC.norm()

    AB = B - A
    ab = AB / AB.norm()

    N = torch.cross(AB, bc)
    n = N / N.norm()

    R = torch.stack([bc, torch.cross(n, bc), n], dim=1)

    D = C + torch.matmul(R, c_tilde)

    return D

def geometric_unit(coords, r, theta, phi):
    """Extend chain of cartesian coordinates

    Args:
        coords: [N, 3] cartesian coordinates of current chain (N >= 3)
        r, theta, phi: [M, 1] internal coordinates of points to be added to chain

    Returns:
        coords: [N+M, 3] cartesian coordinates of extended chain
    """

    assert r.size() == theta.size() == phi.size()
    N = r.size(0)

    # compute SRF (special reference frame) coordinates
    c_tilde = torch.stack([r * torch.ones(phi.size()) * torch.cos(theta),
                           r * torch.cos(phi) * torch.sin(theta),
                           r * torch.sin(phi) * torch.sin(theta)])

    # extend chain
    for i in range(N):
        A, B, C = coords[-3, :], coords[-2, :], coords[-1, :]
        D = extend(A, B, C, c_tilde[:,i])
        coords = torch.cat([coords, D.view(1, 3)])

    return coords

def cartesian_coords(bond_length, bond_angle, torsion):
    """
    Convert from internal to cartesian coordinates

    For a set of points ABCD:
        * the bond lengths are the lengths of the vectors AB, BC, CD
        * the bond angle are the angles between the vectors (AB, BC) and (BC, CD)
        * the torsions are the angles between the planes spaned by ABC and BCD

   This function places the first three coordinates ABC with
   * A at the origin
   * B on the positive x-axis
   * C in the xy-plane

    Args:
        bond_length: [N - 1]
        bond_angle: [N - 2]
        torsion: [N -3]

    Returns:
        coords: [N, 3]
    """
    # adapted from https://github.com/conradry/pytorch-rgn/blob/master/model.py

    # initial coordinates
    A = torch.zeros(3, dtype=bond_length.dtype)
    B = torch.tensor([bond_length[0], 0.0, 0.0])
    C = B + torch.tensor([bond_length[1] * torch.cos(bond_angle[0]), bond_length[1] * torch.sin(bond_angle[0]), 0.0])
    coords = torch.stack([A, B, C])

    r = bond_length[2:]
    theta = bond_angle[1:]
    phi = torsion

    coords = geometric_unit(coords, r, theta, phi)

    return coords
