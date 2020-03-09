import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def compute_internal_coords(coords):
    # torsion angle is angle from axb to bxc counterclockwise around b
    # see https://www.math.fsu.edu/~quine/MB_10/6_torsion.pdf

    delta = coords[1:] - coords[:-1]
    bond_length = delta.norm(dim=1)
    bond_angle = math.pi - torch.acos(torch.sum(delta[:-1] * delta[1:], dim=1) / (bond_length[:-1] * bond_length[1:]))

    a = delta[:-2]
    b = delta[1:-1]
    c = delta[2:]

    axb = torch.cross(a, b, dim=1)
    bxc = torch.cross(b, c, dim=1)

    # orthogonal basis in plane perpendicular to b
    u1 = axb
    u2 = torch.cross(b / b.norm(dim=1).unsqueeze(1), axb, dim=1)

    x = torch.sum(bxc * u1, dim=1)
    y = torch.sum(bxc * u2, dim=1)
    torsion = torch.atan2(y, x)

    return bond_length, bond_angle, torsion

# adapted from https://github.com/conradry/pytorch-rgn/blob/master/model.py
def compute_cartesian_coords(bond_length, bond_angle, torsion, initial):
    r = bond_length[2:]
    theta = bond_angle[1:]
    phi = torsion

    assert r.size() == theta.size() == phi.size()

    # note negative sign missing from paper
    c_tilde = torch.stack([-r * torch.ones(phi.size()) * torch.cos(theta),
                            r * torch.cos(phi) * torch.sin(theta),
                            r * torch.sin(phi) * torch.sin(theta)])

    coords = initial
    for i in range(r.size(0)):
        A, B, C = coords[-3, :], coords[-2, :], coords[-1, :]

        BC = C - B
        bc = BC / BC.norm()

        AB = B - A
        ab = AB / AB.norm()

        N = torch.cross(AB, bc)
        n = N / N.norm()

        R = torch.stack([bc, torch.cross(n, bc), n], dim=1)

        D = torch.matmul(R, c_tilde[:,i]) + C

        coords = torch.cat([coords, D.view(1, 3)])

    return coords
