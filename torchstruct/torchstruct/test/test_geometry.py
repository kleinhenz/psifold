import torch
import torchstruct
import torch.nn.functional as F

from torchstruct.util import internal_coords, geometric_unit

def test_geometric_unit_RMSD():
    batch_size = 10
    coords = torch.rand(100, batch_size, 3).double()

    r, theta, phi = internal_coords(coords)
    coords_ = geometric_unit(coords[:3], r[2:], theta[1:], phi)
    delta = coords - coords_
    err = delta.norm()
    assert err < 1e-10

def test_geometric_unit_dRMSD():
    batch_size = 10
    coords = torch.rand(100, batch_size, 3).double()
    dist = torch.stack([F.pdist(coords[:,i,:]) for i in range(coords.size(1))])

    r, theta, phi = internal_coords(coords)

    # initial coordinates
    zero = torch.zeros(batch_size, dtype=coords.dtype)
    A = torch.stack([zero, zero, zero], dim=1)
    B = torch.stack([r[0], zero, zero], dim=1)
    C = B + torch.stack([r[1] * torch.cos(theta[0]), r[1] * torch.sin(theta[0]), zero], dim=1)
    coords_ = torch.stack([A, B, C], dim=0)

    coords_ = geometric_unit(coords_, r[2:], theta[1:], phi)
    dist_ = torch.stack([F.pdist(coords_[:,i,:]) for i in range(coords.size(1))])

    delta = dist - dist_
    err = delta.norm()
    assert err < 1e-10
