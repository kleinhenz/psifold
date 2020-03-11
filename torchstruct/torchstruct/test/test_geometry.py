import torch
import torchstruct
import torch.nn.functional as F

from torchstruct.util import internal_coords, cartesian_coords

def test_geometry():
    coords = torch.rand(100, 3).double()
    dist = F.pdist(coords)

    l, theta, phi = internal_coords(coords)
    coords_ = cartesian_coords(l, theta, phi)
    dist_ = F.pdist(coords_)

    delta = dist - dist_
    err = delta.norm()
    assert err < 1e-10
