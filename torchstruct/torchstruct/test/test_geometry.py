import torch
import torchstruct

from torchstruct.util import compute_internal_coords, compute_cartesian_coords

def test_geometry():
    coords = torch.rand(100, 3, dtype=torch.float64)
    l, theta, phi = compute_internal_coords(coords)
    reconstructed_coords = compute_cartesian_coords(l, theta, phi, coords[:3, :])
    delta = coords - reconstructed_coords
    err = delta.norm()
    assert err < 1e-10

