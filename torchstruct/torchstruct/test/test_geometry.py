import torch
import torchstruct
import torch.nn.functional as F

from torchstruct.util import internal_coords, nerf_extend_multi, dRMSD

def test_geometric_unit_RMSD():
    """test that internal_coords -> nerf_extend_multi is the identity
    when initialized with first three true coordinates
    """
    batch_size = 10
    coords = torch.rand(100, batch_size, 3).double()

    r, theta, phi = internal_coords(coords)
    coords_ = nerf_extend_multi(coords[:3], r[2:], theta[1:], phi)
    delta = coords - coords_
    err = delta.norm()
    assert err < 1e-10

def test_geometric_unit_dRMSD():
    """ test that internal_coords -> nerf_extend_multi is the identity up to
    overall translation + rotation from ambiguity in the initial coordinates
    """
    batch_size = 10
    coords = torch.rand(100, batch_size, 3).double()

    r, theta, phi = internal_coords(coords)

    # initial coordinates
    zero = torch.zeros(batch_size, dtype=coords.dtype)
    A = torch.stack([zero, zero, zero], dim=1)
    B = torch.stack([r[0], zero, zero], dim=1)
    C = B + torch.stack([r[1] * torch.cos(theta[0]), r[1] * torch.sin(theta[0]), zero], dim=1)
    coords_ = torch.stack([A, B, C], dim=0)
    coords_ = nerf_extend_multi(coords_, r[2:], theta[1:], phi)

    mask = torch.ones(coords.size(0), coords.size(1), dtype=torch.bool)
    err = dRMSD(coords_, coords, mask)

    assert err < 1e-10
