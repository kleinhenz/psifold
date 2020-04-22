import torch
import torch.nn.functional as F

import psifold
from psifold import internal_coords, internal_to_srf, dRMSD, pnerf

def test_geometric_unit_RMSD():
    """test that internal_coords -> nerf_extend_multi is the identity
    when initialized with first three true coordinates
    """
    batch_size = 10
    coords = torch.rand(100, batch_size, 3).double()

    r, theta, phi = internal_coords(coords)

    init_coords = coords[:3]
    c_tilde = internal_to_srf(r[2:], theta[1:], phi)
    coords_ = psifold.geometry.nerf_extend(init_coords, c_tilde)
    coords_ = torch.cat((init_coords, coords_), dim=0)

    err = torch.norm(coords - coords_)
    assert err < 1e-10

def test_geometric_unit_dRMSD():
    """ test that internal_coords -> nerf_extend_multi is the identity up to
    overall translation + rotation from ambiguity in the initial coordinates
    """
    batch_size = 10
    coords = torch.rand(100, batch_size, 3).double()

    r, theta, phi = internal_coords(coords, pad=True)
    c_tilde = internal_to_srf(r, theta, phi)
    coords_ = pnerf(c_tilde, nfrag=7)
    err = dRMSD(coords_, coords)

    assert err < 1e-10
