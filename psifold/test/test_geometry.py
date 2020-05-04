import torch
import torch.nn.functional as F

import psifold
from psifold import internal_coords, internal_to_srf, dRMSD_masked, pnerf, nerf

def test_nerf_RMSD():
    """test that internal_coords -> nerf is the identity when initialized with first three true coordinates"""
    batch_size = 10
    coords = torch.rand(100, batch_size, 3).double()

    r, theta, phi = internal_coords(coords)

    init_coords = coords[:3]
    srf = internal_to_srf(r[2:], theta[1:], phi)
    coords_ = psifold.geometry.nerf_extend(init_coords, srf)
    coords_ = torch.cat((init_coords, coords_), dim=0)

    err = torch.norm(coords - coords_)
    assert err < 1e-10

def test_nerf_dRMSD():
    """test that internal_coords -> nerf is the identity up to overall rigid transformation"""
    batch_size = 10
    coords = torch.rand(100, batch_size, 3).double()

    r, theta, phi = internal_coords(coords, pad=True)
    srf = internal_to_srf(r, theta, phi)
    coords_ = nerf(srf)
    mask = torch.ones(100, batch_size, dtype=torch.bool)
    err = dRMSD_masked(coords_, coords, mask)

    assert err < 1e-10

def test_pnerf_forward():
    """test pnerf forward pass matches nerf"""

    batch_size = 10
    coords = torch.rand(100, batch_size, 3).double()

    r, theta, phi = internal_coords(coords, pad=True)
    srf = internal_to_srf(r, theta, phi)

    coords0 = nerf(srf)
    coords1 = pnerf(srf, nfrag=7)

    err = torch.norm(coords0 - coords1)

    assert err < 1e-10

def test_pnerf_backward():
    """test pnerf backward pass matches nerf"""

    batch_size = 32
    L = 100

    coords = torch.rand(L, batch_size, 3).double()
    r, theta, phi = internal_coords(coords, pad=True)
    srf = internal_to_srf(r, theta, phi)

    srf0 = srf.detach().clone()
    srf1 = srf.detach().clone()

    srf0.requires_grad = True
    coords0 = nerf(srf0)
    coords0.backward(torch.ones_like(coords0))
    grad0 = srf0.grad

    srf1.requires_grad = True
    coords1 = pnerf(srf1, nfrag=7)
    coords1.backward(torch.ones_like(coords1))
    grad1 = srf1.grad

    err = torch.norm(grad0 - grad1)

    assert err < 1e-10

def test_procrustes():
    batch_size = 32
    L = 100

    X = torch.rand(L, batch_size, 3).double()
    R, _, _ = torch.svd(torch.rand(batch_size, 3, 3).double())
    Y = torch.matmul(X.permute(1, 0, 2), R).permute(1, 0, 2) + torch.rand(batch_size, 3)

    Z = psifold.geometry.procrustes(X, Y)
    err = torch.norm(X - Z)

    assert err < 1e-10
