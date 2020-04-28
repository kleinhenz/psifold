import torch
import torch.nn.functional as F

import psifold
from psifold import internal_coords, internal_to_srf, dRMSD, pnerf, nerf

def test_nerf_RMSD():
    """test that internal_coords -> nerf is the identity when initialized with first three true coordinates"""
    batch_size = 10
    coords = torch.rand(100, batch_size, 3).double()

    r, theta, phi = internal_coords(coords)

    init_coords = coords[:3]
    c_tilde = internal_to_srf(r[2:], theta[1:], phi)
    coords_ = psifold.geometry.nerf_extend(init_coords, c_tilde)
    coords_ = torch.cat((init_coords, coords_), dim=0)

    err = torch.norm(coords - coords_)
    assert err < 1e-10

def test_nerf_dRMSD():
    """test that internal_coords -> nerf is the identity up to overall rigid transformation"""
    batch_size = 10
    coords = torch.rand(100, batch_size, 3).double()

    r, theta, phi = internal_coords(coords, pad=True)
    c_tilde = internal_to_srf(r, theta, phi)
    coords_ = nerf(c_tilde)
    err = dRMSD(coords_, coords)

    assert err < 1e-10

def test_pnerf_forward():
    """test pnerf forward pass matches nerf"""

    batch_size = 10
    coords = torch.rand(100, batch_size, 3).double()

    r, theta, phi = internal_coords(coords, pad=True)
    c_tilde = internal_to_srf(r, theta, phi)

    coords0 = nerf(c_tilde)
    coords1 = pnerf(c_tilde, nfrag=7)

    err = torch.norm(coords0 - coords1)

    assert err < 1e-10

def test_pnerf_backward():
    """test pnerf backward pass matches nerf"""

    batch_size = 32
    L = 100

    coords = torch.rand(L, batch_size, 3).double()
    r, theta, phi = internal_coords(coords, pad=True)
    c_tilde = internal_to_srf(r, theta, phi)

    c_tilde0 = c_tilde.detach().clone()
    c_tilde1 = c_tilde.detach().clone()

    c_tilde0.requires_grad = True
    coords0 = nerf(c_tilde0)
    coords0.backward(torch.ones_like(coords0))
    grad0 = c_tilde0.grad

    c_tilde1.requires_grad = True
    coords1 = pnerf(c_tilde1, nfrag=7)
    coords1.backward(torch.ones_like(coords1))
    grad1 = c_tilde1.grad

    err = torch.norm(grad0 - grad1)

    assert err < 1e-10

def test_procrustes():
    X = torch.rand(10, 3).double()
    R, _, _ = torch.svd(torch.rand(3, 3).double())
    Y = torch.matmul(X, R) + torch.rand(3)
    Z = psifold.geometry.procrustes(X, Y)
    err = torch.norm(X - Z)
    assert err < 1e-10
