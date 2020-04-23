#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

import psifold
from psifold import pnerf, internal_coords, internal_to_srf, dRMSD

def run(length, batch_size, device, epochs=250, sigma=1e-3, lr=1e-4):
    """
    test that backpropagation through pnerf and dRMSD works
    """

    coords = torch.rand(length, batch_size, 3)
    coords = coords.to(device)
    r, theta, phi = internal_coords(coords, pad=True)

    r = r + sigma * torch.randn_like(r)
    theta = theta + sigma * torch.randn_like(theta)
    phi = phi + sigma * torch.randn_like(phi)

    r.requires_grad = True
    theta.requires_grad = True
    phi.requires_grad = True

    optimizer = torch.optim.Adam([r, theta, phi], lr=lr)

    loss_history = []
    for epoch in tqdm(range(epochs)):
        c_tilde = internal_to_srf(r, theta, phi)
        coords_ = pnerf(c_tilde, nfrag=7)
        loss = dRMSD(coords_, coords)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    print(f"loss = {loss_history[-1]:0.2e}")
    loss_history = np.array(loss_history)
    fig, ax = plt.subplots()
    ax.semilogy(loss_history)
    ax.set_xlabel("iter")
    ax.set_ylabel("dRMSD")
    plt.show()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    length = 64
    batch_size = 8
    run(length, batch_size, device)

if __name__ == "__main__":
    main()
