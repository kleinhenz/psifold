import torch
import torch.nn as nn
import torch.nn.functional as F

# adapted from https://github.com/conradry/pytorch-rgn/blob/master/model.py
def geometric_unit(pred_coords, pred_torsions, bond_angles, bond_lens):
    for i in range(3):
        A, B, C = pred_coords[-3], pred_coords[-2], pred_coords[-1]

        T = bond_angles[i]
        R = bond_lens[i]
        P = pred_torsions[:,i]

        D2 = torch.stack([-R * torch.ones(P.size())*torch.cos(T),
                           R * torch.cos(P) * torch.sin(T),
                           R * torch.sin(P) * torch.sin(T)], dim=1)

        BC = C - B
        bc = BC / torch.norm(BC, 2, dim=1, keepdim=True)

        AB = B - A

        N = torch.cross(AB, bc)
        M = torch.stack([bc, torch.cross(n, bc), n], dim=2)

        D = torch.bmm(M, D2.view(-1, 3, 1)).squeeze() + C
        pred_coords = torch.cat([pred_coords, D.view(1, -1, 3)])

    return pred_coords
