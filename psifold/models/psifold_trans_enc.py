import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import psifold

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PsiFoldTransformerEncoder(nn.Module):
    """
    PsiFold implementation
    """
    def __init__(self, hidden_size=512, n_layers=8, dropout=0.1, radius=3.806, ff_dim=None, nhead=None):
        super(PsiFoldTransformerEncoder, self).__init__()

        # default scaling for parameters
        # see https://arxiv.org/pdf/1908.08962.pdf
        if ff_dim is None:
            ff_dim = hidden_size * 4

        if nhead is None:
            assert hidden_size % 64 == 0
            nhead = hidden_size // 64

        # save info needed to recreate model from checkpoint
        self.model_name = "psifold_transformer_encoder"
        self.model_args = {"hidden_size" : hidden_size,
                           "ff_dim" : ff_dim,
                           "nhead" : nhead,
                           "n_layers": n_layers,
                           "dropout": dropout}

        self.fc0 = nn.Linear(41, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc1 = nn.Linear(hidden_size, 2)

        self.radius = nn.Parameter(torch.tensor([radius]))
        self.radius.requires_grad = False

    def forward(self, batch):
        seq = batch["seq"] # (L x B)
        pssm = batch["pssm"] # (L x B x 21)
        length = batch["length"] # (L,)

        L, B = seq.size()

        # (L x B x 20)
        seq = F.one_hot(seq, 20).type(pssm.dtype)

        # (L x B x (20 + 21))
        encoder_in = torch.cat((seq, pssm), dim=2)

        # (L x B x hidden_size)
        encoder_in = self.fc0(encoder_in)

        # (L x B x hidden_size)
        encoder_in = self.pos_encoder(encoder_in)

        # (L x B)
        mask = torch.arange(L, device=seq.device).expand(B, L) >= length.unsqueeze(1)

        # (L x B x hidden_size)
        encoder_out = self.encoder(encoder_in, src_key_padding_mask=mask)

        # (L x B x 2)
        angles = self.fc1(encoder_out)

        # (L x B)
        theta = angles[:,:,0]
        phi = angles[:,:,1]

        # (L x B)
        r_cos_theta = self.radius * torch.cos(theta)
        r_sin_theta = self.radius * torch.sin(theta)

        # (L x B x 3)
        srf = torch.stack([r_cos_theta,
                           r_sin_theta * torch.cos(phi),
                           r_sin_theta * torch.sin(phi)], dim=-1)

        return srf

#        # (L x B x 3)
#        x = self.fc1(encoder_out)
#
#
#        # (L x B x 3)
#        srf = self.radius * x / x.norm(dim=-1).unsqueeze(-1)
#
#        return srf
