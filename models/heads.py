from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class MlpProjHead(nn.Module):
    """
    Projector Head based on MLPs.
    We *disable bias* of all Linear layers.
    """

    def __init__(self, input_dim, n_layers, hidden_dim=4096, output_dim=4096, use_bn=True, last_bn=True,
                 bias=True, activation="relu"):

        super().__init__()
        self.projector = self._build_mlp(n_layers, input_dim, hidden_dim, output_dim,
                                         use_bn, last_bn, bias, activation)
        self.apply(self._init_weights)
        print(self.projector)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, use_bn, last_bn,
                   use_bias=True, activation="relu"):
        mlp = []
        for i in range(num_layers):
            dim1 = input_dim if i == 0 else mlp_dim
            dim2 = output_dim if i == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=use_bias))

            if i < num_layers - 1:
                if use_bn:
                    mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True))
            elif use_bn and last_bn:
                # no affine for last bn
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.projector(x)


class DINOHead(nn.Module):
    """
    Projector Head as in DINO
    """

    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
