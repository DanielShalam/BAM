import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

from bam.sinkhorn import distributed_sinkhorn
from bam.base_loss import BaseLoss


class BAMLoss(BaseLoss):

    def __init__(self, args: argparse):
        super().__init__(args)

        entropy_warmup_epochs = 40
        self.reg_final = args.reg_final
        self.entropy_schedule = np.concatenate((
            np.linspace(self.reg, self.reg_final, entropy_warmup_epochs),
            np.ones(args.epochs - entropy_warmup_epochs) * self.reg_final
        ))
        print(f"\n-> Initialized entropy schedule from {self.reg}-{self.reg_final} "
              f"with warmup of {entropy_warmup_epochs} epochs")
        self.reg = self.entropy_schedule[-1]

    def on_epoch_start(self, epoch):
        self.reg = self.entropy_schedule[epoch]

    @torch.no_grad()
    def _sinkhorn(self, Q):
        assignments = Q / self.reg
        # use the log-sum-exp trick for numerical stability.
        M = torch.max(assignments)
        dist.all_reduce(M, op=dist.ReduceOp.MAX)
        assignments = assignments - M
        return distributed_sinkhorn(
            Q=torch.exp(assignments).t(), world_size=self.world_size,
            num_sink_iter=self.num_sink_iter
        )

    def _sinkhorn_loss(self, Q):

        # positive-masking
        Q[self.mask] = self.mask_value

        # target attention
        with torch.no_grad():
            P = self._sinkhorn(Q=Q.detach())[:self.bs * self.large_crops]

        # source attention
        Q = self._log_softmax(Q / self.temperature)

        # iterate over large crops and use them as the target distribution
        loss = 0.
        for crop_i in range(self.large_crops):
            sub_loss = 0
            n_sub = 0
            P_i = P[crop_i * self.bs: (crop_i + 1) * self.bs]

            for crop_j in range(self.num_crops):
                if crop_i == crop_j:
                    continue
                sub_loss -= (P_i * Q[crop_j * self.bs: (crop_j + 1) * self.bs]).sum(dim=-1).mean()
                n_sub += 1
            loss += (sub_loss / n_sub)
        return loss / self.large_crops

    def forward(self, z: torch.Tensor, targets=None):

        """
        compute BAM loss
        """

        z = F.normalize(z, p=2, dim=-1)
        z_gathered = self.gather_tensors(z, num_crops=self.num_crops, world_size=self.world_size)
        loss = self._sinkhorn_loss(Q=torch.mm(z, z_gathered.t()))   # compute loss
        return loss, None
