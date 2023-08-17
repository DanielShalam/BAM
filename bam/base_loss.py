import wandb
import diffdist
import argparse
import torch
import numpy as np
import torch.distributed as dist

import utils


class BaseLoss(torch.nn.Module):

    def __init__(self, args: argparse):
        super().__init__()

        self.loss_fn = args.loss_fn
        self.reg = args.reg
        self.temperature = args.temperature
        self.num_sink_iter = args.num_sink_iter
        self.bs = args.batch_size_per_gpu
        self.large_crops = args.global_crops_number
        self.num_crops = args.global_crops_number + args.local_crops_number
        self.rank = utils.get_rank()
        self.world_size = args.world_size
        self.mask = None
        num_rows = self.bs * self.num_crops
        self.mask = self._make_pos_mask(num_rows=num_rows, num_cols=num_rows * self.world_size)
        self.mask_value = -500
        self._log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.log = False

        print(
            f"\n-> Initialized Loss function...\n"
            f"\n\t-> Loss: {self.loss_fn},\n"
            f"\n\t-> Lambda: {self.reg},\n"
            f"\n\t-> Rank={self.rank},\n"
            f"\n\t-> World Size: {self.world_size},\n"
            f"\n\t-> N Large crops: {self.large_crops},\n"
            f"\n\t-> N Total crops: {self.num_crops}\n"
        )

    def forward(self, z: torch.Tensor, targets=None):
        raise NotImplementedError

    def on_epoch_start(self, epoch):
        raise NotImplementedError

    def _make_pos_mask(self, num_rows, num_cols):

        """
        Positive-masking
        """

        if self.mask is None or self.mask.shape[0] != num_rows:
            effective_bs = self.bs * self.world_size
            mask = torch.zeros((self.bs, num_cols))
            crop_range = effective_bs * torch.arange(self.num_crops)
            local_offset = self.rank * self.bs
            for i in range(self.bs):
                mask[i, crop_range + local_offset + i] = 1

            mask = mask.repeat((self.num_crops, 1))
            self.mask = mask.bool().cuda(non_blocking=True)

        return self.mask

    def gather_tensors(self, z: torch.Tensor, num_crops: int, world_size: int) -> torch.Tensor:
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if dist.is_available() and dist.is_initialized():
            z_list = [torch.zeros_like(z) for _ in range(world_size)]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            # dist.all_gather(z_list, z)
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(num_crops)]
            # will be sorted as crop_1 from all gpus, crop_2 from all gpus ...
            z_sorted = []
            for m in range(num_crops):
                for i in range(world_size):
                    z_sorted.append(z_list[i * num_crops + m])

            z_gathered = torch.cat(z_sorted, dim=0)
        else:
            z_gathered = z

        return z_gathered
