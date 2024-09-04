from functools import partial

import diffdist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import FloatTensor
import numpy as np

from bam.masks import get_positive_mask, get_top_k_mask, get_top_p_mask


class BAMLoss(nn.Module):
    """
    Balanced-Attention Matching Loss.

    :param n_crops: total number of views.
    :param n_lrg_crops: number of large (global) views.
    :param n_tgt_crops: number of target views (defaults to n_lrg_crops).
    :param ot_reg_scheduler: a scheduler for the OT regularization (defaults to constant scheduler).
    :param softmax_temperature: the temperature of the source attention.
    :param num_sinkhorn_iters: number of sinkhorn iterations.
    :param top_p: if >0., we keep only the highest values for the target matrix, according to the top_p masking.
    :param top_k: if >0., we keep only the highest values for the target matrix, according to the top_k masking.
    :param positive_masking: if to apply positive-masking (defaults to True).
    """

    def __init__(self,
                 n_crops: int,
                 n_lrg_crops: int,
                 n_tgt_crops: int,
                 ot_reg_scheduler: np.ndarray,
                 softmax_temperature: float = 0.1,
                 num_sinkhorn_iters: int = 3,
                 top_p: float = 0.,
                 top_k: int = 0,
                 positive_masking: bool = True):
        super().__init__()

        self.n_crops = n_crops
        self.n_lrg_crops = n_lrg_crops
        self.n_tgt_crops = n_tgt_crops
        self.ot_reg_scheduler = ot_reg_scheduler
        self.softmax_temperature = softmax_temperature
        self.num_sinkhorn_iters = num_sinkhorn_iters
        self.positive_masking = positive_masking

        # TOP-K/P Masking
        self.top_k = top_k
        self.top_p = top_p
        self.masking_fn = None
        if self.top_k > 0:
            self.masking_fn = partial(get_top_k_mask, k=self.top_k, inverse=True)
        elif self.top_p > 0:
            self.masking_fn = partial(get_top_p_mask, is_distribution=True, threshold=self.top_p, inverse=True)

        print(f"BAM Loss initialized. ", f"\n\t-> OT Temperature:      {ot_reg_scheduler}, "
                                         f"\n\t-> Softmax Temperature: {softmax_temperature},"
                                         f"\n\t-> Num Sinkhorn iters:  {num_sinkhorn_iters},"
                                         f"\n\t-> Top k:               {self.top_k},"
                                         f"\n\t-> Top p:               {self.top_p},")
        self.pos_mask = None
        self._log_softmax = nn.LogSoftmax(dim=-1)

    @torch.no_grad()
    def _sinkhorn(self, Q: FloatTensor):
        """
        Sinkhorn
        """

        n, N = Q.size()
        assignments = (Q / self.ot_reg)
        if self.positive_masking:
            assignments.masked_fill_(self.pos_mask[:n, :N], torch.finfo(Q.dtype).min)

        # use the log-sum-exp trick for numerical stability.
        M = torch.max(assignments)
        dist.all_reduce(M, op=dist.ReduceOp.MAX)
        assignments = assignments - M

        return distributed_sinkhorn(
            Q=torch.exp(assignments).t(),
            world_size=dist.get_world_size(),
            num_sink_iter=self.num_sinkhorn_iters,
            n_masked=self.n_tgt_crops if self.positive_masking else 0,
        )

    def keep_top_values(self, logits: FloatTensor, n_crops: int, mask_value: float = 0.):
        assert self.masking_fn is not None
        # compute the average similarity matrix over the large crops
        num_mask = logits.size(-1) - self.top_k
        avg_logits = torch.stack(logits.chunk(n_crops, dim=0)).mean(0)
        logits.scatter_(-1, (-avg_logits).topk(num_mask, dim=-1).indices.repeat(n_crops, 1), mask_value)
        return logits

    def compute_loss(self, Q: FloatTensor):
        """
        Compute BAM Loss
        """
        num_large_crops = self.n_lrg_crops
        batch_size_per_crop = Q.size(0) // self.n_crops

        # positive-masking
        if self.pos_mask is None:   # compute the positive mask only once
            if self.positive_masking:
                self.pos_mask = get_positive_mask(batch_size_per_crop, n_crops=self.n_crops).detach()
            else:
                self.pos_mask = torch.zeros_like(Q).detach().bool()
        self.pos_mask = self.pos_mask[:, :Q.size(1)]

        # self-sinkhorn
        with torch.no_grad():
            P = self._sinkhorn(Q.detach())[:batch_size_per_crop * num_large_crops]  # use only large crops as target
            if self.masking_fn is not None:
                P = self.keep_top_values(P, n_crops=num_large_crops, mask_value=0.)
                P = P / P.sum(dim=-1, keepdim=True)
            P = P.chunk(num_large_crops)

        # self-softmax
        Q = (Q / self.softmax_temperature)
        if self.positive_masking:
            Q.masked_fill_(self.pos_mask, torch.finfo(Q.dtype).min)
        log_Q = self._log_softmax(Q).chunk(self.n_crops)

        loss_pos_mask = (~self.pos_mask[: batch_size_per_crop]).to(log_Q[0].dtype)

        # accumulate loss
        loss = 0.
        for v_tgt in range(num_large_crops):
            sub_loss, n_sub_loss = 0, 0
            for v_src in range(self.n_crops):
                if v_tgt == v_src and self.n_crops > 1:
                    continue
                # cross-entropy
                sub_loss -= ((P[v_tgt] * log_Q[v_src]) * loss_pos_mask).sum(dim=-1).mean()
                n_sub_loss += 1
            loss += (sub_loss / n_sub_loss)
        loss = loss / num_large_crops
        return loss

    def forward(self, z: FloatTensor, z_teacher: FloatTensor = None, epoch: int = None):
        self.ot_reg = self.ot_reg_scheduler[epoch] if epoch is not None else self.ot_reg[-1]
        bs_per_crop = z.size(0) // self.n_crops

        # l2-normalize + gather targets from gpus
        z = F.normalize(z, dim=-1)
        z_gathered = F.normalize(z_teacher, dim=-1) if z_teacher is not None else z
        z_gathered = gather_sort_tensors(z_gathered[:bs_per_crop * self.n_tgt_crops], n_crops=self.n_tgt_crops)

        loss = self.compute_loss(Q=z @ z_gathered.t())
        return loss

    @staticmethod
    def build_from_args(args):
        ot_reg_final = args.reg_final if args.reg_final is not None else args.reg
        reg_warmup_epochs = 30
        ot_reg_scheduler = np.concatenate((
            np.linspace(args.reg, ot_reg_final, reg_warmup_epochs),
            np.ones(args.epochs - reg_warmup_epochs) * ot_reg_final
        ))
        return BAMLoss(n_lrg_crops=args.global_crops_number,
                       n_crops=args.local_crops_number + args.global_crops_number,
                       n_tgt_crops=args.target_crops_number,
                       top_k=args.top_k_sink,
                       top_p=args.top_p_sink,
                       ot_reg=ot_reg_scheduler,
                       softmax_temperature=args.temperature,
                       positive_masking=args.positive_masking, )


class BamDistillLoss(BAMLoss):
    """
    Distill the Balanced-Attention matrix from a target network (e.g. momentum network or a pretrained one).
    """

    def compute_loss(self, Q: torch.Tensor, Q_tgt: torch.Tensor):
        """
        Compute BAM Loss with Distillation
        """
        num_large_crops = self.n_lrg_crops
        batch_size_per_crop = Q.size(0) // self.n_crops

        # positive-masking
        if self.pos_mask is None:   # make positive mask
            if self.positive_masking:
                self.pos_mask = get_positive_mask(batch_size_per_crop, n_crops=self.n_crops).detach()
            else:
                self.pos_mask = torch.zeros_like(Q).detach().bool()
        self.pos_mask = self.pos_mask[:, :Q.size(1)]

        # self-sinkhorn
        with torch.no_grad():
            P = self._sinkhorn(Q_tgt.detach())[:batch_size_per_crop * num_large_crops]  # use only large crops as target
            if self.masking_fn is not None:
                P = self.keep_top_values(P, n_crops=num_large_crops, mask_value=0.)
                P = P / P.sum(dim=-1, keepdim=True)
            P = P.chunk(num_large_crops)

        # self-softmax
        Q = (Q / self.softmax_temperature)
        if self.positive_masking:
            Q.masked_fill_(self.pos_mask, torch.finfo(Q.dtype).min)
        log_Q = self._log_softmax(Q).chunk(self.n_crops)

        loss_pos_mask = (~self.pos_mask[:batch_size_per_crop]).to(log_Q[0].dtype)

        # accumulate loss
        loss = 0.
        for v_tgt in range(num_large_crops):
            sub_loss, n_sub_loss = 0, 0
            for v_src in range(self.n_crops):
                if v_tgt == v_src and self.n_crops > 1:
                    continue  # skip same view
                sub_loss -= ((P[v_tgt] * log_Q[v_src]) * loss_pos_mask).sum(dim=-1).mean()
                n_sub_loss += 1

            loss += (sub_loss / n_sub_loss)
        return loss / num_large_crops

    def forward(self, z: torch.Tensor, z_teacher, labels=None, epoch=None):
        self.ot_reg = self.ot_reg_scheduler[epoch] if epoch is not None else self.ot_reg[-1]
        bs_per_crop = z.size(0) // self.n_crops

        # l2-normalize + gather targets from gpus
        z = F.normalize(z, dim=-1)
        z_teacher = F.normalize(z_teacher, dim=-1)
        z_teacher_gathered = gather_sort_tensors(z_teacher[:bs_per_crop * self.n_tgt_crops], n_crops=self.n_tgt_crops)

        loss = self.compute_loss(
            Q=z @ z_teacher_gathered.t(),
            Q_tgt=z_teacher @ z_teacher_gathered.t())
        return loss

    @staticmethod
    def build_from_args(args):
        ot_reg_final = args.reg_final if args.reg_final is not None else args.reg
        reg_warmup_epochs = 30
        ot_reg_scheduler = np.concatenate((
            np.linspace(args.reg, ot_reg_final, reg_warmup_epochs),
            np.ones(args.epochs - reg_warmup_epochs) * ot_reg_final
        ))
        return BamDistillLoss(n_lrg_crops=args.global_crops_number,
                              n_crops=args.local_crops_number + args.global_crops_number,
                              n_tgt_crops=args.target_crops_number,
                              top_k=args.top_k_sink,
                              top_p=args.top_p_sink,
                              ot_reg_scheduler=ot_reg_scheduler,
                              softmax_temperature=args.temperature,
                              positive_masking=args.positive_masking, )


def gather_sort_tensors(z: torch.Tensor, n_crops: int) -> torch.Tensor:
    """
    Do a gather over all embeddings, so we can compute the loss.
    Final shape is like: (batch_size * num_gpus) x embedding_dim
    """

    if dist.is_initialized():
        # all_gather fills the list as [<proc0>, <proc1>, ...]
        z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
        z_list = diffdist.functional.all_gather(z_list, z)
        # z_list = GatherLayer.apply(z)
        # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
        z_list = [chunk for x in z_list for chunk in x.chunk(n_crops)]
        # will be sorted as crop_1 from all gpus, crop_2 from all gpus ...
        z_sorted = []
        for m in range(n_crops):
            for i in range(dist.get_world_size()):
                z_sorted.append(z_list[i * n_crops + m])
        return torch.cat(z_sorted, dim=0)
    return z


@torch.no_grad()
def distributed_sinkhorn(Q: torch.Tensor,
                         num_sink_iter: int = 3,
                         world_size: int = None,
                         n_masked: int = 0):
    """
    Apply the distributed sinknorn optimization on the scores matrix to
    find the assignments
    """

    n_masked = max(0, n_masked)
    world_size = dist.get_world_size() if world_size is None else world_size

    sum_Q = torch.sum(Q, dtype=Q.dtype)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    k, n = Q.shape
    N = world_size * n

    # we follow the u, r, c and Q notations from
    # https://arxiv.org/abs/1911.05371

    # remove n_masked from rows, since Q is a transposed matrix
    r = torch.ones(k).cuda(non_blocking=True) / (k - n_masked)
    c = torch.ones(n).cuda(non_blocking=True) / N

    for _ in range(num_sink_iter):
        u = torch.sum(Q, dim=1, dtype=Q.dtype)
        dist.all_reduce(u)
        Q *= (r / u).unsqueeze(1)

        Q *= (c / torch.sum(Q, dim=0, dtype=Q.dtype)).unsqueeze(0)

    return (Q / torch.sum(Q, dim=0, keepdim=True, dtype=Q.dtype)).t()
