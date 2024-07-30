import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def get_positive_mask(batch_size_per_crop: int,
                      n_crops: int):
    """
    Mask for all positive values of each source image (all generated crops)
    in a multi-gpu settings
    """

    # init mask
    mask = torch.zeros((batch_size_per_crop, batch_size_per_crop * n_crops * dist.get_world_size()))

    idx_start_crop = torch.arange(n_crops) * (batch_size_per_crop * dist.get_world_size())
    idx_start_crop_for_rank = idx_start_crop + (dist.get_rank() * batch_size_per_crop)
    for sample_idx in range(batch_size_per_crop):
        mask[sample_idx, sample_idx + idx_start_crop_for_rank] = 1

    # repeat mask for n crops
    return mask.repeat((n_crops, 1)).bool().cuda(non_blocking=True)


def get_positive_bias(batch_size_per_gpu: int,
                      n_crops: int,
                      dtype: torch.dtype):
    """
    Mask for all positive values of each source image (all generated crops)
    in a multi-gpu settings
    """

    smallest_values = torch.finfo(dtype).min
    eff_bs = batch_size_per_gpu * dist.get_world_size()
    mask = torch.zeros((batch_size_per_gpu, eff_bs * n_crops), dtype=dtype)

    crop_start_col = eff_bs * torch.arange(n_crops)
    rank_start_col = crop_start_col + dist.get_rank() * batch_size_per_gpu
    for i in range(batch_size_per_gpu):
        mask[i, rank_start_col + i] = smallest_values

    mask = mask.repeat((n_crops, 1)).cuda(non_blocking=True)
    return mask


def get_self_mask(batch_size_per_gpu: int,
                  n_crops: int):
    """
    Build Self Mask in a multi-gpu setting
    (Same as Diagonal masking for 1 GPU)
    """

    eff_bs = batch_size_per_gpu * dist.get_world_size()
    # init mask with the rows of 1 crop
    mask = torch.zeros((batch_size_per_gpu, eff_bs * n_crops))
    diag_idx = torch.arange(eff_bs) + dist.get_rank() * batch_size_per_gpu
    mask.scatter_(-1, diag_idx.unsqueeze(-1), 1)

    # repeat for other crops and fix offset
    mask = mask.repeat((n_crops, 1))
    for i in range(1, n_crops):
        crop_mask = mask[i * batch_size_per_gpu: (i + 1) * batch_size_per_gpu]
        mask[i * batch_size_per_gpu: (i + 1) * batch_size_per_gpu] = torch.roll(crop_mask, shifts=i * eff_bs, dims=1)

    return mask.bool().cuda(non_blocking=True)


def get_top_k_mask(sim: torch.Tensor,
                   k: int,
                   inverse: bool = False):
    """
    Mask for the top-k values of each row.
    Return mask with the same shape as "sim" with 1 indicating a top-k index.
    Set inverse=True to mask all but the top indices.
    """

    mask = torch.zeros_like(sim)
    mask.scatter_(-1, sim.topk(k, dim=-1).indices, 1)
    if inverse:
        return ~(mask.bool())
    return mask.bool()


def get_top_p_mask(sim: torch.Tensor,
                   is_distribution: bool,
                   threshold: float = 0.9,
                   inverse: bool = False):
    """
    Mask for keeping indices with cumulative probability < threshold for each row.
    Use inverse=True to mask all but the top indices.
    """

    sorted_logits, sorted_indices = torch.sort(sim, descending=True)
    if is_distribution:
        cum_probs = torch.cumsum(sorted_logits, dim=-1)     # not need softmax if input is distribution
    else:
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > threshold
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value=False)

    sorted_logits.fill_(1)
    sorted_logits[sorted_indices_to_remove] = 0
    mask = sorted_logits.scatter(-1, sorted_indices, sorted_logits).bool()
    if inverse:
        return ~mask
    return mask
