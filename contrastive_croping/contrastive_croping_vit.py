""" from https://github.com/xyupeng/ContrastiveCrop """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


@torch.no_grad()
def update_box(eval_train_loader, model, len_ds, block_idx, t=0.1):
    was_train = model.training
    model.eval()

    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]

    boxes = []
    print("\n=> Update bounding boxes...")
    for cur_iter, (images, _) in enumerate(eval_train_loader):  # drop_last=False

        Hi, Wi = images.shape[-2:]  # image size
        Hf, Wf = Hi // patch_size, Wi // patch_size     # image size after patching

        # get the attention map
        attn_map = model.get_selfattention(images.cuda(non_blocking=True),
                                           block_idx=block_idx).cpu().detach()  # (N, nh, W, H)
        B, nh = attn_map.size()[:2]  # number of head

        # we keep only the [cls] attention to all other patches
        attn_map = attn_map[:, :, 0, 1:].reshape(B, nh, -1)
        attn_map = attn_map.reshape(B, nh, Wf, Hf)

        # sum head values
        scaled_attn_map = attn_map.sum(1).view(B, -1)  # (B, Hf*Wf)

        # scale attention maps
        scaled_attn_map = scaled_attn_map - scaled_attn_map.min(1, keepdim=True)[0]
        scaled_attn_map = scaled_attn_map / (scaled_attn_map.max(1, keepdim=True)[0] + 1e-5)

        scaled_attn_map = scaled_attn_map.view(B, 1, Hf, Wf)
        scaled_attn_map = F.interpolate(scaled_attn_map, size=images.shape[-2:], mode='bilinear')  # (B, 1, Hi, Wi)

        for hmap in scaled_attn_map:
            hmap = hmap.squeeze(0)  # (Hi, Wi)
            h_filter = (hmap.max(1)[0] > t).int()
            w_filter = (hmap.max(0)[0] > t).int()

            h_min, h_max = torch.nonzero(h_filter).view(-1)[[0, -1]] / Hi  # [h_min, h_max]; 0 <= h <= 1
            w_min, w_max = torch.nonzero(w_filter).view(-1)[[0, -1]] / Wi  # [w_min, w_max]; 0 <= w <= 1
            boxes.append(torch.tensor([h_min, w_min, h_max, w_max]))

    boxes = torch.stack(boxes, dim=0).cuda()  # (num_iters, 4)
    gather_boxes = [torch.zeros_like(boxes) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_boxes, boxes)
    all_boxes = torch.stack(gather_boxes, dim=1).view(-1, 4)
    all_boxes = all_boxes[:len_ds]

    model.train(was_train)
    return all_boxes
