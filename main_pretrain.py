# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path

import utils
from bam import BAMLoss
from online_evaluation import knn_loaders, knn
from vision_transformer import ProjHead

try:
    from apex.optimizers import FusedLAMB
    has_apex = True
except ImportError:
    has_apex = False

custom_archs = ['vit_small', 'vit_base']


def get_args_parser():
    parser = argparse.ArgumentParser('SOT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_small', 'vit_base'] + utils.torchvision_archs,
                        help=""" Name of architecture to train. """)
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid instabilities.""")

    # Evaluation parameters
    parser.add_argument('--knn_freq', default=0, type=int,
                        help=""" Run KNN evaluation every x epochs. """)
    parser.add_argument('--eval_first', default=0, type=int,
                        help=""" Run KNN evaluation before training. """)
    parser.add_argument('--knn_eval_fraction', default=1., type=float,
                        help=""" Randomly sample a (balanced) fraction of images from eval dataset. """)
    parser.add_argument('--knn_train_fraction', default=0.1, type=float,
                        help=""" Randomly sample a (balanced) fraction of images from train dataset. """)
    # Loss parameters
    parser.add_argument('--loss_fn', default='sot', type=str,
                        help=""" Loss function to apply. """)
    parser.add_argument('--reg', default=0.05, type=float,
                        help=""" Initial value of sinkhorn entropy regularization. """)
    parser.add_argument('--reg_final', default=0.05, type=float,
                        help=""" Final value of sinkhorn entropy regularization. """)
    parser.add_argument('--temperature', default=0.1, type=float,
                        help=""" Softmax (source distribution) temperature. """)
    parser.add_argument('--num_sink_iter', default=3, type=int,
                        help=""" Number of sinkhorn iterations. """)

    # Projector parameters
    parser.add_argument('--hidden_dim', default=4096, type=int,
                        help=""" Dimensionality of the hidden layer in projector head. """)
    parser.add_argument('--out_dim', default=4096, type=int,
                        help=""" Dimensionality of the output layer in projector head. """)
    parser.add_argument('--n_layers', default=None, type=int,
                        help=""" Number of layers in projection head. 
                        By default we use 3 layers for Resnet backbone and 4 layers for ViT. """)
    parser.add_argument('--use_bn_in_head', type=utils.bool_flag, default=True,
                        help=""" If to use batch normalization on projection head. """)

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True,
                        help=""" Enable Mixed-precision training. """)
    parser.add_argument('--chunk_size', type=int, default=0)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    parser.add_argument('--batch_size_per_gpu', default=256, type=int)
    parser.add_argument('--epochs', default=100, type=int,
                        help=""" Number of training epochs. """)
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help=""" Number of warmup epochs. We use 10 for Resnet and 40 for ViT. """)
    parser.add_argument('--lr', default=0.0005, type=float,
                        help=""" Highest learning rate used during training. """)
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help=""" Initial and final learning rate used during training. """)
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars', 'lamb'])
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help=""" Stochastic depth rate. """)
    parser.add_argument('--attn_bias', type=utils.bool_flag, default=False)
    parser.add_argument('--weight_decay', type=float, default=0.04,
                        help=""" Initial value of the weight decay.
                        With ViT, a smaller value at the beginning of training works well. """)
    parser.add_argument('--weight_decay_end', type=float, default=0.4,
                        help=""" Final value of the weight decay.
                        using a larger decay by the end of training improves performance for ViTs. """)

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2)
    parser.add_argument('--local_crops_number', type=int, default=6)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.))
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4))
    parser.add_argument('--local_crops_size', type=int, default=96)

    # Misc
    parser.add_argument('--wandb', type=utils.bool_flag, default=True,
                        help=""" Enable Wandb logger. """)
    parser.add_argument('--wandb_key', type=str, default="",
                        help=""" API key for wandb account. """)
    parser.add_argument('--log_freq', default=10, type=int,
                        help=""" Log results every n steps. """)
    parser.add_argument('--debug', type=utils.bool_flag, default=False,
                        help=""" Need to be fix. """)
    parser.add_argument('--data_path', default="/path/to/imagenet/train/", type=str,
                        help=""" Please specify path to the ImageNet training data. """)
    parser.add_argument('--output_dir', default=".", type=str,
                        help=""" Path to save logs and checkpoints. """)
    parser.add_argument('--saveckp_freq', default=20, type=int,
                        help=""" Save checkpoint every x epochs. """)
    parser.add_argument('--seed', default=0, type=int,
                        help=""" Random seed. """)
    parser.add_argument('--num_workers', default=10, type=int,
                        help=""" Number of data loading workers per GPU. """)
    parser.add_argument("--dist_url", default="env://", type=str, help=""" Url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help=""" Please ignore and do not set this argument. """)
    return parser


def train(args):
    utils.init_distributed_mode(args)
    utils.setup_for_distributed(args.rank == 0)
    utils.fix_random_seeds(args.seed)
    utils.save_args_on_master(args)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    data_loader, dataset_size = utils.get_pretrain_loader(args)

    # ============ building networks ... ============
    model, embed_dim = utils.get_arch(args=args)
    proj_head = ProjHead(
        arch=args.arch,
        input_dim=embed_dim,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        output_dim=args.out_dim,
        last_bn=args.use_bn_in_head
    )

    model = utils.MultiCropWrapper(model, proj_head)
    model = model.cuda()

    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # synchronize batch norms (if any)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    print(f"Model is fully built with {args.arch} network.")

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(model)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler

    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches

    elif args.optimizer == "lamb":
        assert has_apex, "Error: LAMB requires Apex to be installed via: \n" \
                         "pip install -v --no-cache-dir --global-option=--cpp_ext --global-option=--cuda_ext ./."
        optimizer = FusedLAMB(params_groups)

    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    fp16_scaler = None
    if args.use_fp16:
        # mixed precision training
        # fp16_scaler = torch.cuda.amp.GradScaler(init_scale=2.**12, growth_interval=500)
        fp16_scaler = torch.cuda.amp.GradScaler()
    else:
        # full precision is faster with tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ============ init schedulers ... ============
    args.world_size = utils.get_world_size()
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * args.world_size) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )

    # ============ preparing loss ... ============
    bam_loss = BAMLoss(args=args).cuda()
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
    )
    start_epoch = to_restore["epoch"]

    # initialize wandb
    utils.init_wandb(args=args, model=model, project="sot-imagenet", out_dir=args.output_dir)

    # build KNN loaders
    knn_train_loader, knn_val_loader = None, None
    if args.knn_freq > 0:
        print("\n\t-> Build dataloaders for KNN Evaluation...")
        knn_train_loader, knn_val_loader = knn_loaders(args)

        if args.eval_first == 1:    # eval before training
            top1, top5 = knn(model, args, knn_train_loader, knn_val_loader)
            print(f"\nKNN Evaluation results:"
                  f"\n\t-> Top-1: {top1:.4f}%, Top-5: {top5:.4f}")
            utils.log_dict({'eval/eval_epoch': start_epoch - 1, 'eval/knn_top_1': top1, 'eval/knn_top_5': top5})

    # training
    print("\nStart training !")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        sot_loss.on_epoch_start(epoch)

        # ============ training one epoch ... ============
        train_stats = train_one_epoch(model, sot_loss, data_loader, optimizer, lr_schedule, wd_schedule,
                                      epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'sot_loss': sot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        save_checkpoint(save_dict, stats=train_stats, epoch=epoch)

        # ============ KNN ============
        if args.knn_freq > 0 and (epoch + 1) % args.knn_freq == 0:
            top1, top5 = knn(model, args, knn_train_loader, knn_val_loader)
            print(f"\nKNN Evaluation results:"
                  f"\n\t-> Top-1: {top1:.4f}%, Top-5: {top5:.4f}%")
            utils.log_dict({'eval/eval_epoch': epoch, 'eval/knn_top_1': top1, 'eval/knn_top_5': top5})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model, sot_loss, data_loader, optimizer, lr_schedule, wd_schedule, epoch, fp16_scaler, args):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    model.train()
    for it, (images, targets) in enumerate(metric_logger.log_every(data_loader, 100, header=header)):

        it = len(data_loader) * epoch + it  # global training iteration

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        images = [im.cuda(non_blocking=True) for im in images]

        optimizer.zero_grad()
        # with torch.cuda.amp.autocast(enabled=fp16_scaler is not None, dtype=torch.bfloat16):
        with torch.cuda.amp.autocast(enabled=fp16_scaler is not None):
            z = model(images, "train")
            loss, _ = sot_loss(z)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        grad_scale = 1. if fp16_scaler is None else fp16_scaler.get_scale()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(model, args.clip_grad)

            utils.cancel_gradients_last_layer(epoch, model, 3)

            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(model, args.clip_grad)

            utils.cancel_gradients_last_layer(epoch, model, 3)

            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        utils.log_dict({
            "global/lr": lr_schedule[it],
            "global/grad_scaling": grad_scale,
            "train_aug_loss": loss.item(),
        })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    output_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    utils.log_dict(output_dict)
    return output_dict


def save_checkpoint(save_dict, stats, epoch):
    """
    Save checkpoint
    """
    utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
    if args.saveckp_freq and epoch % args.saveckp_freq == 0:
        utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))

    log_stats = {**{f'train_{k}': v for k, v in stats.items()}, 'epoch': epoch}
    if utils.is_main_process():
        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SOT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
