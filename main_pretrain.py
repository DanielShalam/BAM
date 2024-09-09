import argparse
import logging
import os
import datetime
from pathlib import Path
import time
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import utils
from bam import BAMLoss, DistillBAMLoss

from models.heads import MlpProjHead
from models.wrappers import MultiCropWrapper, StudentTeacherWrapper
from image_transforms import DataAugmentationDINO
from online_evaluation import get_knn_loaders, distributed_knn
from contrastive_croping.imagenet import ImageFolderCCrop, DataAugmentationCCrop
from contrastive_croping.contrastive_crop import update_box


def get_args_parser():
    parser = argparse.ArgumentParser('main_pretrain', add_help=False)

    # Contrastive Crop parameters
    parser.add_argument('--use_ccrop', type=utils.bool_flag, default=True,
                        help=""" Enable Contrastive-Crop. """)
    parser.add_argument('--box_thresh', type=float, default=0.1,
                        help='Threshold of boxing. default: 0.1')
    parser.add_argument('--loc_interval', type=int, default=20,
                        help='Frequency of box update (in epochs). default: 20')
    parser.add_argument('--ccrop_alpha', type=float, default=0.6,
                        help='Contrastive crop alpha')
    parser.add_argument('--ccrop_img_size', type=int, default=None,
                        help='img size for ccrop dataloader. ')

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        help=""" Name of architecture to train. """)
    parser.add_argument('--patch_size', default=16, type=int,
                        help=""" Patch size for ViTs. """)

    # KNN Evaluation parameters
    parser.add_argument('--knn_freq', default=0, type=int,
                        help=""" Run KNN evaluation every x epochs. """)
    parser.add_argument('--knn_eval_fraction', default=1., type=float,
                        help=""" Randomly sample a (balanced) fraction of images from eval dataset. """)
    parser.add_argument('--knn_train_fraction', default=0.1, type=float,
                        help=""" Randomly sample a (balanced) fraction of images from train dataset. """)
    parser.add_argument('--eval_first', type=utils.bool_flag, default=False,
                        help=""" Run KNN evaluation before training. """)

    # BAM
    parser.add_argument('--reg', default=0.05, type=float,
                        help=""" Initial value of sinkhorn entropy regularization. """)
    parser.add_argument('--reg_final', default=None, type=float,
                        help=""" Final value of sinkhorn entropy regularization. (default: --reg)""")
    parser.add_argument('--temperature', default=0.1, type=float,
                        help=""" Softmax (source distribution) temperature. """)
    parser.add_argument('--num_sink_iter', default=3, type=int,
                        help=""" Number of sinkhorn iterations. """)
    parser.add_argument('--top_k_sink', default=0, type=int,
                        help=""" Keep only the Top-K values of the Sinkhorn matrix. set <= 0 to disable. """)
    parser.add_argument('--positive_masking', default=True, type=utils.bool_flag,
                        help=""" Enable positive-masking for BAM loss. """)
    parser.add_argument('--target_crops_number', default=None, type=int,
                        help=""" How many target crops participate in the OT computation. """)

    # Projector parameters
    parser.add_argument('--hidden_dim', default=4096, type=int,
                        help=""" Dimensionality of the hidden layer in projector head. """)
    parser.add_argument('--out_dim', default=4096, type=int,
                        help=""" Dimensionality of the output layer in projector head. """)
    parser.add_argument('--n_layers', default=3, type=int,
                        help=""" Number of layers in projection head. """)
    parser.add_argument('--proj_bias', default=True, type=utils.bool_flag,
                        help=""" Use bias in head. """)
    parser.add_argument('--proj_act', default='gelu', type=str, choices=["relu", "gelu"],
                        help=""" Activation function in head. """)
    parser.add_argument('--proj_last_bn', type=utils.bool_flag, default=True,
                        help=""" If to use batch normalization on projection head output. """)
    parser.add_argument('--proj_use_bn', type=utils.bool_flag, default=True,
                        help=""" If to use batch normalization on projection head. """)

    # ViT parameters
    parser.add_argument('--drop_path_rate', type=float, default=0.,
                        help=""" Stochastic depth rate. """)
    parser.add_argument('--qkv_bias', type=utils.bool_flag, default=True,
                        help=""" Enable bias on qkv projection. """)

    # Training/Optimization parameters
    parser.add_argument('--teacher', type=utils.bool_flag, default=False,
                        help=""" Enable momentum teacher. """)
    parser.add_argument('--teacher_m', type=float, default=0.995,
                        help=""" Teacher momentum coefficient. """)

    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True,
                        help=""" Enable Mixed-precision training. """)

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
    parser.add_argument('--weight_decay', type=float, default=0.04,
                        help=""" Initial value of the weight decay.
                        With ViT, a smaller value at the beginning of training works well. """)
    parser.add_argument('--weight_decay_end', type=float, default=0.4,
                        help=""" Final value of the weight decay.
                        using a larger decay by the end of training improves performance for ViTs. """)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    parser.add_argument('--grad_checkpointing', type=int, default=0,
                        help='Enable gradient checkpointing every n model functions.')
    parser.add_argument('--compile', type=utils.bool_flag, default=False,
                        help='Run with torch.compile().')

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2)
    parser.add_argument('--local_crops_number', type=int, default=0)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.08, 1.))
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.25))
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
    parser.add_argument('--seed', default=0, type=int, help=""" Random Seed. """)
    parser.add_argument('--num_workers', default=10, type=int, help=""" Number of data loading workers per GPU. """)
    parser.add_argument("--dist_url", default="env://", type=str, help=""" Url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help=""" Please ignore and do not set this argument. """)
    return parser


def build_heads_from_args(args):
    """ Build projector head.
    """
    proj_head = MlpProjHead(
        input_dim=args.embed_dim, output_dim=args.out_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        use_bn=args.proj_use_bn, last_bn=args.proj_last_bn, bias=args.proj_bias, activation=args.proj_act
    )
    return proj_head


def get_ccrop_datasets_and_loaders(args):
    """ Build dataloaders with contrastive-crop if args.use_ccrop is set to True.
    """
    transform_rcrop = DataAugmentationDINO(
        args.global_crops_scale, args.local_crops_scale, args.local_crops_size, args.local_crops_number
    )
    transform_ccrop = DataAugmentationCCrop(
        args.global_crops_scale, args.local_crops_scale, args.local_crops_size, args.local_crops_number,
        alpha=args.ccrop_alpha,
    )
    train_dataset = ImageFolderCCrop(args.data_path, transform_rcrop=transform_rcrop, transform_ccrop=transform_ccrop)
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=True, )

    # Dataloader for box update
    ccrop_img_size = args.ccrop_img_size if args.ccrop_img_size is not None else 224
    print("Contrastive crop image size:", ccrop_img_size)
    transform_eval = transforms.Compose([
        transforms.Resize((ccrop_img_size, ccrop_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_train_dataset = ImageFolder(args.data_path, transform=transform_eval)
    eval_sampler = torch.utils.data.DistributedSampler(eval_train_dataset, shuffle=False)
    eval_train_loader = torch.utils.data.DataLoader(eval_train_dataset, sampler=eval_sampler,
                                                    batch_size=64, num_workers=args.num_workers,
                                                    shuffle=False, pin_memory=True, drop_last=False, )
    return train_dataset, train_loader, eval_train_loader


def train(args):
    utils.init_distributed_mode(args)
    utils.setup_for_distributed(args.rank == 0)
    utils.fix_random_seeds(args.seed + args.rank)
    utils.save_args_on_master(args)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    args.world_size = utils.get_world_size()
    cudnn.benchmark = True

    if args.target_crops_number is None:
        # default number of target views is number of large views
        args.target_crops_number = args.global_crops_number

    # ============ preparing data ... ============

    if args.use_ccrop:
        # load with contrastive-crop
        train_set, data_loader, eval_train_loader = get_ccrop_datasets_and_loaders(args)
        dataset_size = len(train_set)
    else:
        data_loader, dataset_size = utils.get_pretrain_loader(args)
    print(f"Data loaded: there are {dataset_size} images.")

    # ============ building networks ... ============

    model, embed_dim = utils.get_arch(args=args, num_classes=0)
    args.embed_dim = embed_dim
    # enable gradient checkpointing
    if args.grad_checkpointing > 0:
        model.set_grad_checkpointing(enable=True, every=args.grad_checkpointing)
    model = MultiCropWrapper(model, build_heads_from_args(args))
    model = model.cuda()
    print("Number of parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    teacher, teacher_no_ddp = None, None
    if args.teacher:
        print("Initialize teacher model...")
        teacher = utils.get_arch(args=args, num_classes=0)[0]
        teacher = MultiCropWrapper(teacher, build_heads_from_args(args))
        teacher = teacher.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.teacher:
            teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
            teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
            teacher_without_ddp = teacher.module
    elif args.teacher:
        teacher_without_ddp = teacher

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    if args.compile:
        print("Compiling model...")
        model = torch.compile(model, mode='max-autotune')

    if args.teacher:
        momentum_scheduler = utils.cosine_scheduler(args.teacher_m, 1, args.epochs, len(data_loader))
        model = StudentTeacherWrapper(model, teacher, momentum_scheduler)
        model = model.cuda()

    # ============ preparing optimizer ... ============

    params_groups = utils.get_params_groups(model_without_ddp)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    print(f"{args.arch} Model is fully built.")

    # precision scaler
    fp16_scaler = utils.AmpScalerWrapper(enable=args.use_fp16)

    # ============ init schedulers ... ============

    args.lr = args.lr * (args.batch_size_per_gpu * args.world_size) / 256.  # linear scaling rule for lr
    args.num_steps = len(data_loader)

    lr_schedule = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, args.num_steps, warmup_epochs=args.warmup_epochs)

    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, args.num_steps)

    # ============ preparing loss ... ============

    if not args.teacher:
        loss_fn = BAMLoss.build_from_args(args)
    else:
        loss_fn = DistillBAMLoss.build_from_args(args)  # Distillation loss
    loss_fn = loss_fn.cuda()
    print(loss_fn)

    # wandb
    utils.init_wandb(args=args, model=model, project="BAM-Pretraining", out_dir=args.output_dir)

    # ============ optionally resume training ... ============

    to_restore = {"epoch": 0, "boxes": None}
    utils.restart_from_checkpoint(os.path.join(args.output_dir, "checkpoint.pth"),
                                  model=model, optimizer=optimizer,
                                  fp16_scaler=fp16_scaler, run_variables=to_restore)
    start_epoch = to_restore["epoch"]
    if args.use_ccrop and start_epoch > args.warmup_epochs:
        assert to_restore["boxes"] is not None
        train_set.boxes = to_restore["boxes"]

    # ============ preparing KNN dataloaders ============

    knn_train_loader, knn_val_loader = get_knn_loaders(args)

    if args.eval_first:
        top1, top5 = distributed_knn(model_without_ddp.backbone, args, knn_train_loader, knn_val_loader)
        eval_dict = {'val/epoch': start_epoch-1, 'eval/KNN-Top-1': top1, 'eval/KNN-Top-5': top5}
        print(f"\nKNN Evaluation results:\n\t-> Top-1: {top1:.4f}%, Top-5: {top5:.4f}%")
        if args.teacher:
            top1, top5 = distributed_knn(teacher_without_ddp.backbone, args, knn_train_loader, knn_val_loader)
            eval_dict.update({'val/Teacher_KNN-Top-1': top1, 'val/Teacher_KNN-Top-5': top5})
            print(f"\n(Teacher) KNN Evaluation results:\n\t-> Top-1: {top1:.4f}%, Top-5: {top5:.4f}%")
        utils.log_dict(eval_dict)

    # ============ train ============

    # update boxes if we resumed from a checkpoint
    if args.use_ccrop and start_epoch > args.loc_interval and (start_epoch - 1) % args.loc_interval == 0:
        print("=>Update boxes from previous checkpoint (uses teacher network)...")
        all_boxes = update_box(args.arch,
                               eval_train_loader,
                               teacher_without_ddp.backbone if args.teacher
                               else model_without_ddp.backbone,
                               dataset_size,
                               block_idx=-1,
                               t=args.box_thresh)
        train_set.boxes = all_boxes.cpu()

    print("\nStart training !")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # check ContrastiveCrop
        train_set.use_box = args.use_ccrop and (epoch >= max(args.loc_interval, args.warmup_epochs) + 1)

        # ============ training one epoch ... ============

        train_stats = train_one_epoch(model, loss_fn, data_loader, optimizer, lr_schedule, wd_schedule,
                                      epoch, fp16_scaler, args)

        # ============ update bounding boxes ... ============

        if args.use_ccrop and epoch >= args.warmup_epochs and \
                epoch != (args.epochs - 1) and epoch % args.loc_interval == 0:
            # all_boxes: tensor (len_ds, 4); (h_min, w_min, h_max, w_max)
            all_boxes = update_box(
                args.arch, eval_train_loader,
                teacher_no_ddp.backbone if args.teacher
                else model_without_ddp.backbone,
                dataset_size, block_idx=-1,
                t=args.box_thresh
            )
            # on_cuda=True
            assert len(all_boxes) == dataset_size
            train_set.boxes = all_boxes.cpu()

        # ============ writing logs ... ============

        ckpt_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        if args.use_ccrop:
            ckpt_dict['boxes'] = train_set.boxes
        if fp16_scaler is not None:
            ckpt_dict['fp16_scaler'] = fp16_scaler.state_dict()

        save_checkpoint(ckpt_dict, stats=train_stats, epoch=epoch)

        # ============ KNN ============

        if args.knn_freq > 0 and ((epoch + 1) % args.knn_freq == 0 or epoch == args.epochs - 1):
            top1, top5 = distributed_knn(model_without_ddp.backbone, args, knn_train_loader, knn_val_loader)
            print(f"\nKNN Evaluation results:\n\t-> Top-1: {top1:.4f}%, Top-5: {top5:.4f}%")
            eval_dict = {'val/epoch': epoch, 'eval/KNN-Top-1': top1, 'eval/KNN-Top-5': top5}
            if args.teacher:
                top1, top5 = distributed_knn(teacher_no_ddp.backbone, args, knn_train_loader, knn_val_loader)
                eval_dict.update({'val/Teacher_KNN-Top-1': top1, 'val/Teacher_KNN-Top-5': top5})
                print(f"\n(Teacher) KNN Evaluation results:\n\t-> Top-1: {top1:.4f}%, Top-5: {top5:.4f}%")
            utils.log_dict(eval_dict)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model, loss_fn, data_loader, optimizer, lr_schedule, wd_schedule, epoch, fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    n_train_steps = len(data_loader)

    model.train()
    for it, (images, targets) in enumerate(metric_logger.log_every(data_loader, 100, header=header)):
        # update schedulers
        lr, wd = utils.update_lr_and_wd(optimizer, lr_schedule, wd_schedule, n_train_steps * epoch + it)

        images = [im.cuda(non_blocking=True) for im in images]
        with torch.cuda.amp.autocast(enabled=fp16_scaler.scaler is not None):
            if args.teacher:
                z_source, z_target = model(images)
            else:
                z_source = model(images)
                z_target = None

            loss = loss_fn(z_source, z_target, epoch=epoch)

        # optimization step
        if args.clip_grad:
            fp16_scaler(optimizer=optimizer, loss=loss, clip_grad=args.clip_grad,
                        parameters=model.parameters())
        else:
            fp16_scaler(optimizer=optimizer, loss=loss)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss.detach().item(), lr=lr, wd=wd)
        utils.log_dict({"global/lr": lr, "train/loss_step": loss.detach().item()})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    output_dict = {"train/" + k: meter.global_avg for k, meter in metric_logger.meters.items()}
    output_dict.update({"train/epoch": epoch})
    utils.log_dict(output_dict)  # log epoch-level

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
    args = argparse.ArgumentParser('main_pretrain', parents=[get_args_parser()]).parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
