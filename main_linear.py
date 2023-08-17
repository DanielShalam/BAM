import os
import argparse
import json
import wandb
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import vision_transformer as vits
import utils


def _get_dataloaders(is_train, args):
    if is_train:
        transform = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(224),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    data_path_suffix = 'train' if is_train else 'val'
    dataset = datasets.ImageFolder(os.path.join(args.data_path, data_path_suffix), transform=transform)

    sampler = None
    if is_train and args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=is_train,
        shuffle=(is_train and sampler is None),
    )
    return loader


def eval_linear(args):
    utils.init_distributed_mode(args)
    utils.setup_for_distributed(args.rank == 0)
    utils.fix_random_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    args.distributed = args.world_size > 1
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim
        if args.n_last_blocks > 0:
            embed_dim = embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
        model.fc = nn.Identity()
        model.head = nn.Identity()
        print("Model output dim =", embed_dim)

    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    model.cuda()
    model.eval()
    model = utils.load_pretrained_weights(model, args.pretrained_weights, 'model', args.arch, args.patch_size)

    # FC layer
    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels, bn_head=args.bn_head).cuda()

    if args.distributed:
        if utils.has_batchnorms(linear_classifier):
            linear_classifier = nn.SyncBatchNorm.convert_sync_batchnorm(linear_classifier)
        linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    print(f"Linear Classifier initialized. Layer shape - [{embed_dim}, {args.num_labels}]")

    # ============ preparing data ... ============
    train_loader = _get_dataloaders(True, args)
    val_loader = _get_dataloaders(False, args)
    print(f"Data loaded with {len(train_loader.dataset)} train and {len(val_loader.dataset)} val images.")

    # set optimizer and scheduler
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        # scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    if start_epoch > 0:
        for i in range(start_epoch):
            scheduler.step()
            print(optimizer.param_groups[0]["lr"])

    # set wandb
    exp_name = args.pretrained_weights.split('/')[-2] + f"bs-{args.batch_size_per_gpu}_lr-{args.lr}"
    utils.init_wandb(project="sot-imagenet-eval", args=args, model=linear_classifier, exp_name=exp_name)

    if args.evaluate:
        print("Evaluate model...")
        test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks,
                                      args.avgpool_patchtokens, epoch=start_epoch)
        print(f"Evaluation results on test images:\n"
              f"Top-1: {test_stats['acc1']:.1f}% | Top-5: {test_stats['acc5']:.1f}% | Loss: {test_stats['loss']:.2f}")

    print("Start training...")
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks,
                            args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(model, linear_classifier, val_loader, args.n_last_blocks,
                                          args.avgpool_patchtokens, epoch=epoch)
            print(f"Accuracy at epoch {epoch} on test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a+") as f:
                f.write(json.dumps(log_stats) + "\n")

            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            save_file = "checkpoint.pth.tar" if not args.augment else "checkpoint_augment.pth.tar"
            torch.save(save_dict, os.path.join(args.output_dir, save_file))

    print("Training of the supervised linear classifier on frozen features completed.\n"
          "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool):

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    log_freq = 50
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    linear_classifier.train()

    for b_idx, (inp, target) in enumerate(metric_logger.log_every(loader, 40, header=header)):

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch and n > 0:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1),
                                        torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)

            else:
                output = model(inp)

        output = linear_classifier(output)
        loss = criterion(output, target)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = inp.shape[0]
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc1=acc1.item(), n=batch_size)
        metric_logger.update(acc5=acc5.item(), n=batch_size)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if wandb.run is not None and b_idx % log_freq == 0:
            wandb.log({'train/' + k: meter.global_avg for k, meter in metric_logger.meters.items()})

    # gather stats from all processes
    metric_logger.synchronize_between_processes()
    if wandb.run is not None:
        wandb.log({"train/epoch_loss": metric_logger.loss.global_avg,
                   "train/epoch_acc1": metric_logger.acc1.global_avg}, )

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(model, linear_classifier, val_loader, n, avgpool, epoch):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    for inp, target in metric_logger.log_every(val_loader, 20, header='Test:'):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        if "vit" in args.arch:
            if n > 0:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat(
                        (output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        else:
            output = model(inp)

        output = linear_classifier(output)
        loss = criterion(output, target)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} Loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    if wandb.run is not None:
        wandb.log({
            "acc1": metric_logger.acc1.global_avg,
            "acc5": metric_logger.acc5.global_avg,
            "val_loss": metric_logger.loss.global_avg,
            "epoch": epoch,
        })

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):

    """
    Linear layer to train on top of frozen features
    """

    def __init__(self, dim, num_labels=1000, bn_head=False):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
        self.bn = None
        if bn_head:
            self.bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.bn is not None:
            x = self.bn(x)
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument('--augment', type=utils.bool_flag, default=False)
    parser.add_argument('--bn_head', type=utils.bool_flag, default=False)
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help="""Per-GPU batch-size""")
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--imagenet_subset', default=False, type=int, choices=[1, 10])
    parser.add_argument('--num_workers', default=10, type=int, help='Number of da\ta loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--debug', type=utils.bool_flag, default=False)
    parser.add_argument('--wandb', type=utils.bool_flag, default=True, help="""if to log results into wandb.""")
    parser.add_argument('--wandb_key', type=str, default="", help="""API key for wandb account.""")

    args = parser.parse_args()
    eval_linear(args)
