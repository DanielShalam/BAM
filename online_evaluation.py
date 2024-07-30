import math
import os
from time import time
import numpy as np
import contextlib
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Subset
from torchvision import datasets, transforms


class SubsetWithTargets(Subset):
    def __init__(self, dataset, indices, targets):
        super().__init__(dataset, indices)
        self.targets = targets


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_subset_indices(args, targets, fraction, distributed_split, balanced, seed=0):
    """
    Split data among GPUs
    """

    if not fraction or fraction <= 0 or fraction >= 1:
        indices = torch.arange(len(targets))
        if distributed_split:
            return indices.chunk(args.world_size)[args.rank]
        return indices

    # compute subset size and make sure it is equally divided by world size
    dataset_length = len(targets)
    n_subset_samples = dataset_length * fraction
    if distributed_split:
        n_subset_samples = math.floor(n_subset_samples / args.world_size) * args.world_size
    else:
        n_subset_samples = math.floor(n_subset_samples)
    print(f"Subset size: {n_subset_samples}/{dataset_length}")

    seed_or_null_context = temp_seed(seed) if seed is not None else contextlib.nullcontext()
    with seed_or_null_context:
        if balanced:
            # balanced sampling
            targets_np = targets.numpy()
            targets_unique = np.unique(targets_np)
            n = math.floor(n_subset_samples / len(targets_unique))
            print("Loading balanced subset with", n, "samples per class...")
            balanced_subset = np.hstack([np.random.choice(np.where(targets_np == c)[0], n, replace=False)
                                         for c in targets_unique])
            np.random.shuffle(balanced_subset)

            balanced_subset = torch.from_numpy(balanced_subset)
            if distributed_split:
                # if we need to split samples among gpus
                return balanced_subset.chunk(args.world_size)[args.rank]
            else:
                # if each gpu
                return balanced_subset
        else:
            # uniform sampling
            random_subset = np.random.choice(dataset_length, size=n_subset_samples, replace=False)
            if distributed_split:
                return random_subset.chunk(args.world_size)[args.rank]
            else:
                return random_subset


def get_knn_loaders(args, splits=None):
    """
    Create loaders for Online KNN
    """

    if args.knn_freq <= 0:
        return None, None

    if splits is None:
        splits = ['train', 'val']

    # basic eval imagenet transform
    eval_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # train loader
    train_dataloader = None
    train_path = args.data_path if 'train' in args.data_path else os.path.join(args.data_path, 'train')
    if 'train' in splits:
        dataset = datasets.ImageFolder(train_path, transform=eval_transform)
        targets = torch.tensor(dataset.targets)
        # sample and distribute random subsets
        print("Split train dataset to gpus")
        subset_indices = get_subset_indices(args, targets, args.knn_train_fraction,
                                            distributed_split=args.world_size > 1,
                                            balanced=True, seed=0)
        # mask subset loader
        dataset = SubsetWithTargets(dataset, subset_indices, targets[subset_indices])
        train_dataloader = torch.utils.data.DataLoader(
            dataset, sampler=None, batch_size=64, num_workers=8,
            pin_memory=True, drop_last=True, shuffle=False,
        )

    # eval loader
    eval_dataloader = None
    if 'val' in splits:
        val_path = train_path.replace('train', 'val')
        dataset = datasets.ImageFolder(val_path, transform=eval_transform)
        if not args.knn_eval_fraction or args.knn_eval_fraction <= 0 or args.knn_eval_fraction >= 1:
            eval_dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=64, num_workers=8, pin_memory=True, drop_last=False,
            )
        else:
            # use subset
            targets = torch.tensor(dataset.targets)
            subset_indices = get_subset_indices(args, targets, fraction=args.knn_eval_fraction,
                                                distributed_split=False, balanced=True, seed=0)
            dataset = SubsetWithTargets(dataset, subset_indices, targets[subset_indices])
            eval_dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=64, num_workers=8, pin_memory=True, drop_last=False,
            )

    return train_dataloader, eval_dataloader


@torch.no_grad()
def extract_features(model, loader):
    """
    Extract base encoder features
    """

    train_labels = torch.LongTensor(loader.dataset.targets)
    n_samples_local = len(train_labels)
    n_batches = len(loader)

    print(f"\n-> Compute Train Features...")
    train_features = None
    idx = 0
    total_time_start = time()
    for count, (images, targets) in enumerate(loader):
        images = images.cuda(non_blocking=True)
        features = model(images)
        batch_size, dim = features.size()

        if train_features is None:
            train_features = torch.empty((n_samples_local, dim), device=features.device)

        if idx + batch_size > n_samples_local:
            print("Size Warning: final index:", idx + batch_size,
                  "is larger then the expected number of features:", n_samples_local)

        train_features[idx: min(idx + batch_size, n_samples_local), :] = features
        idx += batch_size
        if (count + 1) % 50 == 0:
            print(f"{count + 1}/{n_batches}, time (sec): {time() - total_time_start:.4f}")

    train_features = F.normalize(train_features, dim=-1)
    return train_features.t(), train_labels.view(1, -1).cuda(non_blocking=True)


@torch.no_grad()
def distributed_knn(model, args, train_loader=None, val_loader=None,
                    k: int = 20, T: float = 0.07):
    """
    Distributed knn
    """

    print(f"\n-> Running {k}-KNN Evaluation...")

    temp_loaders = False
    if train_loader is None or val_loader is None:  # make dataloaders if they are not given
        print("\n-> Reloading Data loaders...")
        temp_loaders = True
        train_loader, val_loader = get_knn_loaders(args)

    # constants
    C = np.max(val_loader.dataset.targets) + 1
    retrieval_one_hot = torch.zeros(k, C).cuda(non_blocking=True)
    n_batches = len(val_loader)
    top1, top5, total = 0.0, 0.0, 0

    was_train = model.training
    model.eval()

    # extract train features
    train_features, train_targets = extract_features(model, train_loader)

    total_time_start = time()
    print("\n\t-> Extracting Train features is Done. Start Evaluating... ")
    for idx, (images, targets) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        batch_size = targets.shape[0]

        # forward + find k-most similar samples locally for each gpu
        eval_feat = model(images)
        eval_feat = F.normalize(eval_feat, p=2, dim=-1)
        top_local_sim, top_local_indexes = (eval_feat.mm(train_features)).topk(k, dim=-1, largest=True, sorted=True)

        if args.world_size > 1:
            # gather results from gpus
            res_local = torch.stack((top_local_sim, train_targets[:, top_local_indexes].squeeze()))
            z_list = [torch.zeros_like(res_local) for _ in range(args.world_size)]
            dist.all_gather(z_list, res_local)
            sim_gathered, targets_gathered = torch.cat(z_list, dim=-1).chunk(2)
            sim_gathered = sim_gathered.squeeze(0)
            targets_gathered = targets_gathered.squeeze(0)

            # find k-most similar samples after gathering. each gpu now holds the same top-k values.
            top_gathered_similarities, top_gathered_indices = sim_gathered.topk(k, dim=-1, largest=True, sorted=True)
            retrieved_nn = torch.gather(targets_gathered, 1, top_gathered_indices)

        else:
            # normal knn
            top_gathered_similarities = top_local_sim.squeeze(0)
            retrieved_nn = train_targets[:, top_local_indexes].squeeze(0)

        # voting
        retrieval_one_hot.resize_(batch_size * k, C).zero_()
        retrieval_one_hot.scatter_(1, retrieved_nn.type(torch.int64).view(-1, 1), 1)

        yd_transform = top_gathered_similarities.clone().div_(T).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batch_size, -1, C), yd_transform.view(batch_size, -1, 1)), 1)
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += batch_size

        if (idx + 1) % 50 == 0:
            print(f"\n\t-> Batch {idx + 1}/{n_batches}",
                  f"\n\t-> Time: {time() - total_time_start}",
                  f"\n\t-> Top-1: {top1 * 100.0 / total:.2f}",
                  f"\n\t-> Top-5: {top5 * 100.0 / total:.2f}")

    if temp_loaders:
        del train_loader, val_loader

    model.train(was_train)
    return top1 * 100.0 / total, top5 * 100.0 / total
