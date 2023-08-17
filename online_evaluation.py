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
    split data among gpus
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


def knn_loaders(args, splits=None):
    """
    create loaders for knn evaluation
    """

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
    loader_train = None
    train_path = args.data_path if 'train' in args.data_path else os.path.join(args.data_path, 'train')
    if 'train' in splits:
        dataset_train = datasets.ImageFolder(train_path, transform=eval_transform)
        targets = torch.tensor(dataset_train.targets)
        # sample and distribute random subsets
        print("Split train dataset to gpus")
        subset_indices = get_subset_indices(args, targets, args.knn_train_fraction,
                                            distributed_split=args.world_size > 1,
                                            balanced=True, seed=0)
        # mask subset loader
        data_train = SubsetWithTargets(dataset_train, subset_indices, targets[subset_indices])
        loader_train = torch.utils.data.DataLoader(
            data_train, sampler=None, batch_size=args.batch_size_per_gpu, num_workers=8,
            pin_memory=True, drop_last=False, shuffle=False,
        )

    # eval loader
    loader_eval = None
    if 'val' in splits:
        val_path = train_path.replace('train', 'val')
        dataset_eval = datasets.ImageFolder(val_path, transform=eval_transform)
        if not args.knn_eval_fraction or args.knn_eval_fraction <= 0 or args.knn_eval_fraction >= 1:
            loader_eval = torch.utils.data.DataLoader(
                dataset_eval, batch_size=args.batch_size_per_gpu, num_workers=8, pin_memory=True, drop_last=False,
            )
        else:
            # use subset
            targets = torch.tensor(dataset_eval.targets)
            subset_indices = get_subset_indices(args, targets, fraction=args.knn_eval_fraction,
                                                distributed_split=False, balanced=True, seed=0)
            data_eval = SubsetWithTargets(dataset_eval, subset_indices, targets[subset_indices])
            loader_eval = torch.utils.data.DataLoader(
                data_eval, batch_size=args.batch_size_per_gpu, num_workers=8, pin_memory=True, drop_last=False,
            )

    return loader_train, loader_eval


@torch.no_grad()
def extract_features(model, loader):
    """
    extract base features
    """

    train_labels = torch.LongTensor(loader.dataset.targets)
    print("num samples per gpu:", len(train_labels))
    n_samples_local = len(train_labels)
    n_batches = len(loader)

    print(f"compute features...")
    train_features = None
    idx = 0
    total_time_start = time()
    for count, (images, targets) in enumerate(loader):
        images = images.cuda(non_blocking=True)
        features = model(images, withhead=False)
        batch_size, dim = features.size()

        if train_features is None:
            train_features = torch.empty((n_samples_local, dim), device=features.device)

        if idx+batch_size > n_samples_local:
            print("Size warning. final index:", idx+batch_size,
                  "is larger then the expected number of features:", n_samples_local)

        train_features[idx: min(idx+batch_size, n_samples_local), :] = features
        idx += batch_size
        if (count+1) % 50 == 0:
            print(f"{count + 1}/{n_batches}, time (sec): {time()-total_time_start:.4f}")

    train_features = F.normalize(train_features, dim=-1)
    return train_features.t(), train_labels.view(1, -1).cuda(non_blocking=True)


@torch.no_grad()
def knn(model, args, train_loader=None, val_loader=None, k=20, T=0.07):
    """
    distributed knn
    """

    print(f"Start {k}-KNN evaluation...")
    model.eval()
    # load datasets
    temp_loaders = False
    if train_loader is None or val_loader is None:
        print("reloading data loaders...")
        temp_loaders = True
        train_loader, val_loader = knn_loaders(args)

    # extract train features
    train_features, train_targets = extract_features(model, train_loader)

    # evaluate on validation images
    C = np.max(val_loader.dataset.targets) + 1
    retrieval_one_hot = torch.zeros(k, C).cuda(non_blocking=True)
    n_batches = len(val_loader)
    top1, top5, total = 0.0, 0.0, 0

    total_time_start = time()
    print("Train features extracted. Evaluating... ")
    for idx, (images, targets) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        batch_size = targets.shape[0]

        # forward + find k-most similar samples locally for each gpu
        eval_feat = model(images, withhead=False)
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

        if (idx+1) % 50 == 0:
            print(f"test {idx + 1}/{n_batches}\t",
                  f"time: {time()-total_time_start}\t",
                  f"top1: {top1 * 100.0 / total:.2f}\t",
                  f"top5: {top5 * 100.0 / total:.2f}")

    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total

    if temp_loaders:
        del train_loader, val_loader

    return top1, top5


class Queue(object):
    def __init__(self, fraction, world_size):
        self._fraction = fraction
        self.world_size = world_size
        self.queue = torch.randn()

    def synchronize_between_processes(self):
        if self.world_size > 1:
            gather_queue = self.queue.clone()
            gather_list = [torch.empty_like(self.queue) for _ in range(self.world_size)]
            dist.all_gather(gather_list, gather_queue)
            return torch.cat(gather_queue)

        return self.queue
