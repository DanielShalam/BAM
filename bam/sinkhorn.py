import torch
import torch.distributed as dist


@torch.no_grad()
def distributed_sinkhorn(Q: torch.Tensor, world_size: int, num_sink_iter: int):

    """
    Apply the distributed sinknorn optimization on the scores matrix to
    find the assignments
    """

    eps_num_stab = 1e-12

    # remove potential infs in Q
    # replace the inf entries with the max of the finite entries in Q
    mask = torch.isinf(Q)
    ind = torch.nonzero(mask)
    if len(ind) > 0:
        for i in ind:
            Q[i[0], i[1]] = 0
        m = torch.max(Q)
        for i in ind:
            Q[i[0], i[1]] = m
    sum_Q = torch.sum(Q, dtype=Q.dtype)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    k = Q.shape[0]
    n = Q.shape[1]
    N = world_size * n

    # we follow the u, r, c and Q notations from
    # https://arxiv.org/abs/1911.05371
    r = torch.ones(k) / k
    c = torch.ones(n) / N

    r = r.cuda(non_blocking=True)
    c = c.cuda(non_blocking=True)

    for _ in range(num_sink_iter):
        u = torch.sum(Q, dim=1, dtype=Q.dtype)
        dist.all_reduce(u)

        # for numerical stability, add a small epsilon value
        # for non-zero Q values.
        if len(torch.nonzero(u == 0)) > 0:
            Q = Q + eps_num_stab
            u = torch.sum(Q, dim=1, dtype=Q.dtype)
            dist.all_reduce(u)
        u = r / u

        # remove potential infs in "u"
        # replace the inf entries with the max of the finite entries in "u"
        mask = torch.isinf(u)
        ind = torch.nonzero(mask)
        if len(ind) > 0:
            for i in ind:
                u[i[0]] = 0
            m = torch.max(u)
            for i in ind:
                u[i[0]] = m

        Q *= u.unsqueeze(1)
        Q *= (c / torch.sum(Q, dim=0, dtype=Q.dtype)).unsqueeze(0)

    return (Q / torch.sum(Q, dim=0, keepdim=True, dtype=Q.dtype)).t()
