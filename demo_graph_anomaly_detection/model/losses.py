import torch
from cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation


def local_global_loss_(l_enc, g_enc, batch, measure, device):
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
    neg_mask = torch.ones((num_nodes, num_graphs)).to(device)
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())
    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum() #
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos

def calculate_svdd_loss(output, c, R, nu, device):

    output = output.to(device)
    c = c.to(device)
    dist = torch.sum((output - c) ** 2, dim=1)
    scores = dist - R ** 2
    loss = R ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    return loss, dist


def calculate_hypersphere_loss(output, c, R_max, R_min, device):
    output = output.to(device)
    c = c.to(device)
    dist = torch.sum((output - c) ** 2, dim=1)
    low = torch.ones_like(dist) * (R_min ** 2)
    high = torch.ones_like(dist) * (R_max ** 2)
    loss = torch.mean(torch.maximum(dist, high) - torch.minimum(dist, low) - (high - low))
    return loss, dist


