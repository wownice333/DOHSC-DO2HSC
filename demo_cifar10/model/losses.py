import torch

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



