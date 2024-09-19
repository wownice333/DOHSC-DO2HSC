import torch
import numpy as np
def obtain_c(model, dataloader, eps=0.1):
    emb, y = model.get_result(dataloader)
    c = torch.mean(torch.tensor(emb), dim=0)

    # If c is close to 0, set to +-eps
    # To avoid trivial problem
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


def get_radius_bi_hypersphere(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R_max and R_min via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu), np.quantile(
        np.sqrt(dist.clone().data.cpu().numpy()), nu)
