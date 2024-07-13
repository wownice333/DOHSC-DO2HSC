import numpy as np
import scipy
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def test_DOHSC_loader_cifar(test_loader, model, c, R, device, trained_class, temp_auc=-1.):
    model.eval()
    c = c.to(device)
    label_score = []
    total_emb = []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.float().to(device)
            z = model(x)
            # print(z.shape)
            # print(c.shape)
            dist = torch.sum((z - c) ** 2, dim=1)

            scores = dist - R ** 2
            total_emb.append(z)
            label_score += list(zip(
                y.cpu().data.numpy().tolist(),
                scores.cpu().data.numpy().tolist(),
                dist.cpu().data.numpy().tolist()
            ))
        emb = torch.cat(total_emb, dim=0)
        test_scores = label_score
        labels, scores, dist = zip(*label_score)
        labels = np.array(labels)
        dist = np.array(dist)
        scores = np.array(scores)
        test_auc = roc_auc_score(labels, scores)

    if temp_auc < test_auc:
        temp_auc = test_auc
    return test_auc, temp_auc


def test_stage_for_DO2HSC_loader_cifar(test_loader, model, c, R_max, R_min, device, trained_class, temp_auc=-1.):
    c = c.to(device)
    model.eval()
    label_score = []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.float().to(device)
            z = model(x)
            dist = torch.sqrt(torch.sum((z - c) ** 2, dim=1))

            scores = (dist - R_max) * (dist - R_min)

            label_score += list(zip(
                y.cpu().data.numpy().tolist(),
                scores.cpu().data.numpy().tolist(),
                dist.cpu().data.numpy().tolist()))

    labels, scores, dist = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    dist = np.array(dist)
    test_auc = roc_auc_score(labels, scores)

    if temp_auc < test_auc:
        temp_auc = test_auc
    return test_auc, temp_auc


def load_model_DO2HSC(test_loader, model, c, R_max, R_min, device):
    c = c.to(device)
    model.eval()
    label_score = []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.float().to(device)
            z = model(x)
            dist = torch.sqrt(torch.sum((z - c) ** 2, dim=1))

            scores = (dist - R_max) * (dist - R_min)

            label_score += list(zip(
                y.cpu().data.numpy().tolist(),
                scores.cpu().data.numpy().tolist(),
                dist.cpu().data.numpy().tolist()))

    labels, scores, dist = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    dist = np.array(dist)
    test_auc = roc_auc_score(labels, scores)

    return test_auc


def load_model_DOHSC(test_loader, model, c, R, device):
    model.eval()
    c = c.to(device)
    label_score = []
    total_emb = []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.float().to(device)
            z = model(x)
            dist = torch.sum((z - c) ** 2, dim=1)

            scores = dist - R ** 2
            total_emb.append(z)
            label_score += list(zip(
                y.cpu().data.numpy().tolist(),
                scores.cpu().data.numpy().tolist(),
                dist.cpu().data.numpy().tolist()
            ))
        emb = torch.cat(total_emb, dim=0)
        test_scores = label_score
        labels, scores, dist = zip(*label_score)
        labels = np.array(labels)
        dist = np.array(dist)
        scores = np.array(scores)
        test_auc = roc_auc_score(labels, scores)

    return test_auc
