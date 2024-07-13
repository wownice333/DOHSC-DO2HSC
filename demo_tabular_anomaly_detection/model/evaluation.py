import numpy as np
import scipy
import torch
from sklearn.metrics import precision_recall_fscore_support, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm


def test_stage_DOHSC_loader(test_loader, model, c, R, device, dataset_name, temp_auc, temp_f1):
    model.eval()
    c = c.to(device)
    label_score = []
    with torch.no_grad():
        tq = tqdm(test_loader, total=len(test_loader))
        for x, y in tq:
            x = x.float().to(device)
            z = model(x)
            dist = torch.sum((z - c) ** 2, dim=1)

            scores = dist - R ** 2

            label_score += list(zip(
                y.cpu().data.numpy().tolist(),
                scores.cpu().data.numpy().tolist(),
                dist.cpu().data.numpy().tolist()
            ))

        test_scores = label_score
        labels, scores, dist = zip(*label_score)
        labels = np.array(labels)
        dist = np.array(dist)
        scores = np.array(scores)

        y_pred = np.where(scores > 0, 1, 0)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, y_pred, average="binary")

        test_auc = roc_auc_score(labels, scores)
    if temp_f1 < f1:
        temp_f1 = f1
        temp_auc = test_auc
    return test_auc, temp_auc, f1, temp_f1


def test_stage_for_DO2HSC_loader(test_loader, model, c, R_max, R_min, device, dataset_name, temp_auc,
                                 temp_f1):
    c = c.to(device)
    model.eval()
    label_score = []
    with torch.no_grad():
        tq = tqdm(test_loader, total=len(test_loader))
        for x, y in tq:
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

    y_pred = np.where(scores > 0, 1, 0)
    _, _, f1, _ = precision_recall_fscore_support(
        labels, y_pred, average="binary")

    test_auc = roc_auc_score(labels, scores)

    if temp_f1 < f1:
        temp_f1 = f1
        temp_auc = test_auc
        
    return test_auc, temp_auc, f1, temp_f1


def load_model_DO2HSC(test_loader, model, c, R_max, R_min, device):
    c = c.to(device)
    model.eval()
    label_score = []
    with torch.no_grad():
        for x, y in test_loader:
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

    y_pred = np.where(scores > 0, 1, 0)
    _, _, f1, _ = precision_recall_fscore_support(
        labels, y_pred, average="binary")

    test_auc = roc_auc_score(labels, scores)

    return test_auc, f1


def load_model_DOHSC(test_loader, model, c, R, device):
    model.eval()
    c = c.to(device)
    label_score = []
    total_emb = []

    with torch.no_grad():
        for x, y in test_loader:
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
        labels, scores, dist = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        y_pred = np.where(scores > 0, 1, 0)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, y_pred, average="binary")

        test_auc = roc_auc_score(labels, scores)

    return test_auc, f1
