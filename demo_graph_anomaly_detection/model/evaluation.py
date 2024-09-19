import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def test_stage_DOHSC_loader(test_loader, model, c, R, device, trained_class, temp_auc=-np.inf):
    model.eval()
    c = c.to(device)
    label_score = []
    with torch.no_grad():
        tq = tqdm(test_loader, total=len(test_loader))
        for data in tq:  
            data = data.to(device)
            _, z = model(data.x, data.edge_index, data.batch, data.num_graphs)

            dist = torch.sum((z - c) ** 2, dim=1)
            scores = dist - R ** 2
            label_score += list(zip(
                data.y.tolist(),
                scores.cpu().data.numpy().tolist()))

    labels, scores = zip(*label_score)
    labels = np.array(labels)
    labels = np.where(labels == trained_class, 0, 1)
    scores = np.array(scores)
    test_auc = roc_auc_score(labels, scores)
    if temp_auc < test_auc:
        temp_auc = test_auc
    return test_auc, temp_auc


def test_stage_DO2HSC_loader(test_loader, model, c, R_max, R_min, device, trained_class, temp_auc=-np.inf):
    model.eval()
    c = c.to(device)
    label_score = []
    with torch.no_grad():
        tq = tqdm(test_loader, total=len(test_loader))
        for data in tq:  
            data = data.to(device)
            _, z = model(data.x, data.edge_index, data.batch, data.num_graphs)
    
            dist = torch.sqrt(torch.sum((z - c) ** 2, dim=1))
            scores = (dist - R_max) * (dist - R_min)

            label_score += list(zip(
                data.y.tolist(),
                scores.cpu().numpy().tolist()))

    labels, scores = zip(*label_score)
    labels = np.array(labels)
    labels = np.where(labels == trained_class, 0, 1)

    scores = np.array(scores)
    # in_point_num = np.where(scores <= 0)[0].shape[0]
    # print('Number of test data within R1-R2:', in_point_num, 'Ratio:', in_point_num / len(scores))
    test_auc = roc_auc_score(labels, scores)
    if temp_auc < test_auc:
        temp_auc = test_auc
    return temp_auc