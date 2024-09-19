import torch
from tqdm import tqdm

from Utils.initialize import obtain_c, get_radius, get_radius_bi_hypersphere
from model.evaluation import test_stage_DOHSC_loader, test_stage_DO2HSC_loader
from model.losses import calculate_svdd_loss, calculate_hypersphere_loss


def pretrain(DS, model, dataloader, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    pbar = tqdm(range(5))
    for epoch in pbar:
        loss_all = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, emb, pro_emb = model(data.x, data.edge_index, data.batch, data.num_graphs)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
    c = obtain_c(model, dataloader).to(device)
    torch.save({'center': c.cpu().data.numpy().tolist(), }, './pre_weights/' + DS + '_pretrained_center.pth')
    return c


def train_DOHSC(model, train_loader, test_loader, c, nu, epochs, lam, train_class, lr_milestones, lr, device):
    R = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=lr_milestones, gamma=0.1)

    temp_auc = -1.
    test_auc = -1.
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        loss_all = 0
        svdd_all = 0
        total_dist = []
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, pro_emb = model(data.x, data.edge_index, data.batch, data.num_graphs)
            svdd_loss, dist = calculate_svdd_loss(pro_emb, c, R, nu, device)
            loss_all += loss.item()
            svdd_all += svdd_loss.item()
            total_loss = loss + lam * svdd_loss
            total_loss.backward()
            optimizer.step()
            total_dist.append(dist)
        scheduler.step()
        total_dist = torch.cat(total_dist, dim=0)
        if (epoch % 10 == 0) or (epoch > epochs - 10):
            R = torch.tensor(get_radius(total_dist, nu), device=device)
            
            test_auc, temp_auc = test_stage_DOHSC_loader(test_loader, model,c, R, device, train_class, temp_auc)
            model.train()
            R = 0.0
        pbar.set_description(
            "Epoch{}| dist: max{:.4}, median{:.4}, min{:.4}| Mutual Loss{:.4} | SVDD Loss{:.4}".format(
                epoch,
                float(torch.max(torch.sqrt(total_dist)).detach()),
                float(torch.median(torch.sqrt(total_dist)).detach()),
                float(torch.min(torch.sqrt(total_dist)).detach()),
                loss_all,
                svdd_all
            )
        )
    return temp_auc, total_dist


def train_DO2HSC(model, train_loader, test_loader, c, nu, epochs, lam, train_class, lr_milestones, lr, device):
    temp_auc = -1.
    test_auroc = -1.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=lr_milestones, gamma=0.1)
    test_auc, total_dist = train_DOHSC(model, train_loader, test_loader, c, nu, epochs=5, lam=1, train_class=train_class, lr_milestones=None, lr=lr,
                                       device=device)

    R_max, R_min = torch.tensor(get_radius_bi_hypersphere(total_dist, nu), device=device)
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        loss_all = 0
        svdd_all = 0
        total_dist = []
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, pro_emb = model(data.x, data.edge_index, data.batch, data.num_graphs)
            svdd_loss, dist = calculate_hypersphere_loss(pro_emb, c, R_max, R_min, device)
            loss_all += loss.item()
            svdd_all += svdd_loss.item()
            total_loss = loss + lam * svdd_loss
            total_loss.backward()
            optimizer.step()
            total_dist.append(dist)
        scheduler.step()
        total_dist = torch.cat(total_dist, dim=0)
        if (epoch % 10 == 0) | (epoch > epochs - 100):
            test_auroc = test_stage_DO2HSC_loader(test_loader, model, c, R_max, R_min, device,
                                                                      train_class, temp_auc)
            model.train()
        pbar.set_description(
            "Train_{} Epoch{}| dist: max{:.4}, median{:.4}, min{:.4}| Mutual Loss{:.4} | SVDD Loss{:.4}".format(
                train_class,
                epoch,
                float(torch.max(torch.sqrt(total_dist)).detach()),
                float(torch.median(torch.sqrt(total_dist)).detach()),
                float(torch.min(torch.sqrt(total_dist)).detach()),
                loss_all,
                svdd_all)
        )
    return temp_auc
