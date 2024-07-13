import scipy
import torch
import torch.nn as nn
from Utils.initialize import obtain_c, get_radius, get_radius_bi_hypersphere
from model.evaluation import test_stage_DOHSC_loader, test_stage_for_DO2HSC_loader
from model.losses import calculate_svdd_loss, calculate_hypersphere_loss
from model.model import DeepSVDDNetwork_mlp as DeepSVDDNetwork
from model.model import pretrain_autoencoder_mlp as pretrain_autoencoder
from tqdm import tqdm


def set_c(model, dataloader, device, eps=0.1):
    """Initializing the center for the hypersphere"""
    model.eval()
    z_ = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.float().to(device)
            z = model.encoder(x)
            z_.append(z.detach())
    z_ = torch.cat(z_)
    c = torch.mean(z_, dim=0)

    # If c is close to 0, set to +-eps
    # To avoid trivial problem
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


def pretrain(train_loader, input_dim, latent_dim, lr_ae, weight_decay_ae, lr_milestones, num_epochs_ae, device):
    """Pretrain AUTO ENCODER for using Deep SVDD"""
    ae = pretrain_autoencoder(input_dim, latent_dim).to(device)
    ae.apply(weights_init_normal)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr_ae,
                                 weight_decay=weight_decay_ae)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=lr_milestones, gamma=0.1)
    ae.train()

    for epoch in range(num_epochs_ae):
        total_loss = 0
        tq = tqdm(train_loader, total=len(train_loader))
        for x, _ in tq:
            x = x.float().to(device)

            optimizer.zero_grad()
            x_hat = ae(x)
            reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
            reconst_loss.backward()
            optimizer.step()

            total_loss += reconst_loss.item()
            errors = {
                'epoch': epoch,
                'train loss': reconst_loss.item()
            }

            tq.set_postfix(errors)

        epoch_loss = total_loss / len(train_loader)

    scheduler.step()

    save_weights_for_DeepSVDD(ae, train_loader, input_dim, latent_dim, device)


def save_weights_for_DeepSVDD(model, dataloader, input_dim, latent_dim, device):
    """Initializing for Deep SVDD's weights from pretrained AUTO ENCODER's weights"""
    c = set_c(model, dataloader, device)
    net = DeepSVDDNetwork(input_dim, latent_dim).to(device)
    state_dict = model.state_dict()
    net.load_state_dict(state_dict, strict=False)
    torch.save({'center': c.cpu().data.numpy().tolist(),
                'net_dict': net.state_dict()}, './pretrained_weight/pretrained_weights.pth')


def train_DOHSC(model, train_loader, test_loader, c, nu, epochs, train_class, lr_milestones, lr, weight_decay,
                dataset_name, device):
    R = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=lr_milestones, gamma=0.1)

    temp_auc = -1.
    test_auroc = -1.
    f1 = -1.
    temp_f1 = -1.
    model.train()
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        svdd_all = 0
        total_dist = []
        for data, _ in train_loader:
            data = data.float().to(device)
            optimizer.zero_grad()
            output = model(data)
            svdd_loss, dist = calculate_svdd_loss(output, c, R, nu, device)
            svdd_all += svdd_loss.item()
            total_loss = svdd_loss
            total_loss.backward()
            optimizer.step()
            total_dist.append(dist)
        scheduler.step()
        total_dist = torch.cat(total_dist, dim=0)
        epoch_loss = svdd_all / len(train_loader)
        if epoch % 10 == 0 | (epoch == epochs):
            R = torch.tensor(get_radius(total_dist, nu), device=device)
            test_auroc, temp_auc, f1, temp_f1 = test_stage_DOHSC_loader(test_loader, model, c, R, device,
                                                                        dataset_name, temp_auc, temp_f1)
            model.train()
            R = 0.0
        pbar.set_description(
            "Epoch{}| SVDD Loss{:.4}".format(
                epoch,
                epoch_loss
            )
        )
    return temp_auc, total_dist, model, temp_f1


def train_DO2HSC(model, train_loader, test_loader, c, nu, epochs, train_class, lr_milestones, lr, weight_decay,
                 dataset_name, device):
    temp_auc = -1.
    temp_f1 = -1.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=lr_milestones, gamma=0.1)
    _, total_dist, model, _ = train_DOHSC(model, train_loader, test_loader, c, nu, epochs=20, train_class=train_class,
                                          lr_milestones=None, lr=lr, weight_decay=weight_decay,
                                          dataset_name=dataset_name,
                                          device=device)

    R_max, R_min = torch.tensor(get_radius_bi_hypersphere(total_dist, nu), device=device)

    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        svdd_all = 0
        total_dist = []
        for data, _ in train_loader:
            data = data.float().to(device)
            optimizer.zero_grad()
            output = model(data)
            svdd_loss, dist = calculate_hypersphere_loss(output, c, R_max, R_min, device)
            svdd_all += svdd_loss
            svdd_loss.backward()
            optimizer.step()
            total_dist.append(dist)
        scheduler.step()
        total_dist = torch.cat(total_dist, dim=0)

        epoch_loss = svdd_all / len(train_loader)

        if (epoch % 10 == 0) | (epoch >= epochs - 10):
            test_auroc, temp_auc, f1_score, temp_f1 = test_stage_for_DO2HSC_loader(test_loader, model, c, R_max, R_min,
                                                                                   device,
                                                                                   dataset_name, temp_auc,
                                                                                   temp_f1)
            model.train()
        pbar.set_description(
            "Train_{} Epoch{}|  SVDD Loss{:.4}".format(
                train_class,
                epoch,
                epoch_loss)
        )
    return temp_auc, temp_f1
