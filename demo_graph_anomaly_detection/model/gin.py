from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, LinearSVC
from torch.nn import Sequential, Linear, LeakyReLU, SELU, ELU, CELU, PReLU, Tanhshrink, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from tqdm import tqdm
import numpy as np
import os.path as osp
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn


class NodeNorm(nn.Module):
    def __init__(self, nn_type="n", unbiased=False, eps=1e-5, power_root=2):
        super(NodeNorm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps
        self.nn_type = nn_type
        self.power = 1 / power_root

    def forward(self, x):
        if self.nn_type == "n":
            # print(x)
            mean = torch.mean(x, dim=1, keepdim=True)
            std = (
                torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = (x - mean) / std
        elif self.nn_type == "v":
            std = (
                torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / std
        elif self.nn_type == "m":
            mean = torch.mean(x, dim=1, keepdim=True)
            x = x - mean
        elif self.nn_type == "srv":  # squre root of variance
            std = (
                torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / torch.sqrt(std)
        elif self.nn_type == "pr":
            std = (
                torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / torch.pow(std, self.power)
        return x

    def __repr__(self):
        original_str = super().__repr__()
        components = list(original_str)
        nn_type_str = f"nn_type={self.nn_type}"
        components.insert(-1, nn_type_str)
        new_str = "".join(components)
        return new_str


def get_normalization(norm_type, num_channels=None):
    if norm_type is None:
        norm = None
    elif norm_type == "batch":
        norm = nn.BatchNorm1d(num_features=num_channels)
    elif norm_type == "node_n":
        norm = NodeNorm(nn_type="n")
    elif norm_type == "node_v":
        norm = NodeNorm(nn_type="v")
    elif norm_type == "node_m":
        norm = NodeNorm(nn_type="m")
    elif norm_type == "node_srv":
        norm = NodeNorm(nn_type="srv")
    elif norm_type.find("node_pr") != -1:
        power_root = norm_type.split("_")[-1]
        power_root = int(power_root)
        norm = NodeNorm(nn_type="pr", power_root=power_root)
    elif norm_type == "layer":
        norm = nn.LayerNorm(normalized_shape=num_channels)
    else:
        raise NotImplementedError
    return norm


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, latent_dim, mode, normalization=None):
        super(Encoder, self).__init__()

        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers

        self.mode = mode
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.node_norm = get_normalization(norm_type=normalization)
        if self.mode == 'pretrain':
            for i in range(num_gc_layers):
                if i:
                    nn = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
                else:
                    nn = Sequential(Linear(num_features, dim), LeakyReLU(), Linear(dim, dim))
                conv = GINConv(nn)
                bn = torch.nn.BatchNorm1d(dim)
                self.convs.append(conv)
                self.bns.append(bn)
        else:
            for i in range(num_gc_layers):
                if i:
                    nn = Sequential(Linear(dim, dim, bias=False), LeakyReLU(), Linear(dim, dim, bias=False))
                else:
                    nn = Sequential(Linear(num_features, dim, bias=False), LeakyReLU(), Linear(dim, dim, bias=False))
                conv = GINConv(nn)
                bn = torch.nn.BatchNorm1d(dim)

                self.convs.append(conv)
                self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)
        xs = []
        for i in range(self.num_gc_layers):
            x = F.leaky_relu(self.convs[i](x, edge_index))
            x = self.node_norm(x)
            x = self.bns[i](x)
            # print(self.mode)
            xs.append(x)
            # if i == 2:
            # feature_map = x2
        # print(xs.shape)
        xpool = [global_add_pool(x, batch) for x in xs]
        # xpool = [global_mean_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        # print(x.shape)
        # print(torch.cat(xs, 1).shape)
        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x_g, x_l = self.forward(x, edge_index, batch)
                ret.append(x_g.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        try:
            num_features = dataset.num_features
        except:
            num_features = 1
        dim = 32

        self.encoder = Encoder(num_features, dim)

        self.fc1 = Linear(dim * 5, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        x, _ = self.encoder(x, edge_index, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # print(data.x.shape)
        # [ num_nodes x num_node_labels ]
        # print(data.edge_index.shape)
        #  [2 x num_edges ]
        # print(data.batch.shape)
        # [ num_nodes ]
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    for percentage in [1.]:
        for DS in [sys.argv[1]]:
            if 'REDDIT' in DS:
                epochs = 200
            else:
                epochs = 100
            path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', DS)
            accuracies = [[] for i in range(epochs)]
            # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
            dataset = TUDataset(path, name=DS)  # .shuffle()
            num_graphs = len(dataset)
            print('Number of graphs', len(dataset))
            dataset = dataset[:int(num_graphs * percentage)]
            dataset = dataset.shuffle()

            kf = KFold(n_splits=10, shuffle=True, random_state=None)
            for train_index, test_index in kf.split(dataset):

                # x_train, x_test = x[train_index], x[test_index]
                # y_train, y_test = y[train_index], y[test_index]
                train_dataset = [dataset[int(i)] for i in list(train_index)]
                test_dataset = [dataset[int(i)] for i in list(test_index)]
                print('len(train_dataset)', len(train_dataset))
                print('len(test_dataset)', len(test_dataset))

                train_loader = DataLoader(train_dataset, batch_size=128)
                test_loader = DataLoader(test_dataset, batch_size=128)
                # print('train', len(train_loader))
                # print('test', len(test_loader))

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = Net().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(1, epochs + 1):
                    train_loss = train(epoch)
                    train_acc = test(train_loader)
                    test_acc = test(test_loader)
                    accuracies[epoch - 1].append(test_acc)
                    tqdm.write('Epoch: {:03d}, Train Loss: {:.7f}, '
                               'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                                            train_acc, test_acc))
            tmp = np.mean(accuracies, axis=1)
            print(percentage, DS, np.argmax(tmp), np.max(tmp), np.std(accuracies[np.argmax(tmp)]))
            input()
