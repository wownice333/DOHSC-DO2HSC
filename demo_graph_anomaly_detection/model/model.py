import numpy as np
import torch
import torch.nn as nn
from model.gin import Encoder
from model.losses import local_global_loss_
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Graph_Representation_Learning(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, latent_dim,dataset_num_features, device, mode='train', alpha=0.5, beta=1., gamma=.1):
        super(Graph_Representation_Learning, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mode = mode

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers, self.mode, latent_dim, 'node_m')
        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        self.projector = Orthogonal_Projector(self.embedding_dim, latent_dim)  #
        self.latent_dim = latent_dim
        self.device = device

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def get_result(self, loader):
        embedding = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(self.device)
                x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
                if x is None or len(x[1]) == 0:
                    x = torch.ones(batch.shape[0], 1).to(self.device)
                _, pro_x = self.forward(x, edge_index, batch, num_graphs)
                embedding.append(pro_x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        embedding = np.concatenate(embedding, 0)
        y = np.concatenate(y, 0)
        return embedding, y

    def forward(self, x, edge_index, batch, num_graphs, stage='train'):
        if x is None or len(x[1]) == 0:
            x = torch.ones(batch.shape[0], 1).to(self.device)
        y, M = self.encoder(x, edge_index, batch)
        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        # self.projector.apply(weightConstraint(g_enc))
        pro_x = self.projector(g_enc)
        mode = 'fd'
        measure = 'JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, batch, measure, self.device)

        return local_global_loss, pro_x


class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim,bias=False),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim,bias=False),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim,bias=False),
            nn.LeakyReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim,bias=False)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class Orthogonal_Projector(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim,bias=False),
        )

    def forward(self, x):
        return self.block(x)


class weightConstraint(object):
    def __init__(self, input):
        self.input=input
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data.t()
            d = w.shape[1]
            U,S,V=self.input.svd()
            S=S.diag()
            W=V[:,:d].matmul(torch.linalg.inv(S)[:d,:d])
            module.weight.data = W.t()
