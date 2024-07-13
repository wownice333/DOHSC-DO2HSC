import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Orthogonal_Projector(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim, bias=False),
        )

    def forward(self, x):
        return self.block(x)


class weightConstraint(object):
    def __init__(self, input):
        self.input = input

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data.t()
            d = w.shape[1]
            U, S, V = self.input.svd()
            S = S.diag()
            W = V[:, :d].matmul(S[:d, :d])
            module.weight.data = W.t()


class DeepSVDDNetwork_mlp(nn.Module):

    def __init__(self, input_dim, rep_dim=32):
        super(DeepSVDDNetwork_mlp, self).__init__()

        self.rep_dim = rep_dim
        self.input_dim= input_dim
        
        # self.ff1=nn.Linear(self.input_dim, self.rep_dim, bias=False)
        self.ff1=nn.Linear(self.input_dim, 500, bias=False)
        self.ff2 = nn.Linear(500, self.rep_dim, bias=False)
        self.ff3 = nn.Linear(self.rep_dim, self.rep_dim, bias=False)
        self.orthogonal_projector = Orthogonal_Projector(self.rep_dim, self.rep_dim)

    def forward(self,x):
        x=F.leaky_relu(self.ff1(x))
        x=self.ff2(x)
        x=self.ff3(x)
        self.orthogonal_projector.apply(weightConstraint(x))
        x=self.orthogonal_projector(x)
        return x

class  MLP_decoder(nn.Module):

    def __init__(self, input_dim, rep_dim=32):
        super(MLP_decoder, self).__init__()

        self.rep_dim = rep_dim
        self.input_dim= input_dim
        self.ff1=nn.Linear(self.rep_dim, self.input_dim, bias=False)
    def forward(self,x):
        x=self.ff1(x)
        return x

class pretrain_autoencoder_mlp(nn.Module):

    def __init__(self, input_dim, rep_dim=64):
        super(pretrain_autoencoder_mlp,self).__init__()
        self.rep_dim = rep_dim
        self.input_dim=input_dim
        self.encoder = DeepSVDDNetwork_mlp(rep_dim=rep_dim,input_dim=input_dim)
        self.decoder = MLP_decoder(rep_dim=rep_dim,input_dim=input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
