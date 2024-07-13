import numpy as np
import os
import random
import torch
from numpy import loadtxt

from Utils.arguments import arg_parse
from Utils.cifar10 import CIFAR10_Dataset
from Utils.dataloader import get_fmnist
from model.DOHSC import train_DOHSC_cifar, pretrain_cifar, train_DO2HSC_cifar
from model.model import pretrain_autoencoder_cifar, DeepSVDDNetwork_cifar_ELU
from model.evaluation import load_model_DO2HSC

seed = 2021
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    args = arg_parse()
    batch_size = args.batch_size
    lr = args.lr
    nu = args.nu
    R = 0.0
    repNum = args.repNum
    latent_dim = args.latent_dim
    epochs = args.num_epochs

    lr = args.lr
    lr_milestones = args.lr_milestones
    weight_decay = args.weight_decay

    pretrain_epochs = args.num_epochs_ae
    pretrain_lr = args.lr_ae
    pretrain_weight = args.weight_decay_ae
    #
    auclist = np.zeros([repNum, 1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # nu = 0.003#args.nu
    for j in [8]:
        args.normal_class = j
        dataset = CIFAR10_Dataset(root="./data", normal_class=args.normal_class)
        train_loader, test_loader = dataset.loaders(batch_size=args.batch_size)
        if args.eval == True:
            auclist=[]
            model = DeepSVDDNetwork_cifar_ELU(latent_dim).to(device)
            state_dict = torch.load('./weights/DO2HSC/DO2HSC_' + str(args.normal_class) + '.pth')
            model.load_state_dict(state_dict['model'])
            c = torch.Tensor(state_dict['c']).to(device)
            R_max = state_dict['R_max']
            R_min = state_dict['R_min']
            test_auc = load_model_DO2HSC(test_loader, model, c, R_max, R_min, device)

            auclist.append(test_auc)
            algorithm_name = 'Bi-Hypersphere'
            AUCmean_std = np.around([np.mean(auclist), np.std(auclist)], decimals=4)
            print("Testing Statistic Results:" + str(AUCmean_std))
            with open('./result/' + algorithm_name + '_cifar10_result.txt', 'a') as f:
                f.write('Normal Class:' + str(j) + '\n')
                f.write('Percentile:' + str(args.nu) + '\n')
                f.write('10 Times Result:' + str(AUCmean_std[0]) + ' (' + str(AUCmean_std[1]) + ')\n')
        else:
            for nu in [0.001,0.003,0.005,0.008,0.01,0.03,0.05,0.08,0.1,0.3,0.5]:
                print('================')
                print('lr: {}'.format(lr))
                print('================')
                for rep in range(repNum):
                    if not args.pretrain:
                        c = torch.randn(latent_dim).to(device)
                        model = DeepSVDDNetwork_cifar_ELU(latent_dim).to(device)
                        state_dict = torch.load('./pretrained_weights/' + str(args.normal_class) + '_pretrained_weights.pth')
                        model.load_state_dict(state_dict['net_dict'])
                        c = torch.Tensor(state_dict['center']).to(device)
                        model = model.to(device)
                    else:
                        pretrain_cifar(train_loader, latent_dim, pretrain_lr, pretrain_weight, lr_milestones, pretrain_epochs,
                                       args.normal_class, device)
                        print('Pretraining and Load Pretrained Weights')
                        model = DeepSVDDNetwork_cifar_ELU(latent_dim)
                        state_dict = torch.load('./pretrained_weights/' + str(args.normal_class) + '_pretrained_weights.pth')
                        model.load_state_dict(state_dict['net_dict'])
                        c = torch.Tensor(state_dict['center']).to(device)
                        model = model.to(device)

                    test_auc = train_DO2HSC_cifar(model, train_loader, test_loader, c, nu, epochs, args.normal_class,
                                                      lr_milestones, lr, weight_decay, device)
                    auclist[rep] = test_auc
                algorithm_name = 'Bi-Hypersphere'
                # algorithm_name='Hypersphere'
                AUCmean_std = np.around([np.mean(auclist), np.std(auclist)], decimals=4)
                print("Testing Statistic Results:" + str(AUCmean_std))
                with open('./result/' + algorithm_name + '_cifar10_result.txt', 'a') as f:
                    f.write('Normal Class:' + str(j) + '\n')
                    f.write('Percentile:' + str(args.nu) + '\n')
                    f.write('10 Times Result:' + str(AUCmean_std[0]) + ' (' + str(AUCmean_std[1]) + ')\n')
