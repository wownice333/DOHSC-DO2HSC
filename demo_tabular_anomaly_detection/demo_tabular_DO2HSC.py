from numpy import loadtxt
from Utils.arguments import arg_parse
from model.model import  pretrain_autoencoder_mlp, DeepSVDDNetwork_mlp
from model.DOHSC import train_DOHSC, pretrain, train_DO2HSC
import torch
import os
import numpy as np
from Utils.loadData import loadData
from torch.utils.data import DataLoader
from model.evaluation import load_model_DO2HSC, load_model_DOHSC
import random

seed = 2021
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    args = arg_parse()

    datasetname = 'thyroid'
    # datasetname='arrhythmia'

    batch_size = args.batch_size
    lr = args.lr
    nu = args.nu
    R = 0.0
    repNum = args.repNum
    latent_dim = args.latent_dim
    epochs = args.num_epochs

    lr=args.lr
    lr_milestones=args.lr_milestones
    weight_decay=args.weight_decay

    pretrain_epochs = args.num_epochs_ae
    pretrain_lr = args.lr_ae
    pretrain_weight= args.weight_decay_ae


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_x, train_y, test_x, test_y=loadData(datasetname)

    train_dataset=torch.utils.data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataset=torch.utils.data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_dim=torch.tensor(train_x).shape[1]
    print('================')
    print('lr: {}'.format(lr))
    print('================')
    if args.eval == True:
        auclist=[]
        f1list=[]
        model = DeepSVDDNetwork_mlp(input_dim, latent_dim).to(device)
        state_dict = torch.load('./weights/DO2HSC/'+datasetname+'_DO2HSC' + '.pth')
        model.load_state_dict(state_dict['model'])
        c = torch.Tensor(state_dict['c']).to(device)
        R_max = state_dict['R_max']
        R_min = state_dict['R_min']
        test_auc, test_f1 = load_model_DO2HSC(test_loader, model, c, R_max, R_min, device)

        auclist.append(test_auc)
        f1list.append(test_f1)
        algorithm_name = 'DO2HSC'
        AUCmean_std = np.around([np.mean(auclist), np.std(auclist)], decimals=4)
        F1mean_std = np.around([np.mean(f1list), np.std(f1list)], decimals=4)
        print("Testing AUC Results:" + str(AUCmean_std))
        print("Testing F1-Score Results:" + str(F1mean_std))
        with open('./result/' + algorithm_name + '_' + datasetname + '_result.txt', 'a') as f:
            f.write('10 Times Result AUC:' + str(AUCmean_std) + ')\n')
            f.write('10 Times Result F1_score:' + str(F1mean_std) + '\n')
    else:
        for nu in [0.001,0.003,0.005,0.008,0.01,0.03,0.05,0.08]:
            auclist = np.zeros([repNum, 1])
            f1_list = np.zeros([repNum, 1])
            for rep in range(repNum):
                if not args.pretrain:
                    print("Load Pretrained Weights")
                    model=DeepSVDDNetwork_mlp(input_dim, latent_dim)
                    state_dict = torch.load('./weights/pretrained_weights.pth')
                    model.load_state_dict(state_dict['net_dict'])
                    c = torch.Tensor(state_dict['center']).to(device)
                    model=model.to(device)
                    # c = torch.randn(latent_dim).to(device)
                    # model = DeepSVDDNetwork_mlp(input_dim, latent_dim).to(device)
                else:
                    pretrain(train_loader,input_dim, latent_dim, pretrain_lr, pretrain_weight,lr_milestones,pretrain_epochs,device)
                    print('Pretraining and Load Pretrained Weights')
                    model=DeepSVDDNetwork_mlp(input_dim, latent_dim)
                    state_dict = torch.load('./pretrained_weight/pretrained_weights.pth')
                    model.load_state_dict(state_dict['net_dict'])
                    c = torch.Tensor(state_dict['center']).to(device)
                    model=model.to(device)
                # test_auc, _, _, f1_score = train_DOHSC(model, train_loader, test_loader, c, nu, epochs, args.normal_class, lr_milestones, lr, weight_decay, device)
                test_auc, f1_score = train_DO2HSC(model, train_loader, test_loader, c, nu, epochs, args.normal_class, lr_milestones, lr, weight_decay, datasetname, device)
                f1_list[rep]= f1_score
                auclist[rep] = test_auc
            algorithm_name='DO2HSC'
            AUCmean_std = np.around([np.mean(auclist), np.std(auclist)], decimals=4)
            F1mean_std = np.around([np.mean(f1_list), np.std(f1_list)], decimals=4)
            print("Testing Statistic Results:" + str(AUCmean_std)+' and:' +str(F1mean_std))
            with open('./result/'+algorithm_name+'_'+datasetname+'_result.txt', 'a') as f:
                f.write('Percentile:' + str(nu) + '\n')
                f.write('10 Times Result AUC:' + str(AUCmean_std) + '\n')
                f.write('10 Times Result F1_score:' + str(F1mean_std) + '\n')
