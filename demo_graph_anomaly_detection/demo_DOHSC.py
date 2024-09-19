from numpy import loadtxt

from model.model import Graph_Representation_Learning
from model.DOHSC import train_DOHSC, pretrain
from Utils.arguments import arg_parse
import os.path as osp
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import torch
import os
from Utils.split_data import *
from model.evaluation import test_stage_DOHSC_loader
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    args = arg_parse()
    batch_size = 128
    lr = args.lr
    DS = args.DS
    nu = args.nu
    R = 0.0
    repNum = args.repNum
    percentage = args.percentage
    hidden_dim = args.hidden_dim
    num_gc_layers = args.num_gc_layers
    latent_dim = args.latent_dim
    epochs = args.epochs
    lam = args.lam
    lr=args.lr
    lr_milestones=args.lr_milestones
    
    path = osp.join(osp.dirname(osp.realpath(__file__)),'data', DS)
    dataset = TUDataset(path, name=DS)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    for tc in np.unique(dataset.data.y):
        args.train_class=train_class = tc
        if not os.path.exists(
                './data/' + DS + '/' + DS + '/test_idx_' + str(train_class) + '.txt') or not os.path.exists(
            './data/' + DS + '/' + DS + '/train_idx_' + str(train_class) + '.txt'):
            print('Split Data')
            train_idx, test_idx = split_data(dataset, DS, train_class, percentage)
        else:
            train_idx = np.array(
                (loadtxt('./data/' + DS + '/' + DS + '/train_idx_' + str(train_class) + '.txt'))).astype(
                dtype=int).tolist()
            test_idx = np.array(
                (loadtxt('./data/' + DS + '/' + DS + '/test_idx_' + str(train_class) + '.txt'))).astype(
                dtype=int).tolist()
        train_dataset = dataset[train_idx]
        print('len(train_dataset)', len(train_dataset))
        test_dataset = dataset[test_idx]
        print('len(test_dataset)', len(test_dataset))
        dataset_num_features = max(dataset.num_features, 1)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        print('================')
        print('lr: {}'.format(lr))
        print('num_features: {}'.format(dataset_num_features))
        print('hidden_dim: {}'.format(args.hidden_dim))
        print('num_gc_layers: {}'.format(args.num_gc_layers))
        print('================')
        if args.eval == True:
            auclist=[]
            f1list=[]
            model = Graph_Representation_Learning(hidden_dim, num_gc_layers, latent_dim, dataset_num_features, device, mode='train').to(device)
            state_dict = torch.load('./weights/DOHSC/'+DS+'_'+str(train_class)+'_model' + '.pth', map_location=device)

            model.load_state_dict(state_dict['model'])
            c = torch.Tensor(state_dict['c']).to(device)
            R = state_dict['R']
            test_auc, _ = test_stage_DOHSC_loader(test_loader, model, c, R, device, train_class)

            auclist.append(test_auc)
            algorithm_name = 'DOHSC'
            AUCmean_std = np.around([np.mean(auclist), np.std(auclist)], decimals=4)

            print("Testing AUC Results:" + str(AUCmean_std))

            with open('./result/' + algorithm_name + '_' + DS + '_result.txt', 'a') as f:
                f.write('10 Times Result AUC:' + str(AUCmean_std) + '\n')
        else:
            for nu in [0.001,0.003,0.005,0.008,0.01,0.03,0.05,0.08]: # 
                auclist = np.zeros([repNum, 1])
                f1_list = np.zeros([repNum, 1])
                for rep in range(repNum):
                    if not args.pretrain:
                        c = torch.randn(latent_dim).to(device)
                    else:
                        c = pretrain(DS, train_loader, lr)
                        print('Pretraining and Load Pretrained Weights')
                    model = Graph_Representation_Learning(hidden_dim, num_gc_layers, latent_dim, dataset_num_features, device, mode='train').to(device)
                    test_auc, _ = train_DOHSC(model, train_loader, test_loader, c, nu, epochs, lam, train_class, lr_milestones, lr, device)
                    auclist[rep] = test_auc
                AUCmean_std = np.around([np.mean(auclist), np.std(auclist)], decimals=4)
                print("Testing Statistic Results:" + str(AUCmean_std))
                with open('./result/' + args.DS + '_result.txt', 'a') as f:

                    f.write(args.DS + '_train_class:' + str(args.train_class) + '\n')
                    f.write(args.DS + '_batch_size:' + str(batch_size) + '\n')
                    f.write(args.DS + '_num_gc_layers:' + str(args.num_gc_layers) + '\n')
                    f.write(args.DS + '_hidden_dim:' + str(args.hidden_dim) + '\n')
                    f.write(args.DS + '_percentage:' + str(percentage) + '\n')

                    f.write('AUC:' + str(AUCmean_std[0]) + '$\pm$' + str(AUCmean_std[1]) + '\n')
