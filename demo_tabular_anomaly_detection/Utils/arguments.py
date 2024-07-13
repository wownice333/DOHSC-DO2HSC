import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Train DOHSC and DO2HSC model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_epochs', '-e', type=int, default=500, help='Num of epochs to DOHSC and DO2HSC train')
    parser.add_argument('--num_epochs_ae', '-ea', type=int, default=100, help='Num of epochs to AE model train')
    parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='learning rate for model')
    parser.add_argument('--lr_ae', '-lr_ae', type=float, default=1e-3, help='learning rate for AE model')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-6, help='weight decay for model')
    parser.add_argument('--weight_decay_ae', '-wd_ae', type=float, default=1e-6, help='weight decay for model')
    parser.add_argument('--lr_milestones', '-lr_mile', type=list, default=[80000], help='learning rate milestones')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='batch size')
    parser.add_argument('--pretrain', '-pt', type=bool, default=True, help='Pretrain to AE model')
    parser.add_argument('--eval', '-eval', type=bool, default=True, help='Load the saved pth file')
    parser.add_argument('--latent_dim', '-ld', type=int, default=16, help='latent dimension')
    parser.add_argument('--normal_class', '-cls', type=int, default=2, help='Set the normal class')
    parser.add_argument('--nu', dest='nu', type=float,
                        help='', default=0.01)
    parser.add_argument('--repNum', dest='repNum', type=int,
                        help='Repeat number.', default=10)

    args = parser.parse_args()
    return args