import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='DOHSC and DO2HSC Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset', default='MUTAG')
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--path', dest='path', type=str, default=".\data")
    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
            help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=16,
            help='')
    parser.add_argument('--latent-dim', dest='latent_dim', type=int, default=8,
                        help='')
    parser.add_argument('--train_class', dest='train_class', type=int, default=0,
                        help='Trained class label')
    parser.add_argument('--pretrain', '-pt', type=bool, default=False, help='Pretrain to center')
    parser.add_argument('--repNum', dest='repNum', type=int,
                        help='Repeat number.', default=10)
    parser.add_argument('--nu', dest='nu', type=float,
                        help='', default=0.01)
    parser.add_argument('--percentage', dest='percentage', type=int,
                        help='',default=0.8)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4, help='weight decay for model')
    parser.add_argument('--lr_milestones', '-lr_mile', type=list, default=[400,450], help='learning rate milestones')
    parser.add_argument('--epochs', '-epochs', type=int, default=500,
                        help='total number of epochs')
    parser.add_argument('--lam', '-lam', type=float, default=1.,
                        help='trade-off factors of loss')
    parser.add_argument('--nodenorm_method', '-nodenorm_method', type=str, default="node_m",
                        help='method to normalize GIN')
    parser.add_argument('--eval', '-eval', type=bool, default=True, help='Load the saved pth file')

    return parser.parse_args()