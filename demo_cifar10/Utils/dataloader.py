import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image
from torchvision import datasets, transforms


class MNIST_loader(data.Dataset):
    """Preprocessing을 포함한 dataloader를 구성"""

    def __init__(self, data, target, transform):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


def get_fmnist(args, dir = './data'):
    """get dataloders"""
    # min, max values for each class after applying GCN (as the original implementation)
    mean = [
        [0.3256056010723114],
        [0.22290456295013428],
        [0.376699835062027],
        [0.25889596343040466],
        [0.3853232264518738],
        [0.1367349475622177],
        [0.3317836821079254],
        [0.16769391298294067],
        [0.35355499386787415],
        [0.30119451880455017]
    ]
    std = [
        [0.35073918104171753],
        [0.34353047609329224],
        [0.3586803078651428],
        [0.3542196452617645],
        [0.37631189823150635],
        [0.26310813426971436],
        [0.3392786681652069],
        [0.29478660225868225],
        [0.3652712106704712],
        [0.37053292989730835]
    ]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5],
                                                         std=[0.5])])

    train = datasets.FashionMNIST(root = dir, train=True, download=True)
    test = datasets.FashionMNIST(root = dir, train=False, download=True)

    # train = datasets.MNIST(root=dir, train=True, download=True)
    # test = datasets.MNIST(root=dir, train=False, download=True)

    x_train = train.data
    y_train = train.targets
    # print(np.unique(y_train))

    x_train = x_train[np.where(y_train == args.normal_class)]
    y_train = y_train[np.where(y_train == args.normal_class)]
    print(args.normal_class)
    data_train = MNIST_loader(x_train, y_train, transform)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)

    x_test = test.data
    y_test = test.targets

    # Normal class인 경우 0으로 바꾸고, 나머지는 1로 변환 (정상 vs 비정상 class)
    y_test = np.where(y_test == args.normal_class, 0, 1)

    data_test = MNIST_loader(x_test, y_test, transform)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)
    return dataloader_train, dataloader_test


def global_contrast_normalization(x):
    """Apply global contrast normalization to tensor. """
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean
    x_scale = torch.mean(torch.abs(x))
    x /= x_scale
    return x
