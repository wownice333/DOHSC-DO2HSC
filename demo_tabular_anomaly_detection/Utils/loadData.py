from scipy.io import loadmat
import numpy as np
from sklearn import datasets
def loadData(datasetname):
    dataset=loadmat('./data/'+datasetname+'.mat')
    # dataset=datasets.load_
    features=dataset['X']
    # print(features.shape)
    labels = dataset['y']
    # print(np.where(labels==0)[0])
    normal_data = features[np.where(labels==0)[0]][:]
    # print(normal_data)
    normal_labels = labels[np.where(labels==0)[0]]

    n_train = int(normal_data.shape[0]*0.5)
    ixs = np.arange(normal_data.shape[0])
    np.random.shuffle(ixs)
    normal_data_test = normal_data[ixs[n_train:]]
    normal_labels_test = normal_labels[ixs[n_train:]]

    train_x = normal_data[ixs[:n_train]]
    train_y = normal_labels[ixs[:n_train]]
    anomalous_data = features[np.where(labels==1)[0]][:]
    anomalous_labels = labels[np.where(labels==1)[0]]
    test_x = np.concatenate((anomalous_data, normal_data_test), axis=0)
    test_y = np.concatenate((anomalous_labels, normal_labels_test), axis=0)
    if datasetname=='thyroid':
        mean=np.mean(train_x,0)
        std=np.std(train_x,0)
        train_x=(train_x-mean)/ (std + 1e-4)
        test_x = (test_x - mean)/(std + 1e-4)
    print('training sample number:'+str(train_x.shape[0]))
    print('testing sample number:'+str(test_x.shape[0]))
    return train_x, train_y, test_x, test_y