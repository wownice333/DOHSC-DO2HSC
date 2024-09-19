import math
import random
import numpy as np


def split_data(dataset, DS, trained_label, percent):
    test_idx = []
    normal_class = trained_label
    abnormal_class=list(set(dataset.data.y.tolist()).difference(set([normal_class])))
    abnormal_num=[]
    normal_class_idx=np.where(np.array(dataset.data.y.tolist())==normal_class)
    normal_num=len(normal_class_idx[0])
    for ab in abnormal_class:
        abnormal_num.append(len(np.where(np.array(dataset.data.y.tolist())==ab)[0]))
    train_sample_num = math.ceil(normal_num*percent)
    train_idx = random.sample(list(normal_class_idx[0]), train_sample_num)

    retain_train_idx = list(set(normal_class_idx[0]).difference(set(train_idx)))

    test_sample_num = min(min(abnormal_num), len(retain_train_idx))
    for ab in abnormal_class:
        temp_test_idx=np.where(np.array(dataset.data.y.tolist()) == ab)
        test_idx.extend(random.sample(list(temp_test_idx[0]),test_sample_num))
    test_idx.extend(retain_train_idx)
    np.savetxt('./data/' + DS + '/' + DS + '/test_idx_' + str(trained_label) + '.txt', test_idx, fmt='%d')
    np.savetxt('./data/' + DS + '/' + DS + '/train_idx_'+str(trained_label)+'.txt', train_idx, fmt='%d')
    return np.array(train_idx).astype(dtype=int).tolist(), np.array(test_idx).astype(dtype=int).tolist()