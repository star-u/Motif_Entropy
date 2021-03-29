"""
=========================================================================
Graph classification on different dataset using the Weisfeiler-Lehman subtree kernel.
=========================================================================

Script makes use of :class:`grakel.WeisfeilerLehman`, :class:`grakel.VertexHistogram`
"""
from __future__ import print_function

from sklearn.preprocessing import StandardScaler

# print(__doc__)

import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset, get_dataset_info
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.model_selection import ShuffleSplit
from util import *


def n_cross(n,index,nodN,random_idx):

    test_start=index*(nodN//n)
    test_end=test_start+(nodN//n)
    train_idx = random_idx[:test_start]+random_idx[test_end:]
    test_idx = random_idx[test_start:test_end]

    #print('test_idx from {} to {}'.format(test_start,test_end))
    return np.array(train_idx),np.array(test_idx)

# Loads the MUTAG dataset
test_dataset = {"1": "MUTAG", "2": "PROTEINS_full", "3": "PTC_FM", "4": "PTC_FR", "5": "PTC_MM", "6": "PTC_MR"}
                # "7": "NCI109", "8": "NCI1"}
Random_state = [35, 46, 35, 35, 38, 44]
    # , 88, 38]

# for k in range(10):
#     temp_accs = []
#     random_idx = [i for i in range(nodN)]
#     random.shuffle(random_idx)
#     for i in range(10):
#         train_idx_temp, test_idx_temp = n_cross(10, i, nodN, random_idx)

kf = KFold(n_splits=10)
def K_Flod_spilt(K,fold,data,label):
    '''
    :param K: 要把数据集分成的份数。如十次十折取K=10
    :param fold: 要取第几折的数据。如要取第5折则 flod=5
    :param data: 需要分块的数据
    :param label: 对应的需要分块标签
    :return: 对应折的训练集、测试集和对应的标签
    '''
    split_list = []
    kf = KFold(n_splits=K)
    for train, test in kf.split(data):
        split_list.append(train.tolist())
        split_list.append(test.tolist())
    train,test=split_list[2 * fold],split_list[2 * fold + 1]
    return  data[train], data[test], label[train], label[test]

f = open('Accuracy_4.txt', 'a')
for iter_number in [2]:
    f.write("-----" + str(iter_number) + "----------------\n")

    for key, value in test_dataset.items():
        dataset = fetch_dataset(value, verbose=False)
        G, y = dataset.data, dataset.target
        # Splits the dataset inweixin_44616888/article/details/105756203to a training and a test set
        G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1,
                                                            random_state=Random_state[int(key) - 1])
        # print((np.array(G_train)).shape)
        # print(G_train,G_test,y_train,y_test)
        # Uses the Weisfeiler-Lehman subtree kernel to generate the kernel matrices 2,10,20
        gk = WeisfeilerLehman(n_iter=iter_number, base_graph_kernel=VertexHistogram, normalize=True)
        K_train = gk.fit_transform(G_train)
        K_test = gk.transform(G_test)

        # Uses the SVM classifier to perform classification
        clf = SVC(kernel="precomputed")
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)

        # Computes and prints the classification accuracy
        acc = accuracy_score(y_test, y_pred)

        f.write(str(key) + ": " + str(value) + "'s Accuracy: " + str(round(acc * 100, 2)) + "%\n")
f.close()
