"""
=========================================================================
Graph classification on different dataset using the Weisfeiler-Lehman subtree kernel.
=========================================================================

Script makes use of :class:`grakel.WeisfeilerLehman`, :class:`grakel.VertexHistogram`
"""
from __future__ import print_function

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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

# Loads the MUTAG dataset
test_dataset = {"1": "MUTAG", "2": "PROTEINS_full", "3": "PTC_FM", "4": "PTC_FR", "5": "PTC_MM", "6": "PTC_MR"}
# ,"7": "NCI109", "8": "NCI1"}

def K_Flod_spilt(K, fold, data, label):
    '''
    :param K: 要把数据集分成的份数。如十次十折取K=10
    :param fold: 要取第几折的数据。如要取第5折则 flod=5
    :param data: 需要分块的数据
    :param label: 对应的需要分块标签
    :return: 对应折的训练集、测试集和对应的标签
    '''
    split_list = []
    kf = ShuffleSplit(K, random_state=44)
    for train, test in kf.split(data):
        split_list.append(train.tolist())
        split_list.append(test.tolist())
    train, test = split_list[2 * fold], split_list[2 * fold + 1]
    return data[train], data[test], label[train], label[test]

split =10
f = open('Accuracy_mean_2.txt', 'a')
temp_accs = [None] * 6
for iter_number in [2]:
    f.write("-----" + str(iter_number) + "----------------\n")

    for key, value in test_dataset.items():
        dataset = fetch_dataset(value, verbose=False)
        G, y = dataset.data, dataset.target
        temp_accs[int(key) - 1] = []
        for i in range(split):

            G_train, G_test, y_train, y_test = K_Flod_spilt(split, i, np.array(G), np.array(y))
            gk = WeisfeilerLehman(n_iter=iter_number, base_graph_kernel=VertexHistogram, normalize=True)
            K_train = gk.fit_transform(G_train)
            K_test = gk.transform(G_test)

            # Uses the SVM classifier to perform classification
            # clf = RandomForestClassifier(n_estimators=35, random_state=39)
            # clf = AdaBoostClassifier(n_estimators=35, random_state=44)
                # SVC(kernel="precomputed")
            clf = SVC(kernel="poly")
            clf.fit(K_train, y_train)
            y_pred = clf.predict(K_test)

            # Computes and prints the classification accuracy
            acc = accuracy_score(y_test, y_pred)
            temp_accs[int(key) - 1].append(acc)
        # print(str(key) + ": " + str(value) + "'s Accuracy: " + str(
        #     round(np.mean(temp_accs[int(key) - 1]) * 100, 2)) + "%\n")

        f.write(str(key) + ": " + str(value) + "'s Accuracy: " + str(round(np.mean(temp_accs[int(key) - 1]) * 100, 2)) + "%\n")
f.close()
