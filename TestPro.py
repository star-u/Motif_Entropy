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

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset, get_dataset_info
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.model_selection import ShuffleSplit
from util import *

# Loads the MUTAG dataset
test_dataset = {"1": "MUTAG", "2": "PROTEINS_full", "3": "PTC_FM", "4": "PTC_FR", "5": "PTC_MM", "6": "PTC_MR",
                "7": "NCI109", "8": "NCI1"}

for key, value in test_dataset.items():
    dataset = fetch_dataset(value, verbose=False)
    G, y = dataset.data, dataset.target
    # print(len(G))
    # print(G[0][1])
    # Splits the dataset into a training and a test set
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=42)
    # print((np.array(G_train)).shape)
    # Uses the Weisfeiler-Lehman subtree kernel to generate the kernel matrices
    gk = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram, normalize=True)
    K_train = gk.fit_transform(G_train)
    K_test = gk.transform(G_test)

    # Uses the SVM classifier to perform classification
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    # Computes and prints the classification accuracy
    acc = accuracy_score(y_test, y_pred)
    print("{}: {}'s Accuracy: ".format(key, value), str(round(acc * 100, 2)) + "%")