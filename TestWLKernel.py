"""
=========================================================================
Graph classification on MUTAG using the Weisfeiler-Lehman subtree kernel.
=========================================================================

Script makes use of :class:`grakel.WeisfeilerLehman`, :class:`grakel.VertexHistogram`
"""
from __future__ import print_function

from sklearn.preprocessing import StandardScaler

print(__doc__)

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.model_selection import ShuffleSplit

# Loads the MUTAG dataset
test_dataset = {"1": "MUTAG", "2": "NCI109", "3": "PTC_FM", "4": "PTC_FR", "5": "PTC_MM", "6": "PTC_MR", "7": "PROTEINS"}

for key, value in test_dataset.items():
        dataset = fetch_dataset(value, verbose=False)
        G, y = dataset.data, dataset.target

        # Splits the dataset into a training and a test set
        G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=42)

        # Uses the Weisfeiler-Lehman subtree kernel to generate the kernel matrices
        gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
        K_train = gk.fit_transform(G_train)
        K_test = gk.transform(G_test)

        # Uses the SVM classifier to perform classification
        clf = SVC(kernel="precomputed")
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)

        # Computes and prints the classification accuracy
        acc = accuracy_score(y_test, y_pred)
        print("{}'s Accuracy: ".format(value), str(round(acc * 100, 2)) + "%")


# print(G,y)

# number_split = 8
# ss = ShuffleSplit(number_split, test_size=0.1, random_state=0)
#
# for train_index, test_index in ss.split(G, y):
#     X_train, Y_train = G[train_index], y[train_index]
#     X_test, Y_test = G[test_index], y[test_index]
#     gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
#     X_train = gk.fit_transform(X_train)
#     X_test = gk.transform(X_test)
#     classifier = SVC(kernel="precomputed")
#     classifier.fit(X_train, Y_train)
#     y_pred = classifier.predict(X_test)
#     acc = accuracy_score(Y_test, y_pred)
#     print("Accuracy:", str(round(acc * 100, 2)) + "%")
