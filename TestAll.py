from sklearn import svm

from Modif import *
import random
from kPCA import *
import numpy as np
dataset = "MUTAG"
original_features = count_motif_entropy(dataset)
original_labels = read_graph_label(dataset)

nodN = len(original_labels.tolist())

def sklearn_svm(features,labels,train_idx,test_idx):
    clf_linear = svm.SVC(kernel='linear')
    #print('train_idx: ',train_idx)
    #print('features[train]: ',features[train_idx])
    #print('labels[train_idx]',labels[train_idx])
    clf_linear.fit(features[train_idx], labels[train_idx])
    score_linear = clf_linear.score(features[test_idx], labels[test_idx])
    #print("The score of linear is : %f" % score_linear)
    return score_linear*100

max_level = 0
max_acc = 0
random_idx = [i for i in range(nodN)]
random.shuffle(random_idx)
for level in range(1, 9):
    # print(original_features[0])
    select_level_features = select_level(original_features, level)
    # kernel_features = js_kernel_process(features, level)
    accs = []
    for i in range(10):
        train_idx_temp, test_idx_temp = n_cross(10, i, nodN, random_idx)
        accs.append(sklearn_svm(select_level_features, original_labels, train_idx_temp, test_idx_temp))
        # accs.append(kernel_svm(kernel_features,train_idx_temp,test_idx_temp))
    avg = acc_calculator(accs)
    if avg > max_acc:
        max_level = level
        max_acc = avg
print('\n\n\n\ndataset: {}   acc: {}  best_level: {}'.format(dataset, max_acc, max_level))
print('\n\n\n\n\n')


def ten_ten_svm(l):
    accs = []
    #
    #
    #
    features = select_level(original_features, l)
    # kernel_features=js_kernel_process(features)
    #
    #
    #

    for k in range(10):
        temp_accs = []
        random_idx = [i for i in range(nodN)]
        random.shuffle(random_idx)
        for i in range(10):
            train_idx_temp, test_idx_temp = n_cross(10, i, nodN, random_idx)
            temp_accs.append(sklearn_svm(features, original_labels, train_idx_temp, test_idx_temp))
            # temp_accs.append(kernel_svm(kernel_features, original_labels,train_idx_temp, test_idx_temp))
        temp_res = acc_calculator(temp_accs)
        print('\n------temp_res: {} -------\n'.format(format(temp_res, '.2f')))
        accs.append(temp_res)

    print('\n\n\n--------------------------\n'
          '--------------------------\n'
          '----------result------------\n'
          '---------------------------\n'
          '---------------------------\n')
    print('dataset: ', dataset)
    acc_calculator(accs)


ten_ten_svm(max_level)
