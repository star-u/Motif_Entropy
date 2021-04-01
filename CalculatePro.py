from __future__ import print_function

from sklearn.preprocessing import StandardScaler

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset, get_dataset_info
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.model_selection import ShuffleSplit
from util import *
import torch
from tkinter import _flatten
from MotifCount import *
from entropy import *
import itertools
import operator
import collections

# Loads the MUTAG dataset
test_dataset = {"1": "DD"}

for key, value in test_dataset.items():
    dataset = fetch_dataset(value, verbose=False)
    G, y = dataset.data, dataset.target

data = read_data_txt(test_dataset["1"])
graph_ids = set(data['_graph_indicator.txt'])
ADJACENCY_SUFFIX = '_A.txt'
adj = data[ADJACENCY_SUFFIX]


# 提取相同value的key值存在列表中
def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


edge_index = 0
node_index_begin = 0


# print(len(adj))
def get_pro():
    global edge_index
    global node_index_begin
    label_pro = []
    for g_id in set(graph_ids):
        print('正在处理图：' + str(g_id))
        node_ids = np.argwhere(data['_graph_indicator.txt'] == g_id).squeeze()
        node_ids.sort()
        # global edge_index
        # global node_index_begin
        temp_nodN = len(node_ids)
        temp_A = np.zeros([temp_nodN, temp_nodN], int)
        while (edge_index < len(adj)) and (adj[edge_index][0] - 1 in node_ids):
            temp_A[adj[edge_index][0] - 1 - node_index_begin][adj[edge_index][1] - 1 - node_index_begin] = 1
            edge_index += 1

        print(temp_A)
        node_index_begin += temp_nodN
        # 求Motif entropy
        num_motif, num_node = count_Motifs(temp_A)
        # print(num_node)

        # 将字符串转化为数字
        temp = []
        for i in range(len(num_node)):
            s = re.split('', num_node[i])
            s.pop()
            s.pop(0)
            s = list(map(int, s))
            # print(s)
            temp.append(s)
        # print(temp)

        # 将节点出现在motif中的次数记为1
        a = np.zeros((len(num_node), Nm), float)
        for i, temp1 in enumerate(temp):
            for k in range(len(temp1)):
                a[i][temp1[k] - 1] = 1
        # print(a)

        # 统计节点出现在motif的总次数
        count_number_motif = []
        count_number_motif1 = 0
        for j in range(Nm):
            for i in range(len(num_node)):
                count_number_motif1 += a[i][j]
            count_number_motif.append(count_number_motif1)
            count_number_motif1 = 0
        # print(count_number_motif)
        a_tensor = torch.from_numpy(a)

        # 将motif的entropy除以出现的次数
        motif_entropy = graphEntropy(num_motif, len(num_node))
        # print(motif_entropy)
        entropy_count = []
        for i in range(Nm):
            if count_number_motif[i] != 0:
                entropy_count.append(motif_entropy[i] / count_number_motif[i])
            else:
                entropy_count.append(0)
        # print(entropy_count)

        # 求motif的entropy
        for i in range(Nm):
            for j in range(len(num_node)):
                if a[j][i] == 1:
                    a[j][i] = entropy_count[i]
        # print(a)

        # 依次分配到对应列值为1的向量中
        count1 = 0
        count_every_vector = []
        for i in range(len(num_node)):
            for j in range(Nm):
                count1 += a[i][j]
            count_every_vector.append(count1)
            count1 = 0
        # print(count_every_vector)

        # 求不同标签的节点概率
        # print(G[g_id - 1][1])
        max_label = max(G[g_id - 1][1].values())
        min_label = min(G[g_id - 1][1].values())

        label = [None] * (max_label + 1)
        key_list = [None] * (max_label + 1)
        for i in range(min_label, max_label + 1):
            label[i] = get_key(G[g_id - 1][1], i)
        # print(label)
        # print(get_key(G[g_id-1][1],i))
        # print(label)

        # 二维列表转化为一维列表
        min_min_number = min(list(_flatten(label)))
        # print(min_min_number)
        # print(min_min_number)
        # 计算不同标签节点的熵
        count_label = [0] * (max_label + 1)
        for i in range(min_label, max_label + 1):
            for m, j in enumerate(label[i]):
                count_label[i] += count_every_vector[j - min_min_number]
        # print(count_label)

        # 归一化
        label_one = [None] * (max_label + 1)
        for i in range(min_label, max_label + 1):
            label_one[i] = []
            for m, j in enumerate(label[i]):
                label_one[i].append(count_every_vector[j - min_min_number] / count_label[i])
        # print(label_one)
        label_pro.append(label_one)

        # if g_id > 2:
        #     break
    return label_pro


test = get_pro()

z = {}
for g_id in graph_ids:
    z = dict(itertools.chain(z.items(), G[g_id - 1][1].items()))
sorted_y = sorted(z.items(), key=operator.itemgetter(0))
z_label = dict(sorted_y)
# print(z_label)
# print(z)
z_label_lsit = []
for va in z_label.values():
    z_label_lsit.append(va)
# print(z_label_lsit)
a = {}
for i in z_label_lsit:
    a[i] = z_label_lsit.count(i)
# print(a)
# 计算value相同的key集合

max_label_2 = max(z.values())
key_list = [None] * (max_label_2 + 1)
# print(max_label_2)
for number_t in range(max_label_2 + 1):
    key_list[number_t] = []
    for key, value in z.items():
        if value == number_t:
            key_list[number_t].append(key)
    # print(len(z))
    # print(len(key_list[number_t]))
# print(key_list)
# print(graph_ids)

# for g_id in graph_ids:
#     print(test[g_id - 1])
#     if g_id >2:
#         break


concate_list = [None] * (max_label_2 + 1)
for i in range(max_label_2 + 1):
    concate_list[i] = []
    for g_id in graph_ids:
        # print(len(test[g_id-1]))
        # print(len(test[g_id-1]))
        if len(test[g_id - 1]) - 1 >= i and test[g_id - 1][i] is not None:
            concate_list[i].extend(test[g_id - 1][i])
        # else:
        #     continue
    # print(len(concate_list[i]))

# print(len(concate_list[2]))
# print(len(key_list[2]))

dict_k_c = [None] * (max_label_2 + 1)
for i in range(max_label_2 + 1):
    dict_k_c[i] = {}
    dict_k_c[i] = dict(zip(key_list[i], concate_list[i]))
    # print(len(dict_k_c[i]))

fu = {}
for i in range(max_label_2 + 1):
    fu = dict(itertools.chain(fu.items(), dict_k_c[i].items()))
# collections.OrderedDict(sorted(fu.items()))
sorted_x = sorted(fu.items(), key=operator.itemgetter(0))
# print(dict(sorted_x))
dict_pro = dict(sorted_x)
# print(len(dict_pro))
# 把标签和对应的概率写入文件
# file = open('data/DD/DD_label_pro.txt', 'w')
# for k, v in dict_pro.items():
#     file.write(str(z_label_lsit[k - 1]) + ' ' + str(v) + '\n')
#
# file.close()
