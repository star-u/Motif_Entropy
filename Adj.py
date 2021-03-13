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

from MotifCount import *
from entropy import *

# Loads the MUTAG dataset
test_dataset = {"1": "MUTAG"}

for key, value in test_dataset.items():
    dataset = fetch_dataset(value, verbose=False)
    G, y = dataset.data, dataset.target

data = read_data_txt(value)
graph_ids = set(data['_graph_indicator.txt'])
ADJACENCY_SUFFIX = '_A.txt'
adj = data[ADJACENCY_SUFFIX]
edge_index = 0

node_index_begin = 0


# 提取相同value的key值存在列表中
def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


# print(len(adj))
for g_id in set(graph_ids):
    print('正在处理图：' + str(g_id))
    node_ids = np.argwhere(data['_graph_indicator.txt'] == g_id).squeeze()
    node_ids.sort()

    temp_nodN = len(node_ids)
    temp_A = np.zeros([temp_nodN, temp_nodN], int)
    while (edge_index < len(adj)) and (adj[edge_index][0] - 1 in node_ids):
        temp_A[adj[edge_index][0] - 1 - node_index_begin][adj[edge_index][1] - 1 - node_index_begin] = 1
        edge_index += 1

    # print(temp_A)
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
    print(count_number_motif)
    a_tensor = torch.from_numpy(a)

    # 将motif的entropy除以出现的次数
    motif_entropy = graphEntropy(num_motif, len(num_node))
    print(motif_entropy)
    entropy_count = []
    for i in range(Nm):
        if count_number_motif[i] != 0:
            entropy_count.append(motif_entropy[i] / count_number_motif[i])
        else:
            entropy_count.append(0)
    print(entropy_count)

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
    print(count_every_vector)

    # 求不同标签的节点概率
    print(G[g_id - 1][1])
    max_label = max(G[g_id - 1][1].values())
    min_label = min(G[g_id - 1][1].values())

    label = [None] * (max_label + 1)
    for i in range(min_label, max_label + 1):
        label[i] = get_key(G[g_id - 1][1], i)
        # print(get_key(G[g_id-1][1],i))
    # print(label)

    # 计算不同标签节点的熵
    count_label = [0] * (max_label + 1)
    for i in range(min_label, max_label + 1):
        for m, j in enumerate(label[i]):
            count_label[i] += count_every_vector[j - 1]
    print(count_label)

    # 归一化
    label_one = [None] * (max_label + 1)
    for i in range(min_label, max_label + 1):
        label_one[i] = []
        for m, j in enumerate(label[i]):
            label_one[i].append(count_every_vector[j - 1] / count_label[i])
    print(label_one)

    if g_id > 0:
        break
