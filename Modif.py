import os

from MotifCount import *
from entropy import *

GRAPH_LABELS_SUFFIX = '_graph_labels.txt'
NODE_LABELS_SUFFIX = '_node_labels.txt'
ADJACENCY_SUFFIX = '_A.txt'
GRAPH_ID_SUFFIX = '_graph_indicator.txt'


def complete_path(folder, fname):
    return os.path.join(folder, fname)


def count_motif_entropy(dataset):
    data = dict()
    dataset_name = 'MUTAG'
    dataset_graph_reps = []
    dirpath = 'data/{}'.format(dataset_name)
    for f in os.listdir(dirpath):
        if "README" in f or '.txt' not in f:
            continue
        fpath = complete_path(dirpath, f)
        suffix = f.replace(dataset_name, '')
        print(suffix)
        if 'attributes' in suffix:
            data[suffix] = np.loadtxt(fpath, dtype=np.float, delimiter=',')
        else:
            data[suffix] = np.loadtxt(fpath, dtype=np.int, delimiter=',')

    graph_ids = set(data['_graph_indicator.txt'])
    # print(graph_ids) total how many graphs
    min_label = min(data[NODE_LABELS_SUFFIX])
    max_label = max(data[NODE_LABELS_SUFFIX])
    node_label_num = max_label - min_label + 1
    print('node labels number: ', node_label_num)

    adj = data[ADJACENCY_SUFFIX]
    edge_index = 0
    node_index_begin = 0
    # 控制循环
    b = 0
    for g_id in set(graph_ids):
        print('正在处理图：' + str(g_id))
        node_ids = np.argwhere(data['_graph_indicator.txt'] == g_id).squeeze()
        # print("node_id:", node_ids)
        node_ids.sort()

        temp_nodN = len(node_ids)
        temp_A = np.zeros([temp_nodN, temp_nodN], int)
        # print('nodN: ',temp_nodN)
        while (edge_index < len(adj)) and (adj[edge_index][0] - 1 in node_ids):
            temp_A[adj[edge_index][0] - 1 - node_index_begin][adj[edge_index][1] - 1 - node_index_begin] = 1
            edge_index += 1

        # print(temp_A)

        node_labels = data[NODE_LABELS_SUFFIX][node_ids]

        temp_node_labels = data[NODE_LABELS_SUFFIX][node_index_begin:node_index_begin + temp_nodN]

        temp_node_rep = cal_graph(temp_A, temp_nodN, temp_node_labels, min_label, max_label)
        dataset_graph_reps.extend(temp_node_rep)

        node_index_begin += temp_nodN
        #
        # 限制运行个数
        # b += 1
        # if b > 3:
        #     break
    # print(np.array(dataset_graph_reps))
    return np.array(dataset_graph_reps)

def cal_graph(adj_original, nodN, temp_node_labels, min_label, max_label):
    motif_of_nodes = []
    graph_rep = np.array([])
    node_entro = []
    for index in range(nodN):
        temp_motif, _ = count_Motifs(adj_original)
        motif_of_nodes.append(np.array(temp_motif))

    motif_of_nodes = np.array(motif_of_nodes)
    _, dim = motif_of_nodes.shape

    for temp_label in range(min_label, max_label + 1):
        nodes_reps = motif_of_nodes[temp_node_labels == temp_label]
        if len(nodes_reps) == 0:
            summation = np.zeros((1, dim))
        else:
            summation = np.sum(nodes_reps, axis=0).reshape(1, dim)

        temp_entropy = graphlet_entropy(summation[0])
        graph_rep = np.append(graph_rep, np.array(temp_entropy))
        num_motif, num_node = count_Motifs(adj_original)
        # print(num_motif)
        # print(np.array(num_motif),np.array(num_node))
        # 将字符串转化为数字
        temp = []
        for i in range(len(num_node)):
            s = re.split('', num_node[i])
            s.pop()
            s.pop(0)
            # s1 = [element.strip('') for element in s]
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
        # a_tensor = torch.from_numpy(a)

        # 求motif的entropy,将motif的entropy除以出现的次数
        motif_entropy = graphEntropy(num_motif, len(num_node))
        # print(motif_entropy)
        entropy_count = []
        for i in range(Nm):
            if count_number_motif[i] != 0:
                entropy_count.append(motif_entropy[i] / count_number_motif[i])
            else:
                entropy_count.append(0)
        # node_entro.append(entropy_count)

    # print(entropy_count)
    return np.array(entropy_count)












def read_graph_label(dataset):
    print('reading data labels...')
    filename = 'data/{}/{}_graph_labels.txt'.format(dataset, dataset)
    graph_labels = np.loadtxt(filename, dtype=np.float, delimiter=',')
    return graph_labels

# count_motif_entropy("MUTAG")

def n_cross(n,index,nodN,random_idx):

    test_start=index*(nodN//n)
    test_end=test_start+(nodN//n)
    train_idx = random_idx[:test_start]+random_idx[test_end:]
    test_idx = random_idx[test_start:test_end]

    #print('test_idx from {} to {}'.format(test_start,test_end))
    return np.array(train_idx),np.array(test_idx)

def acc_calculator(accs):
    n = len(accs)
    print('acc number: ', n)
    avg = sum(accs) / n
    print('avg: ', avg)
    print('max: ', max(accs))
    print('min: ', min(accs))
    max_dis = max(accs) - avg
    min_dis = avg - min(accs)
    if max_dis > min_dis:
        print('distance:', max_dis)
    else:
        print('distance:', min_dis)
    return avg