import numpy as np
import os


def complete_path(folder, fname):
    return os.path.join(folder, fname)


def read_graph_label(dataset):
    print('reading data labels...')
    if dataset=='PPI':
        return read_ppi_graph_label()
    elif dataset=='PTC':
        return read_ptc_labels()
    filename = '../data/{}/{}_graph_labels.txt'.format(dataset, dataset)
    graph_labels = np.loadtxt(filename, dtype=np.float, delimiter=',')
    return graph_labels


def read_adjMatrix(file_index):
    array = open(
        'D:/PycharmProjects/motif/undirected-motif-entropy/data/graph' + str(file_index + 1) + '.csv').readlines()
    N = len(array)
    matrix = []
    for line in array:
        line = line.strip('\r\n').split(',')
        line = [int(float(x)) for x in line]
        matrix.append(line)
    matrix = np.array(matrix)
    return matrix, N


def store_matrix(matrix, filename):
    f1 = open(filename, "w")
    for line in matrix:
        for item in line:
            f1.write(str(item) + ',')
        f1.write('\n')


def read_data(filename):
    array = open(filename).readlines()
    matrix = []
    for line in array:
        line = line.strip('\r\n[],').split(',')
        line = [float(x) for x in line]
        matrix.append(line)
    matrix = np.array(matrix)
    return matrix


def read_data_txt(dataset):
    data = dict()
    dataset_name = str(dataset)
    dirpath = 'data/{}'.format(dataset_name)
    print('reading data...')
    for f in os.listdir(dirpath):
        if "README" in f or '.txt' not in f:
            continue
        fpath = complete_path(dirpath, f)
        suffix = f.replace(dataset_name, '')
        # print(suffix)
        if 'attributes' in suffix:
            data[suffix] = np.loadtxt(fpath, dtype=np.float, delimiter=',')
        else:
            data[suffix] = np.loadtxt(fpath, dtype=np.int, delimiter=',')

    return data

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


from scipy.io import loadmat

def read_ppi():
    print('reading ppi')
    res_graph=[]
    res_node_label=[]
    np.set_printoptions(suppress=True)
    m = loadmat("../data/PPIS/PPIs.mat")
    data = m.get('PPIs')
    for i in range(86):
        res_graph.append(np.array(data[0][i][0].todense()).astype(int))
        temp=[]
        for j in data[0][i][1][0][0][0]:
            temp.append(j[0])
        res_node_label.append(np.array(temp))
    return res_graph,res_node_label



import csv
def read_ppi_graph_label():
    res=[]
    with open('../data/PPIS/PPI_label.csv')as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            res.append(1) if row[0]=='1' else res.append(0)
    return np.array(res)

def read_ptc():
    graphs=[]
    for i in range(1,345):
        with open('../data/ptc/graph_'+str(i)+'.csv')as f:
            graph=csv.reader(f)
            temp_graph = []
            for line in graph:
                temp_line=[]
                for item in line:
                    s=0 if item=='0' else 1
                    temp_line.append(s)
                temp_graph.append(temp_line)
            graphs.append(np.array(temp_graph))

    labels=[]
    for i in range(1, 345):
        with open('../data/ptc/feature_' + str(i) + '.csv')as f:
            fea = csv.reader(f)
            node_labels=[]
            for line in fea:
                temp_label = 0
                for item in line:
                    if item=='1':
                        break
                    temp_label+=1
                node_labels.append(temp_label)
            #print(node_labels)
            labels.append(np.array(node_labels))
    return graphs,labels


def read_ptc_labels():
    labels = []
    with open('../data/ptc/ptc_label.csv')as f2:
        l = csv.reader(f2)
        for row in l:
            labels.append(1) if row[0] == '1' else labels.append(0)
    return np.array(labels)