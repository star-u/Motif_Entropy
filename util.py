import numpy as np
import os


def complete_path(folder, fname):
    return os.path.join(folder, fname)



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
        elif suffix != "_label_pro.txt":
            data[suffix] = np.loadtxt(fpath, dtype=np.int, delimiter=',')

    return data
