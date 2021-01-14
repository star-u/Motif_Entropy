import torch

from MotifCount import *
from entropy import *

A = np.array([[0, 1, 1, 0, 0],
              [1, 0, 0, 1, 1],
              [1, 0, 0, 0, 1],
              [0, 1, 0, 0, 1],
              [0, 1, 1, 1, 0]])

num_motif, num_node = count_Motifs(A)
# print(num_node)

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
print(count_every_vector)

# 求不同标签的节点概率
label_a = [1, 3, 5]
label_b = [2, 4]
count_labela = 0
count_labelb = 0
for i, j in enumerate(label_a):
    count_labela += count_every_vector[j-1]
print(count_labela)

for i, j in enumerate(label_b):
    count_labelb += count_every_vector[j-1]
print(count_labelb)

label_a_p = []
label_b_p = []
for i, j in enumerate(label_a):
    label_a_p.append(count_every_vector[j-1]/count_labela)
print(label_a_p)
for i, j in enumerate(label_b):
    label_b_p.append(count_every_vector[j-1]/count_labelb)
print(label_b_p)