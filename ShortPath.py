import numpy as np


data = np.array([[1, 3, 5, 9], [2, 1, 3, 4], [5, 2, 6, 7], [6, 8, 4, 3]])
print(data)

def short_path(data):
    n = data.shape[0]
    dist = np.zeros((n,n))
    sum = 0

    for i in range(n):
        sum += data[0][i]
        dist[0][i] = sum

    sum = 0
    for j in range(n):
        sum += data[j][0]
        dist[j][0] = sum

    for i in range(1, n):
        for j in range(1,n):
            dist[i][j] = min(dist[i-1, j], dist[i, j-1])+data[i][j]

    print(dist)
    return dist[n-1][n-1]

print(short_path(data))

# def add(x: str, y: int)->str:
#     return x+str(y)
#
# print(add.__annotations__)