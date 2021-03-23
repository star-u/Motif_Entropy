import numpy as np
import fractions as fc
from numpy.linalg import *

a = np.array([[1, 0, 1], [-1, 1, 1], [0, -1, 1]])
b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
m = np.linalg.inv(a)

A = np.dot(np.dot(a, b), m)
# print(A)
B = np.array([[fc.Fraction(1, 3), fc.Fraction(-2, 3), fc.Fraction(-2, 3)],
              [fc.Fraction(-2, 3), fc.Fraction(1, 3), fc.Fraction(-2, 3)]
                 , [fc.Fraction(-2, 3), fc.Fraction(-2, 3), fc.Fraction(1, 3)]])
A8 = np.linalg.matrix_power(B, 8)
# print(A8)

C = np.array([[1, 2, 0, 0], [2, 6, 0, 0], [0, 0, 3, 5], [0, 0, 1, 2]])
C1 = np.linalg.inv(C)
# print(det(C), '\n', det(C1))

AAA = np.array([[1, 1, 2, 2, 3], [2, 2, 3, 1, 4], [1, 0, 1, 1, 5], [2, 3, 5, 5, 4]])
PPP = np.array([[-2, 1, 1, 0], [-1, 1, -1, 0], [2, -1, 0, 0], [-3, 0, 1, 1]])
PPPP = np.array([[1,1,0,-1],[-4,1,0,1],[2,-1,0,0],[-3,0,1,1]])
print(np.dot(PPP, AAA))
print(np.dot(PPPP, AAA))

AA = np.array([[1, 0, 0, 1, 1], [0, 1, 0, 2, 2], [0, 0, 1, fc.Fraction(1,2), 0]])
PP = np.array([[1, -1, 2], [1, 1, -2], [3, -1, 8], [1, 3, -6]])
# print(np.dot(PP, AA))

