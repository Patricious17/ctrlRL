import control as ctrl
import control.matlab as mtl
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

#implementation based on Overleaf document on Inverse reinforcement Learning
A = np.array([[-1,  2],[2.2,  1.7]]);     A1 = np.array([[-1,  2],[2.2,  1.7]])
#eigs = la.eig(A);    eval = eigs[0];    evec = eigs[1];    print('eigen values of A:\n', eval)
B = np.array([[2], [1.6]]);     B1 = np.array([[2], [1.6]])
Q = np.array([[6, 0], [0, 6]])

if __name__== "__main__":

    pass