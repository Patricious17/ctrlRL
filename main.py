import control as ctrl
import control.matlab as mtl
import numpy as np
import numpy.linalg as la

if __name__== "__main__":
    print('Main executed')
    A = np.array([[-1,  2],[2.2,  1.7]]);     A1 = np.array([[-1,  2],[2.2,  1.7]])
    eigs = la.eig(A);    eval = eigs[0];    evec = eigs[1];    print('eigen values of A:\n', eval)
    B = np.array([[2], [1.6]]);     B1 = np.array([[2], [1.6]])
    Q = np.array([[6, 0], [0, 6]])
    K,X,E = mtl.lqr(A,B, .1*Q,1)
    Acl = A-B*K
    Acl = A

    eigs = la.eig(Acl);    eval = eigs[0];    evec = eigs[1];    print('eigen values of Acl:\n', eval)

    P = mtl.lyap(Acl.transpose(),Q);    eigs = la.eig(P);    eval = eigs[0];    evec = eigs[1];    print('eigen values of P:\n', eval)


    print('Main finished')
