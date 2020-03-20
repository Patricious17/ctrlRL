import control as ctrl
import control.matlab as mtl
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
#from matplotlib import rc; rc('text', usetex=True)
import pandas as pd


def newtonKleinman(A,B,Q,K, N):
    assert ((la.eig(A-B*K)[0]) < 0).min(), 'initial gain K is not stabilizing, Newton-Kleinman is not guaranteed to converge!'
    Kopt, Popt, Eopt = mtl.lqr(A, B, Q, 1)
    assert ((Eopt < 0).min() or ((la.eig(Popt)[0]) > 0).min()), 'There is no stabilizing solution to Riccati equation for provided (A,B,Q,R), your efforts are futile!'

    # P that corresponds to initial gain K
    P = mtl.lyap((A-np.dot(B, K)).transpose(), (Q + np.dot(K.transpose(), K))); assert ((la.eig(P)[0])>0).min(), 'P iterate is not positive definite!'
    errorP = []
    errorP.append(la.norm(P - Popt))

    for i in range(N):
        '''Newton-Kleinman is repeated Lyapunov equation that yields iterates P. Each P that constitutes Lyapunov function x'Px '''
        P = mtl.lyap((A - np.dot(B, np.dot(B.transpose(), P))).transpose(),
                     (Q + np.dot(np.dot(B.transpose(), P).transpose(), np.dot(B.transpose(), P)))); assert ((la.eig(P)[0])>0).min(), 'P iterate is not positive definite!'
        errorP.append(la.norm(P-Popt))

    plt.plot(errorP, 'o');    plt.grid();    plt.show()
    pass

def inverseReinfLearning1(A,B,Qtrue,N):
    Popt, Eopt, Kopt = mtl.care(A, B, Qtrue, 1);    Aopt = A - np.dot(B, Kopt)
    Q = Qtrue*1.1;    P, E, K = mtl.care(A, B, Q, 1)

    errorP = [];    errorP.append(la.norm(P - Popt))

    for i in range(5):
        print(i)
        '''     2*             (P B          B'             P)  -       (A'             P) -       (P A)    '''
        Qlyap = 2*np.dot(np.dot(P,B), np.dot(B.transpose(), P)) - np.dot(A.transpose(), P) - np.dot(P,A)
        ''' making sure it is symmetric '''
        Qlyap = 0.5*(Qlyap + Qlyap.transpose()); assert ((la.eig(Qlyap)[0])>0).min(), 'Qlyap iterate is not positive definite!'
        '''  '''
        P = mtl.lyap(Aopt.transpose(), Qlyap); assert ((la.eig(P)[0])>0).min(), 'P iterate is not positive definite!'
        errorP.append(la.norm(P - Popt))

    plt.plot(errorP, 'o');    plt.grid();    plt.show()
    pass


def inverseReinfLearning(A,B,Qtrue,N, method = 'Lyap', debugMode = False):
    Popt, Eopt, Kopt = mtl.care(A, B, Qtrue, 1);    Aopt = A - B * Kopt
    Q = Qtrue*0.01;    P, E, K = mtl.care(A, B, Q, 1)
    errorP = [];    errorP.append(la.norm(P - Popt))
    errorQ = [];    errorQ.append(la.norm(Q - Qtrue))
    errorK = [];    errorK.append(la.norm(K - Kopt))

    for i in range(N):
        '''  -(      (P Aopt) +       (Aopt'             P) +              (P B B'            P) '''
        Qi = -(np.dot(P,Aopt) + np.dot(Aopt.transpose(), P) + la.multi_dot([P,B,B.transpose(),P]))
        Qi = 0.5 * (Qi + Qi.transpose()); assert (not debugMode) or ((la.eig(Qi)[0]) > 0).min(), 'Qi iterate is not positive definite!'

        if method == 'Lyap':
            Qlyap = Qi - la.multi_dot([P,B,B.transpose(),P])
            Qlyap = 0.5 * (Qlyap + Qlyap.transpose()); assert (not debugMode) or ((la.eig(Qlyap)[0])>0).min(), 'Qlyap iterate is not positive definite!'
            P = mtl.lyap(A.transpose(), Qlyap); assert (not debugMode) or ((la.eig(P)[0]) > 0).min(), 'P iterate is not positive definite!'
            pass
        elif method == 'Ricc':
            P,E,K = mtl.care(A, B, Qi); assert (not debugMode) or ((la.eig(P)[0]) > 0).min(), 'P iterate is not positive definite!'
            pass

        errorP.append(la.norm(P - Popt)); errorK.append(la.norm(K - Kopt)); errorQ.append(la.norm(Qi - Qtrue))


    plt.plot(errorP, 'o', label=r'$||P-\mathcal{P}||_F$'); plt.plot(errorK, 'o', label=r'$||K-\mathcal{K}||_F$'); plt.plot(errorQ, 'o', label=r'$||Q-\mathcal{Q}||_F$')
    plt.legend(ncol=3, loc='upper right');    plt.show()
    pass



if __name__== "__main__":
    print('Main executed')
    A = np.array([[-1,  2],[2.2,  1.7]]);     A1 = np.array([[-1,  2],[2.2,  1.7]])
    #eigs = la.eig(A);    eval = eigs[0];    evec = eigs[1];    print('eigen values of A:\n', eval)
    B = np.array([[2], [1.6]]);     B1 = np.array([[2], [1.6]])
    Q = np.array([[6, 0], [0, 6]])
    q=np.random.rand(2, 2);    Qrand = np.dot(q.transpose(), q)

    #K0,X,E = mtl.lqr(A,B, Qrand,1)
    K0, X, E = mtl.lqr(A, B, Q, 1)

    #eigs = la.eig(A-np.dot(B,K0));    eval = eigs[0];    evec = eigs[1];    print('eigen values of Acl:\n', eval)

    #newtonKleinman(A, B, Q, K0, 10)
    #inverseReinfLearning1(A, B, Q, 10)
    inverseReinfLearning(A, B, Q, 10, 'Ricc', debugMode=True)
    #inverseReinfLearning_Lyap1(A,B,Q,50)

    print('Main finished')



#
#
# def inverseReinfLearning_Lyap1(A, B, Qtrue, N):
#     errorP = []
#     Kopt, Popt, Eopt = mtl.lqr(A, B, Qtrue, 1)
#     assert ((Eopt < 0).min() or ((la.eig(Popt)[
#         0]) > 0).min()), 'There is no stabilizing solution to Riccati equation for provided (A,B,Q,R), your efforts are futile!'
#     '''Optimal agent'''
#     Aopt = A-B*Kopt
#
#     '''Initial guess for Q'''
#     Q = Qtrue*3
#     K, P, E = mtl.lqr(A, B, Q, 1); Acl = A-B*K
#
#     for i in range(N):
#         P = mtl.lyap((A - B * (B.transpose() * P)).transpose(),-(Aopt.transpose()*P+P*Aopt)); assert ((la.eig(P)[0])>0).min(), 'P iterate is not positive definite!'
#         errorP.append(la.norm(P-Popt))
#
#     plt.plot(errorP, 'o')
#     plt.grid()
#     plt.show()
#
#     pass
#
# def inverseReinfLearning_Lyap2(A, B, Qtrue, N):
#     errorP = []
#     Kopt, Popt, Eopt = mtl.lqr(A, B, Qtrue, 1)
#     assert ((Eopt < 0).min() or ((la.eig(Popt)[
#         0]) > 0).min()), 'There is no stabilizing solution to Riccati equation for provided (A,B,Q,R), your efforts are futile!'
#     '''Optimal agent'''
#     Aopt = A-B*Kopt
#
#     '''Initial guess for Q'''
#     Q = 1.01*Qtrue
#     P, E, K = mtl.care(A, B, Q, 1); Acl = A-B*K
#
#     for i in range(N):
#         Ptarget,L,G= mtl.care(A=A-B*B.transpose()*P - Aopt, B=B, Q=(P*B*B.transpose()*P+(P*B*B.transpose()*P).transpose())/2)
#         P = P + (Ptarget-P)/la.norm(Ptarget-P)
#         errorP.append(la.norm(P - Popt))
#         print(P-Popt)
#
#     plt.plot(errorP, 'o')
#     plt.grid()
#     plt.show()
#
#     pass
#
