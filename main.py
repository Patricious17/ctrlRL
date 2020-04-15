import control as ctrl
import control.matlab as mtl
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib import rc; rc('xtick', labelsize=20);  rc('ytick', labelsize=20) #rc('text', usetex=True)
import pandas as pd

plt.rcParams.update({'font.size': 22})


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


def inverseReinfLearning(A, B, Qtrue, N, p=None, debugMode = False):
    Popt, Eopt, Kopt = mtl.care(A, B, Qtrue, 1);    Aopt = A - B * Kopt
    Q = Qtrue*0.9 +0.1*(1-0.5)*np.random.rand(Qtrue.shape[0],Qtrue.shape[1]); Q = 0.5*(Q+Q.transpose())
    print(Q)
    mtl.lyap(A+Aopt,Q)
    P, E, K = mtl.care(A, B, Q, 1)
    errorP = [];    errorP.append(la.norm(P - Popt));    errorQ = [];    errorQ.append(la.norm(Q - Qtrue));    errorK = [];    errorK.append(la.norm(K - Kopt))
    eigsP = ([],[]); eigsP[0].append(la.norm(P)); eigsP[1].append(la.norm(P,2))
    eigsQ = ([],[]); eigsQ[0].append(la.norm(Q)); eigsQ[1].append(la.norm(Q,2))
    Qi = None

    for i in range(N):
        print(Qi)
        '''         -(      (P Aopt)  +       (Aopt'             P) +             ( P  B  B'             P )) '''
        target_Qi = -(np.dot(P, Aopt) + np.dot(Aopt.transpose(), P) + la.multi_dot([P, B, B.transpose(), P]))
        Qi = ((1-p['alpha']**2)**0.5)*Qi + ((p['alpha'] ** 2)**0.5) * target_Qi if Qi is not None else target_Qi
        Qi = 0.5 * (Qi + Qi.transpose()) if not p['isQDiagonal'] else np.multiply(0.5 * (Qi + Qi.transpose()), np.eye(Qi.shape[0]))
        print("Q",i," eigenvalues are: ",la.eig(Qi)[0])
        if not ((la.eig(Qi)[0]) >= 0).min():
            pass
        assert (not debugMode) or ((la.eig(Qi)[0]) >= 0).min(), 'Qi iterate is not positive definite!'

        if p['updateP'] == 'Lyap':
            Qlyap = Qi - la.multi_dot([P,B,B.transpose(),P])
            Qlyap = 0.5 * (Qlyap + Qlyap.transpose())
            assert (not debugMode) or ((la.eig(Qlyap)[0])>0).min(), 'Qlyap iterate is not positive definite!'
            P = mtl.lyap(A.transpose(), Qlyap); assert (not debugMode) or ((la.eig(P)[0]) > 0).min(), 'P iterate is not positive definite!'
            pass
        elif p['updateP'] == 'Ricc':
            P,E,K = mtl.care(A, B, Qi); assert (not debugMode) or ((la.eig(P)[0]) > 0).min(), 'P iterate is not positive definite!'
            pass

        errorP.append(la.norm(P - Popt,'fro')); errorK.append(la.norm(K - Kopt,'fro')); errorQ.append(la.norm(Qi - Qtrue,'fro'))
        eigsP[0].append(la.norm(P)); eigsP[1].append(la.norm(P, 2))
        eigsQ[0].append(la.norm(Qi)); eigsQ[1].append(la.norm(Qi, 2))


    fig1, (ax1_1) = plt.subplots(nrows=1, ncols=1)
    ax1_1.plot(errorP, 'o', label=r'$||P-\mathcal{P}||_F$', markersize=12)
    ax1_1.plot(errorK, 'o', label=r'$||K-\mathcal{K}||_F$', markersize=12)
    ax1_1.plot(errorQ, 'o', label=r'$||Q-\mathcal{Q}||_F$', markersize=12)
    ax1_1.legend(ncol=2, loc='upper right', prop={'size': 30})

    fig2, (ax2_1, ax2_2) = plt.subplots(nrows=2, ncols=1)
    ax2_1.plot(eigsP[0], 'o', label=r'$||P||_F$', markersize=12)
    ax2_1.plot(eigsP[1], 'o', label=r'$\lambda_{max}(P)$', markersize=12)
    ax2_1.legend(ncol=1,loc='upper right', prop={'size': 30})

    ax2_2.plot(eigsQ[0], 'o', label=r'$||Q||_F$', markersize=12)
    ax2_2.plot(eigsQ[1], 'o', label=r'$\lambda_{max}(Q)$', markersize=12)
    ax2_2.legend(ncol=1, loc='upper right', prop={'size': 30})

    plt.show()

    pass


if __name__== "__main__":
    print('Main executed')
    case = '3D'
    '''2D case '''
    if case == '2D':
        A = np.array([[-1,  2],[2.2,  1.7]]);     A1 = np.array([[-1,  2],[2.2,  1.7]])
        # eigs = la.eig(A);    eval = eigs[0];    evec = eigs[1];    print('eigen values of A:\n', eval)
        B = np.array([[2], [1.7]]);    B1 = np.array([[2], [1.7]])
        Q = np.array([[5, 0], [0, 6]])
        q = np.random.rand(2, 2);    Qrand = np.dot(q.transpose(), q)
    '''3D case '''
    if case == '3D':
        A = np.array([[-1, 2, -1], [2.2, 1.7, -1], [0.5, 0.7, -1]]);        A1 = np.array([[-1, 2, -1], [2.2, 1.7, -1], [0.5, 0.7, -1]]);
        eigs = la.eig(A);        eval = eigs[0];        evec = eigs[1];        print('eigen values of A:\n', eval)
        B = np.array([[2], [1.6], [3]]);        B1 = np.array([[2], [1.6], [0.9]])
        Q = np.array([[5, 0, 0], [0, 8, 0], [0, 0, 1]])
        q = np.random.rand(3, 3);    Qrand = np.dot(q.transpose(), q)



    #K0,X,E = mtl.lqr(A,B, Qrand,1)
    K0, X, E = mtl.lqr(A, B, Q, 1)

    #eigs = la.eig(A-np.dot(B,K0));    eval = eigs[0];    evec = eigs[1];    print('eigen values of Acl:\n', eval)

    #newtonKleinman(A, B, Q, K0, 10)
    #inverseReinfLearning1(A, B, Q, 10)
    algorithmProperties = {'updateP' : 'Ricc', 'alpha': 1.0, 'isQDiagonal' : False}
    inverseReinfLearning(A, B, Q, 30, p =algorithmProperties, debugMode=True)
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
