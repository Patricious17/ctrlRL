import control as ctrl
import control.matlab as mtl
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def newtonKleinman(A,B,Q,K, N):
    assert ((la.eig(A-B*K)[0]) < 0).min(), 'initial gain K is not stabilizing, Newton-Kleinman is not guaranteed to converge!'
    errorP = []
    Kopt, Popt, Eopt = mtl.lqr(A, B, Q, 1)
    assert ((Eopt < 0).min() or ((la.eig(Popt)[0]) > 0).min()), 'There is no stabilizing solution to Riccati equation for provided (A,B,Q,R), your efforts are futile!'

    # P that corresponds to initial gain K
    P = mtl.lyap((A-B*K).transpose(), (Q + K.transpose() * K)); assert ((la.eig(P)[0])>0).min(), 'P iterate is not positive definite!'

    for i in range(N):
        '''Newton-Kleinman is repeated Lyapunov equation that yields iterates P. Each P that constitutes Lyapunov function x'Px '''
        P = mtl.lyap((A - B * (B.transpose() * P)).transpose(),
                     (Q + (B.transpose() * P).transpose() * (B.transpose() * P))); assert ((la.eig(P)[0])>0).min(), 'P iterate is not positive definite!'
        errorP.append(la.norm(P-Popt))

    plt.plot(errorP, 'o')
    plt.grid()
    plt.show()
    pass

def inverseReinfLearning_Lyap1(A, B, Qtrue, N):
    errorP = []
    Kopt, Popt, Eopt = mtl.lqr(A, B, Qtrue, 1)
    assert ((Eopt < 0).min() or ((la.eig(Popt)[
        0]) > 0).min()), 'There is no stabilizing solution to Riccati equation for provided (A,B,Q,R), your efforts are futile!'
    '''Optimal agent'''
    Aopt = A-B*Kopt

    '''Initial guess for Q'''
    Q = Qtrue*3
    K, P, E = mtl.lqr(A, B, Q, 1); Acl = A-B*K

    for i in range(N):
        P = mtl.lyap((A - B * (B.transpose() * P)).transpose(),-(Aopt.transpose()*P+P*Aopt)); assert ((la.eig(P)[0])>0).min(), 'P iterate is not positive definite!'
        errorP.append(la.norm(P-Popt))

    plt.plot(errorP, 'o')
    plt.grid()
    plt.show()

    pass

def inverseReinfLearning_Lyap2(A, B, Qtrue, N):
    errorP = []
    Kopt, Popt, Eopt = mtl.lqr(A, B, Qtrue, 1)
    assert ((Eopt < 0).min() or ((la.eig(Popt)[
        0]) > 0).min()), 'There is no stabilizing solution to Riccati equation for provided (A,B,Q,R), your efforts are futile!'
    '''Optimal agent'''
    Aopt = A-B*Kopt

    '''Initial guess for Q'''
    Q = 1.01*Qtrue
    P, E, K = mtl.care(A, B, Q, 1); Acl = A-B*K

    for i in range(N):
        Ptarget,L,G= mtl.care(A=A-B*B.transpose()*P - Aopt, B=B, Q=(P*B*B.transpose()*P+(P*B*B.transpose()*P).transpose())/2)
        P = P + (Ptarget-P)/la.norm(Ptarget-P)
        errorP.append(la.norm(P - Popt))
        print(P-Popt)

    plt.plot(errorP, 'o')
    plt.grid()
    plt.show()

    pass

if __name__== "__main__":
    print('Main executed')
    A = np.array([[-1,  2],[2.2,  1.7]]);     A1 = np.array([[-1,  2],[2.2,  1.7]])
    eigs = la.eig(A);    eval = eigs[0];    evec = eigs[1];    print('eigen values of A:\n', eval)
    B = np.array([[2], [1.6]]);     B1 = np.array([[2], [1.6]])
    Q = np.array([[6, 0], [0, 6]])
    K0,X,E = mtl.lqr(A,B, 19*Q,1)

    eigs = la.eig(A-B*K0);    eval = eigs[0];    evec = eigs[1];    print('eigen values of Acl:\n', eval)
    #P = mtl.lyap((A-B*K0).transpose(),Q);    eigs = la.eig(P);    eval = eigs[0];    evec = eigs[1];    print('eigen values of P:\n', eval)

    #newtonKleinman(A, B, Q, K0, 10)
    inverseReinfLearning_Lyap1(A,B,Q,50)

    print('Main finished')





