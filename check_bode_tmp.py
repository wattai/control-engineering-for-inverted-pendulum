#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 20:54:06 2017

@author: wattai
"""




import numpy as np
from scipy import signal
import scipy as sp
import matplotlib.pyplot as plt


def Ric(H):
    # Riccati 方程式の '擬似'安定化解 を Hamilton 行列の形で解くソルバ
    H_eval, H_evec = sp.linalg.eig(H)
    return (H_evec[int(len(H)/2):, (H_eval.real < 0)] @ sp.linalg.pinv(H_evec[:int(len(H)/2), (H_eval.real < 0)])).real

#def hinfsyn(A, B, C, D):
    
    
    
            

if __name__ == "__main__":
    
    # パラメータ設定
    m = 1
    c = 1
    k = 400
    
    num = [k]
    den = [m, c, k]
    s1 = signal.lti(num, den)
    w, mag, phase = signal.bode(s1, np.arange(1, 20000, 1))
    """
    # プロット
    plt.figure(1)
    plt.subplot(211)
    plt.loglog(w, 10**(mag/20))
    plt.ylabel("Amplitude")
    plt.axis("tight")
    plt.subplot(212)
    plt.semilogx(w, phase)
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("Phase[deg]")
    plt.axis("tight")
    plt.ylim(-180, 180)
    #plt.savefig('../files/150612TF02.svg')
    plt.show()
    """
    
    """
    # パラメータ設定
    m = 1
    c = 1
    k = 400
    
    A = np.array([[0, 1], [-k/m, -c/m]])
    B = np.array([[0], [k/m]])
    C = np.array([1, 0])
    D = np.array([0])
    
    s1 = signal.lti(A, B, C, D)
    w, mag, phase = signal.bode(s1, np.arange(1, 500, 1)) 
    
    # プロット
    plt.figure(1)
    plt.subplot(211)
    plt.loglog(w, 10**(mag/20))
    plt.ylabel("Amplitude")
    plt.axis("tight")
    plt.subplot(212)
    plt.semilogx(w, phase)
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("Phase[deg]")
    plt.axis("tight")
    plt.ylim(-180, 180)
    #plt.savefig('../files/150613ABCD01.svg')
    plt.show()
    """
    
    # 物理パラメータの定義 -------------
    l = 0.129 #+ 0.01 * np.random.randn() # 車軸と車体重心間の距離 [m]
    r = 0.042 #+ 0.01 * np.random.randn() # 車輪半径 [m]
    M = 0.318 #+ 0.1 * np.random.randn() # 車体質量 [kg]
    m = 0.064 #+ 0.01 * np.random.randn() # 車輪質量 [kg]
    J_b = 1.46e-3 #+ 0.001 * np.random.randn() # 車体慣性モーメント [kg * m^2]
    J_w = 1.13e-4 #+ 0.0001 * np.random.randn() # 車輪慣性モーメント [kg * m^2]
    g = 9.8 # 重力加速度 [m / sec^2]
    
    c_1 = m*(r**2) + M*(r+l)**2 + J_w + J_b
    c_2 = J_w + (M+m)*(r**2) + M*l*r
    # ----------------------------
    
    # プラント行列の定義 --------------
    Ap = np.array([[0., 1., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 1.],
                  [0., 0., M*g*l/c_1, 0.]
                 ])
    Bp = np.array([[0.],
                  [1.],
                  [0.],
                  [-c_2/c_1]
                 ])
    #Cp = np.eye(2, 4)
    Cp = np.array([[0., 0., 0., 1.]])
    Dp = np.array([[0]])
    # -----------------------------

    # H inf 制御器 -----------------
    """
    A_k = np.array([[2.590e-10, 1., 3.235e-12, -1.370e-9], # gopt +0 by hinf 4*4
                    [0.6893424, 0.7213022, 217.4438, 35.840863],
                    [3.440e-32, 3.410e-36, 1.325e-22, -0.3660254],
                    [-0.1575026,-0.1648048, -13.083737, -15.777322]])
    B_k = np.array([[1.371e-9],
                    [0.2171464],
                    [1.3660254],
                    [7.5387036]])
    C_k = np.array([[0.6893424, 0.7213022, 217.4438, 35.840863]])
    D_k = np.array([[0.2171464]])
    """
    A_k = np.array([[-7.1982553, 2.2732067, -15.157808, 226.13836], # gopt +0 by ccontrq 4*4
                    [45.580684, -13.179226, 43.779669, -653.14599],
                    [-0.7428765, 0.1913499, -0.6661122, 8.6458358],
                    [0.5595782, -0.1491249, 0.0030360, -6.0420083]])
    B_k = np.array([[0.6821519],
                    [-27.640598],
                    [0.4506237],
                    [-0.3394171]])
    C_k = np.array([[-0.0192689, -31.174462, 193.64163, -2888.9271]])
    D_k = np.array([[0.2171464]])  
    
    # -----------------------------
    
    
    # LTIシステムインスタンスの生成    
    sys_plant = signal.lti(Ap, Bp, Cp, Dp)
    sys_controller = signal.lti(A_k, B_k, C_k, D_k)
    w, mag1, phase1 = signal.bode(sys_plant, np.arange(0.01, 10000, 1)) 
    w, mag2, phase2 = signal.bode(sys_controller, np.arange(0.01, 10000, 1))

    #numws = [1, 10] # バネマスダンパ系デフォ
    #denws = [1, 1000]
    #numws = [50000, 0.001] # バネマスダンパ系デフォ
    #denws = [100000, 500000]
    fcs = 100000
    wcs = 2 * np.pi * fcs
    numws = [1, 0] # バネマスダンパ系デフォ
    denws = [1/0.001, wcs]
    #numws = [1*0.01, 10*0.1, 1*0.1]
    #denws = [0.001, 1000, 10]
    Ws = signal.lti(numws, denws)
    #numwt = [1/100, 0.1] # バネマスダンパ系デフォ
    #denwt = [1/0.01, 1]
    #numwt = [1/100, 1/1] # バネマスダンパ系デフォ
    #denwt = [10/1, 0]
    #numwt = [1/10 * 0.001, 100000*0.001, 1*0.001]
    #denwt = [1/0.1, 100, 1000 ]
    fct = 0.1
    wct = 2 * np.pi * fct
    numwt = [0, wct] # バネマスダンパ系デフォ
    denwt = [1/0.1, wct]
    Wt = signal.lti(numwt, denwt)
    
    #s = Ws
    #Ws = Wt
    #Wt = s
    
    w, mag3, phase3 = signal.bode(Ws, np.arange(0.01, 10000, 1))
    w, mag4, phase4 = signal.bode(Wt, np.arange(0.01, 10000, 1))
    
    
    # プロット
    plt.figure(1)
    plt.subplot(211)
    #plt.loglog(w, 10**(mag1/20), label="plant")
    #plt.loglog(w, 10**(mag2/20), label="controller")

    plt.semilogx(w, mag1, label="Plant")
    plt.semilogx(w, mag2, label="Controller")
    plt.semilogx(w, mag3, label="Ws")
    plt.semilogx(w, mag4, label="Wt")
    plt.ylabel("Amplitude [dB]")
    plt.axis("tight")
    #loc='lower right'で、右下に凡例を表示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    plt.subplot(212)
    plt.semilogx(w, phase1, label="Plant")
    plt.semilogx(w, phase2, label="Controller")
    plt.semilogx(w, phase3, label="Ws")
    plt.semilogx(w, phase4, label="Wt")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [deg]")
    plt.axis("tight")
    #loc='lower right'で、右下に凡例を表示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.ylim(-180, 180)
    #plt.savefig('../files/150613ABCD01.svg')
    plt.show()
    
    
    # 一般化プラントの生成 ---------------------------------------------------------
    gamma = 1.004559375
    gamma = 0.85000
    gamma = 0.7
    gamma = 0.05
    
    a_x = np.concatenate([Ap, np.zeros([Ap.shape[0], Ws.A.shape[1] + Wt.A.shape[1]])], axis=1)
    a_ws = np.concatenate([Ws.B @ Cp, Ws.A, np.zeros([Ws.A.shape[0], Wt.A.shape[1]])], axis=1)
    a_wt = np.concatenate([Wt.B @ Cp, np.zeros([Wt.A.shape[0], Ws.A.shape[1]]), Wt.A], axis=1)
    A = np.concatenate([a_x, a_ws, a_wt], axis=0)
    
    B1 = np.concatenate([np.zeros([Ap.shape[0], Ws.B.shape[1]]), -Ws.B, np.zeros([Wt.A.shape[0], Ws.B.shape[1]])], axis=0)
    B2 = np.concatenate([Bp, np.zeros([Ws.B.shape[0], Bp.shape[1]]), np.zeros([Wt.B.shape[0], Bp.shape[1]])], axis=0)
    
    c_z1 = np.concatenate([Ws.D @ Cp, Ws.C, np.zeros([Ws.C.shape[0], Wt.C.shape[1]])], axis=1)
    c_z2 = np.concatenate([Wt.D @ Cp, np.zeros([Wt.C.shape[0], Ws.C.shape[1]]), Wt.C], axis=1)
    C1 = np.concatenate([c_z1, c_z2], axis=0)
    C2 = np.concatenate([-Cp, np.zeros([Cp.shape[0], Ws.A.shape[1]]), np.zeros([Cp.shape[0], Wt.A.shape[1]])], axis=1)
    
    D11 = np.concatenate([-Ws.D, np.zeros([Wt.D.shape[0], Ws.D.shape[1]])], axis=0)
    D12 = np.concatenate([np.zeros([Ws.C.shape[0], Bp.shape[1]]), np.zeros([Wt.C.shape[0], Bp.shape[1]])], axis=0)
    D21 = np.concatenate([np.eye(Cp.shape[0])])
    D22 = np.zeros([D21.shape[0], D12.shape[1]])
    
    """
    # 一般化プラントの例 ----------------
    gamma = 0.75 #2**(-1/2) + 0.1 # gamma > 2**(-1/2) が理論的限界
    gam = 1
    A = np.array([[1]])
    B1 = np.array([[1/gam]])
    B2 = np.array([[2]])
    C1 = np.array([[1], [0]])
    C2 = np.array([[2]])
    D11 = np.array([[0], [0]])
    D12 = np.array([[0], [1]])
    D21 = np.array([[1/gam]])
    D22 = np.array([[0]])
    # -------------------------------
    """
    
    
    epsilon1 = 0.0001
    U = np.eye(C1.shape[1])
    C1 = np.concatenate([C1, epsilon1 * U], axis=0)
    D11 = np.concatenate([D11, np.zeros([U.shape[0], B1.shape[1]])], axis=0)
    D12 = np.concatenate([D12, np.zeros([U.shape[0], B2.shape[1]])], axis=0)
    
    epsilon2 = 0.0001 # D12 列フルランク を満たすための処置
    U = np.eye(B2.shape[1])
    C1 = np.concatenate([C1, np.zeros([U.shape[0], C1.shape[1]])], axis=0)
    D11 = np.concatenate([D11, np.zeros([U.shape[0], B1.shape[1]])], axis=0)
    D12 = np.concatenate([D12, epsilon2 * U], axis=0)
    
    delta1 = 0.0001
    U = np.eye(B1.shape[0])
    B1 = np.concatenate([B1, delta1 * U], axis=1)
    D11 = np.concatenate([D11, np.zeros([D11.shape[0], U.shape[1]])], axis=1)
    D21 = np.concatenate([D21, np.zeros([D21.shape[0], U.shape[1]])], axis=1)
    
    delta2 = 0.0001
    U = np.eye(C2.shape[0])
    D21 = np.concatenate([D21, delta2 * U], axis=1)
    B1 = np.concatenate([B1, np.zeros([B1.shape[0], U.shape[1]])], axis=1)
    D11 = np.concatenate([D11, np.zeros([D11.shape[0], U.shape[1]])], axis=1)
    

    
    
    # 正規化 ----------------
    #B1 = B1 / gamma
    #D21 = D21 / gamma
    #C1 = C1 / gamma
    #D12 = D12 / gamma
    
    E12 = D12.T @ D12
    D12 = D12 @ E12**(-1/2)
    B2 = B2 @ E12**(-1/2)
    
    E21 = D21 @ D21.T
    D21 = E21**(-1/2) @ D21
    C2 = E21**(-1/2) @ C2
    # ----------------------
    
    
    B = np.bmat([[B1, B2]])
    C = np.bmat([[C1], [C2]])
    D = np.bmat([[D11, D12], [D21, D22]])
    
    P = np.bmat([[A, B], [C, D]])
    
    #Up, _I, Rp = sp.linalg.svd(D12)
    #Rp_tilda, __I, Up_tilda = sp.linalg.svd(D21)
    
    #print(D)
    #G = sp.linalg.block_diag(Up.T, sp.linalg.inv(Rp_tilda)) @ P @ sp.linalg.block_diag(Up_tilda.T, sp.linalg.inv(Rp))
    #print(G)
    #D = G.copy()
    #D11 = D[:D11.shape[0], :D11.shape[1]]
    #D12 = D[:D11.shape[0], D11.shape[1]:]
    #D21 = D[D11.shape[0]:, :D11.shape[1]]
    #D22 = D[D11.shape[0]:, D11.shape[1]:]
    
    #P[A.shape[0]:, A.shape[1]:] = G.copy()
    # ------------------------------------------------------------------------
    
    __1 = np.concatenate([A - np.eye(A.shape[0]) * 1j, B1], axis=1)
    __2 = np.concatenate([C2, D21], axis=1)
    __ = np.concatenate([__1, __2], axis=0)
    if np.linalg.matrix_rank(__) == __.shape[0]:
        print("row is full rank !!\n")
    else:
        print("row is NOT full rank...\n")
    
    _1 = np.concatenate([A - np.eye(A.shape[0]) * 1j, B2], axis=1)
    _2 = np.concatenate([C1, D12], axis=1)
    _ = np.concatenate([_1, _2], axis=0)
    if np.linalg.matrix_rank(_) == _.shape[1]:
        print("colmun is full rank !!\n")
    else:
        print("column is NOT full rank...\n")
        

    
    """
    D1_ = np.concatenate([D11, D12], axis=1)
    D_1 = np.concatenate([D11, D21], axis=0)
    gam_Inw = np.eye(B1.shape[1]) * (gamma**2)
    gam_Inz = np.eye(C1.shape[0]) * (gamma**2)
    
    gam_Inw_ = np.zeros((D1_.T @ D1_).shape)
    gam_Inw_[0:gam_Inw.shape[0], 0:gam_Inw.shape[1]] = gam_Inw
    R = (D1_.T @ D1_) - gam_Inw_

    gam_Inz_ = np.zeros((D_1 @ D_1.T).shape)
    gam_Inz_[0:gam_Inz.shape[0], 0:gam_Inz.shape[1]] = gam_Inz
    R_tilda = (D_1 @ D_1.T) - gam_Inz_
    
    
    H = np.bmat([[A, np.zeros(A.shape)], [-C1.T @ C1, -A.T]]) - np.bmat([[B],[-C1.T @ D1_]]) @ sp.linalg.inv(R) @ np.bmat([D1_.T @ C1, B.T])
    J = np.bmat([[A.T, np.zeros(A.shape)], [-B1 @ B1.T, -A]]) - np.bmat([[C.T],[-B1 @ D_1.T]]) @ sp.linalg.inv(R_tilda) @ np.bmat([D_1 @ B1.T, C])
    
    X = Ric(H)
    Y = Ric(J)
    
    F_I = - sp.linalg.inv(R) @ (D1_.T @ C1 + B.T @ X)
    """
    
    
    # gamma の正規化が前提 ----
    #Hx = np.bmat([[ A - B2 @ D12.T @ C1, B1 @ B1.T  - B2 @ B2.T ], [ -C1.T @ C1 + C1.T @ D12 @ D12.T @ C1, -(A - B2 @ D12.T @ C1).T ]])
    #Hy = np.bmat([[ (A - B1 @ D21.T @ C2).T, C1.T @ C1  - C2.T @ C2 ], [ -B1 @ B1.T + B1 @ D21.T @ D21 @ B1.T, -(A - B1 @ D21.T @ C2) ]])
    # -----------------------
    
    Hx = np.bmat([[ A - B2 @ D12.T @ C1, B1 @ B1.T / (gamma**2) - B2 @ B2.T ], [ -C1.T @ C1 + C1.T @ D12 @ D12.T @ C1, -(A - B2 @ D12.T @ C1).T ]])
    Hy = np.bmat([[ (A - B1 @ D21.T @ C2).T, C1.T @ C1 / (gamma**2) - C2.T @ C2 ], [ -B1 @ B1.T + B1 @ D21.T @ D21 @ B1.T, -(A - B1 @ D21.T @ C2) ]])
    
    X = Ric(Hx)
    Y = Ric(Hy)
      
    """
    r1 = - np.eye(B1.shape[1]) #/ (gamma**2) # 正規化してると gamma 要らない
    r2 = np.eye(B2.shape[1])
    R = sp.linalg.pinv(sp.linalg.block_diag(r1, r2))    
    X_ = sp.linalg.solve_continuous_are(A - B2 @ D12.T @ C1, np.c_[B1, B2], C1.T @ C1 - C1.T @ D12 @ D12.T @ C1, R)

    r1 = - np.eye(C1.T.shape[1]) #/ (gamma**2) # 正規化してると gamma 要らない
    r2 = np.eye(C2.T.shape[1])
    R_ = sp.linalg.pinv(sp.linalg.block_diag(r1, r2))
    Y_ = sp.linalg.solve_continuous_are((A - B1 @ D21.T @ C2).T, np.c_[C1.T, C2.T], B1 @ B1.T - B1 @ D21.T @ D21 @ B1.T, R_)
    
    X = X_.copy()
    Y = Y_.copy()
    """
    
    if (X >= 0).all():
        print("X >= 0 is True !!\n")
    else:
        print("X >= 0 is NOT True...\n")
    if (Y >= 0).all():
        print("Y >= 0 is True !!\n")
    else:
        print("Y >= 0 is NOT True...\n")
    
    Ax = A - B2 @ D12.T @ C1 + ( B1 @ B1.T - B2 @ B2.T ) @ X
    Ay = A - B1 @ D21.T @ C2 + Y @ ( C1.T @ C1 - C2.T @ C2 )
         
    if (Ax < 0).all():
        print("Ax < 0 is True !! Stable Solution !!")
    if (Ay < 0).all():
        print("Ay < 0 is True !! Stable Solution !!")
        
    
    if np.max(np.abs(sp.linalg.eigvals(X @ Y))) < (gamma**2):
        print("lambda_max(XY) < gamma**2 is True !!\n")
    else:
        print("lambda_max(XY) < gamma**2 is False !!\n") 
    print("lambda_max(XY): ", np.max(np.abs(sp.linalg.eigvals(X @ Y))))
        
    """
    B1_hat = B1 @ D21.T + Y @ C2.T
    B2_hat = B2 + Y @ C1.T @ D12 / (gamma**2)
    C1_hat = -(D12.T @ C1 + B2.T @ X) @ sp.linalg.pinv(np.eye(len(Y@X)) - Y @ X / (gamma**2))    
    C2_hat = -(C2 + D21 @ B1.T @ X / (gamma**2)) @ sp.linalg.pinv(np.eye(len(Y@X)) - Y @ X / (gamma**2))
    A_hat = (A - B1 @ D21.T @ C2) + Y @ (C1.T @ C1 / (gamma**2) - C2.T @ C2) + B2_hat @ C1_hat
    
    B_hat = np.bmat([[B1_hat, B2_hat]])
    C_hat = np.bmat([[C1_hat], [C2_hat]])
    D_hat = sp.linalg.block_diag(np.eye(B2_hat.shape[1]), np.eye(B1_hat.shape[1]))[::-1]
    """

    
    Z = sp.linalg.pinv( np.eye(len(Y@X)) - Y @ X )
    Foo = -D12.T @ C1 - B2.T @ X
    Loo = -B1 @ D21.T - Y @ C2.T
    A_hat = A + B1 @ B1.T @ X + B2 @ Foo + Z @ Loo @ C2
    B2_hat = B2 + Y @ C1.T @ D12
    C2_hat = C2 + D21 @ B1.T @ X


    
    # 物理パラメータの定義 -------------
    l = 0.129 # 車軸と車体重心間の距離 [m]
    r = 0.042 # 車輪半径 [m]
    M = 0.318 # 車体質量 [kg]
    m = 0.064 # 車輪質量 [kg]
    J_b = 1.46e-3 # 車体慣性モーメント [kg * m^2]
    J_w = 1.13e-4 # 車輪慣性モーメント [kg * m^2]
    g = 9.8 # 重力加速度 [m / sec^2]
    
    c_1 = m*(r**2) + M*(r+l)**2 + J_w + J_b
    c_2 = J_w + (M+m)*(r**2) + M*l*r
    # ----------------------------
    
    # プラント行列の定義 --------------
    Ap = np.array([[0., 1., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 1.],
                  [0., 0., M*g*l/c_1, 0.]
                 ])
    Bp = np.array([[0.],
                  [1.],
                  [0.],
                  [-c_2/c_1]
                 ])
    #Cp = np.eye(2, 4)
    Cp = np.array([[0., 0., 0., 1.]])
    Dp = np.array([[0]])
    # ----------------------------
    
    """
    Ap = np.array([[1]])
    Bp = np.array([[2]])
    Cp = np.array([[2]])
    Dp = np.array([[0]])
    
    A_k = np.array([[-5]])
    B_k = np.array([[-2]])
    C_k = np.array([[1]])
    D_k = np.array([[0]])
    """
    
    
    
    
    dt = 0.001
    simtime = 15
    time = 0.0
    
    print("Ap: ")
    print(Ap)
    print("Bp: ")
    print(Bp)
    print("Cp: ")
    print(Cp)
    print("Dp: ")
    print(Dp)    

    
    x = np.array([[ 0.0 ],
                  [ 0.0 ],
                  [ 0.1 ],
                  [ 0.0 ]])
    
    x = np.array([[ 0.0 ],
                  [ 0.0 ],
                  [ 0.05 ],
                  [ 0.0 ],
                  [ 0.0 ],
                  [ 0.0 ]])
    
    #x = np.array([[1]])

    
    print("x0: ")
    print(x)
    
    u = np.zeros([Bp.shape[1], 1])
    y = np.zeros([Cp.shape[0], 1])
    
    t_history = []
    dx_history = np.zeros([0, len(x)])
    x_history = np.zeros([0, len(x)])
    y_history = np.zeros([0, len(y)])
    u_history = np.zeros([0, len(u)])

    
    x_k = np.zeros([B2.shape[0], 1])
    

    y_k = np.zeros([C_k.shape[0], 1])
    dx_k_history = np.zeros([0, len(x_k)])
    x_k_history = np.zeros([0, len(x_k)])
    y_k_history = np.zeros([0, len(y_k)])

    z_k = np.zeros([1, 1])
    dz_k = np.zeros([1, 1])
    
    while(time <= simtime):
        
        """
        # プラント側の計算 ------------------
        w = 0.00*np.random.randn(A.shape[1], 1) #　状態外乱の生成
        dx = (A @ x[:A.shape[0]]) + B2 @ (u @ E12**(-1/2)) + w # 状態微分
        x[:A.shape[0]] = x[:A.shape[0]] + dx * dt # 状態遷移(オイラー積分)
        #v = 0*np.random.randn(y.shape[0], 1) # 観測ノイズ生成
        y = (- E21**(-1/2) @ C2) @ (x[:A.shape[0]] + w) #+ Dp @ u # 状態観測
        """
        w = 0.1*np.random.randn(B1.shape[1], 1) #　状態外乱の生成
        rho1 = np.array([[0.0],
                         [-1000/(r/2)],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0]]) # 目標値の設定
        rho2 = np.array([[0.0],
                         [2000/(r/2)],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0]]) # 目標値の設定
        #rho1 = np.array([[0.0]])
        #rho2 = np.array([[0.0]])
        
        if time > 15.: rho = rho2
        else: rho = rho1   
        #w = w + rho
                                  
                               
        dx = A @ x +  B1 @ w +  B2 @ u
        x = x + dx * dt # 状態遷移(オイラー積分)
        z = C1 @ x + D11 @ w + D12 @ u
        y = C2 @ x + D21 @ w #+ D22 @ u
        # -------------------------------
        
        
        # コントローラ側の計算 ----------------

        """
        # オブザーバとしての表現 -----------------
        T = np.eye(len(Y@X)) - (Y @ X )#/ (gamma**2)) # 正規化してると gamma 要らない
        K = B2.T @ X
        H = np.linalg.pinv(T) @ Y @ C2.T
        w_hat = B1.T @ X @ x_k #/ (gamma**2) # 正規化してると gamma 要らない
        
        dx_k = A @ x_k + B1 @ w_hat + B2 @ u + H @ (y - C2 @ x_k) # np.concatenate([rho, y], axis=0) # 推定状態の微分
        x_k = x_k + dx_k * dt # 状態推定
        u = - K @ x_k #+ D_k @ y # np.concatenate([rho, y], axis=0) # 推定状態の観測
        # -----------------------------------
        """

        # シンプルな表現 -----------------------
        dx_k = A_hat @ x_k - (Z @ Loo) @ y # どれも同じ
        #dx_k = (A + B2 @ Foo + Loo @ C2) @ x_k - Loo @ y # どれも同じ
        #dx_k = (A + Loo @ C2) @ x_k - Loo @ y + B2 @ u # どれも同じ
        x_k = x_k + dx_k * dt
        u = Foo @ x_k
        # -----------------------------------                                          
        
        
        dx_history = np.r_[dx_history, dx.T]
        x_history = np.r_[x_history, x.T]
        y_history = np.r_[y_history, E21**(-1/2) @ y.T]
        u_history = np.r_[u_history, E12**(-1/2) @ u.T]
        
        dx_k_history = np.r_[dx_k_history, dx_k.T]
        x_k_history = np.r_[x_k_history, x_k.T]
        y_k_history = np.r_[y_k_history, y_k.T]
        
        t_history.append(time)
        time += dt
    
    plt.figure()
    plt.subplot(211)
    plt.plot(t_history, u_history, color="blue", label=u"$\ u $")
    plt.ylabel(u"$u(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.subplot(212)
    plt.plot(t_history, r*x_history[:, 0], label=u"$\ x $")
    plt.plot(t_history, r*x_history[:, 1], label=u"$\ \dot{x} $")
    plt.plot(t_history, x_history[:, 2], label=u"$\ \phi $")
    plt.plot(t_history, x_history[:, 3], label=u"$\ \dot{\phi} $")
    #plt.plot(t_history, x_history[:, 4], label=u"$\ \phi $")
    #plt.plot(t_history, x_history[:, 5], label=u"$\ \dot{\phi} $")
    plt.xlabel(u"$t [sec]$", fontsize=16)
    plt.ylabel(u"$x(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.subplot(211)
    plt.plot(t_history, u_history, color="blue", label=u"$\ u $")
    plt.ylabel(u"$u(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.subplot(212)
    plt.plot(t_history, r*x_k_history[:, 0], label=u"$\ x $")
    plt.plot(t_history, r*x_k_history[:, 1], label=u"$\ \dot{x} $")
    plt.plot(t_history, x_k_history[:, 2], label=u"$\ \phi $")
    plt.plot(t_history, x_k_history[:, 3], label=u"$\ \dot{\phi} $")
    plt.xlabel(u"$t [sec]$", fontsize=16)
    plt.ylabel(u"$ \hat{x}(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
        
    Ksys = signal.lti(A_hat, - (Z @ Loo), Foo, 0)
    w, mag5, phase5 = signal.bode(Ksys, np.arange(0.01, 10000, 1))
    
    # プロット
    plt.figure(1)
    plt.subplot(211)
    plt.semilogx(w, mag1, label="Plant")
    plt.semilogx(w, mag5, label="Controller")
    plt.ylabel("Amplitude [dB]")
    plt.axis("tight")
    #loc='lower right'で、右下に凡例を表示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    
    