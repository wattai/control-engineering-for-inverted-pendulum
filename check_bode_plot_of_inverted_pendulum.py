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
import control as ct
from control import robust as rb


def Ric(H):
    # Riccati 方程式の '擬似'安定化解 を Hamilton 行列の形で解くソルバ
    H_eval, H_evec = sp.linalg.eig(H)
    return (H_evec[int(len(H)/2):, (H_eval.real < 0)] @ sp.linalg.pinv(H_evec[:int(len(H)/2), (H_eval.real < 0)])).real

def hinfsyn(A, B, C, D, n_in, n_out, gamma):
    # 正則 Hoo 制御問題を, プラントの正則化を行ってから解くソルバ
    # 完成した制御器を, 状態空間表現で返す.
    
    A = A.copy()
    B1 = B[:, :-n_in].copy()
    B2 = B[:, -n_in:].copy()
    C1 = C[:-n_out, :].copy()
    C2 = C[-n_out:, :].copy()
    D11 = D[:-n_out, :-n_in].copy()
    D12 = D[:-n_out, -n_in:].copy()
    D21 = D[-n_out:, :-n_in].copy()
    D22 = D[-n_out:, -n_in:].copy()
    
    
    # 正則化 -----------------------------------------------------
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
    # -----------------------------------------------------------
    
    
    # Normalization ----------------
    B1 = B1 / gamma
    D21 = D21 / gamma
    #C1 = C1 / gamma
    #D12 = D12 / gamma
    
    E12 = D12.T @ D12
    D12 = D12 @ np.sqrt(np.linalg.pinv(E12))
    B2 = B2 @ np.sqrt(np.linalg.pinv(E12))
    
    E21 = D21 @ D21.T
    D21 = np.sqrt(np.linalg.pinv(E21)) @ D21
    C2 = np.sqrt(np.linalg.pinv(E21)) @ C2
    
    
    
    """
    B = np.bmat([[B1, B2]])
    C = np.bmat([[C1], [C2]])
    D = np.bmat([[D11, D12], [D21, D22]])
    Up, _, Rp = np.linalg.svd(D12)
    Rp_tilda, __, Up_tilda = np.linalg.svd(D21)
    tmp1 = sp.linalg.block_diag(Up.T, sp.linalg.inv(Rp_tilda))
    tmp2 = sp.linalg.block_diag(Up_tilda.T, sp.linalg.inv(Rp))
    """
    # -------------------------------
    
    
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
    

    
    # gamma の正規化が前提 ----
    Hx = np.bmat([[ A - B2 @ D12.T @ C1, B1 @ B1.T  - B2 @ B2.T ], [ -C1.T @ C1 + C1.T @ D12 @ D12.T @ C1, -(A - B2 @ D12.T @ C1).T ]])
    Hy = np.bmat([[ (A - B1 @ D21.T @ C2).T, C1.T @ C1  - C2.T @ C2 ], [ -B1 @ B1.T + B1 @ D21.T @ D21 @ B1.T, -(A - B1 @ D21.T @ C2) ]])
    # -----------------------
    
    #Hx = np.bmat([[ A - B2 @ D12.T @ C1, B1 @ B1.T / (gamma**2) - B2 @ B2.T ], [ -C1.T @ C1 + C1.T @ D12 @ D12.T @ C1, -(A - B2 @ D12.T @ C1).T ]])
    #Hy = np.bmat([[ (A - B1 @ D21.T @ C2).T, C1.T @ C1 / (gamma**2) - C2.T @ C2 ], [ -B1 @ B1.T + B1 @ D21.T @ D21 @ B1.T, -(A - B1 @ D21.T @ C2) ]])
    
    #Hx = np.bmat([[ A - B2 @ D12.T @ C1, B1 @ B1.T / (gamma**2) - B2 @ sp.linalg.pinv(E12) @ B2.T ], [ -C1.T @ C1 + C1.T @ D12 @ D12.T @ C1, -(A - B2 @ D12.T @ C1).T ]])
    #Hy = np.bmat([[ (A - B1 @ D21.T @ C2).T, C1.T @ C1 / (gamma**2) - C2.T @ C2 ], [ -B1 @ B1.T + B1 @ D21.T @ D21 @ B1.T, -(A - B1 @ D21.T @ C2) ]])
    
    
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
    
    eigval_absmax = np.max(np.abs(sp.linalg.eigvals(X @ Y)))
    #if eigval_absmax < (gamma**2):
    if eigval_absmax < 1:
        print("eigval_max(XY) < 1 or gamma**2 is True !!\n")
    else:
        print("eigavl_max(XY) < 1 or gamma**2 is False !!\n") 
    print("eigval_max(XY): ", eigval_absmax)
        
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
    #D11_hat = D_hat[:C1_hat.shape[0], :B1_hat.shape[1]]
    
    
    Z = sp.linalg.pinv( np.eye(len(Y@X)) - Y @ X )
    Foo = -D12.T @ C1 - B2.T @ X
    Loo = -B1 @ D21.T - Y @ C2.T
    B2_hat = B2 + Y @ C1.T @ D12
    C2_hat = C2 + D21 @ B1.T @ X
    A_hat = A + B1 @ B1.T @ X + B2 @ Foo + Z @ Loo @ C2_hat
    

    Ak = A_hat.copy()
    Bk = ( -(Z @ Loo) @ (- np.sqrt(np.linalg.pinv(E21))) ).copy()
    Ck = ( np.sqrt(np.linalg.pinv(E12)) @ Foo ).copy()
    #Bk = ( -(Z @ Loo) ).copy()
    #Ck = ( Foo ) .copy()
    
    return A_hat, Bk, Ck
    
def generate_inverted_pendulum_model(real=False):
    
    if real == True:
        # 物理パラメータの定義 -------------
        l = 0.129 + 0.01 * np.random.randn() # 車軸と車体重心間の距離 [m]
        r = 0.042 + 0.01 * np.random.randn() # 車輪半径 [m]
        M = 0.318 + 0.1 * np.random.randn() # 車体質量 [kg]
        m = 0.064 + 0.01 * np.random.randn() # 車輪質量 [kg]
        J_b = 1.46e-3 + 0.001 * np.random.randn() # 車体慣性モーメント [kg * m^2]
        J_w = 1.13e-4 + 0.0001 * np.random.randn() # 車輪慣性モーメント [kg * m^2]
        g = 9.8 # 重力加速度 [m / sec^2]
        params = [l, r, M, m, J_b, J_w, g]
        
        c_1 = m*(r**2) + M*(r+l)**2 + J_w + J_b
        c_2 = J_w + (M+m)*(r**2) + M*l*r
        # ----------------------------
    elif real == False:
        # 物理パラメータの定義 -------------
        l = 0.129 #+ 0.01 * np.random.randn() # 車軸と車体重心間の距離 [m]
        r = 0.042 #+ 0.01 * np.random.randn() # 車輪半径 [m]
        M = 0.318 #+ 0.1 * np.random.randn() # 車体質量 [kg]
        m = 0.064 #+ 0.01 * np.random.randn() # 車輪質量 [kg]
        J_b = 1.46e-3 #+ 0.001 * np.random.randn() # 車体慣性モーメント [kg * m^2]
        J_w = 1.13e-4 #+ 0.0001 * np.random.randn() # 車輪慣性モーメント [kg * m^2]
        g = 9.8 # 重力加速度 [m / sec^2]
        params = [l, r, M, m, J_b, J_w, g]
        
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
    #Cp = np.eye(4, 4)
    Cp = np.array([[0., 0., 0., 1.]])
    #Dp = np.array([[0]])
    Dp = np.zeros([Cp.shape[0], 1])
    # -----------------------------
            
    return Ap, Bp, Cp, Dp, params
    
def generate_mixed_sensitivity_generalized_plant(Ap, Bp, Cp, Dp, Ws, Wt):

    # 一般化プラントの生成 ---------------------------------------------------------   
    a_x = np.concatenate([Ap, np.zeros([Ap.shape[0], Ws.A.shape[1] + Wt.A.shape[1]])], axis=1)
    a_ws = np.concatenate([Ws.B @ Cp, Ws.A, np.zeros([Ws.A.shape[0], Wt.A.shape[1]])], axis=1)
    a_wt = np.concatenate([Wt.B @ Cp, np.zeros([Wt.A.shape[0], Ws.A.shape[1]]), Wt.A], axis=1)
    
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
    
    #A = np.concatenate([a_x, a_ws, a_wt], axis=0)    
    A = np.bmat([[a_x], [a_ws], [a_wt]])
    B = np.bmat([[B1, B2]])
    C = np.bmat([[C1], [C2]])
    D = np.bmat([[D11, D12], [D21, D22]])
    
    P = np.bmat([[A, B], [C, D]])
    # ------------------------------------------------------------------------
    
    return A, B, C, D
    
def generate_test_generalized_plant():
    
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
    
    B = np.bmat([[B1, B2]])
    C = np.bmat([[C1], [C2]])
    D = np.bmat([[D11, D12], [D21, D22]])
    
    return A, B, C, D
    
def load_scilab_hinf_controller():
    
    # Scilab にて計算した H inf 制御器 -----------------
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
    
    return A_k, B_k, C_k, D_k

if __name__ == "__main__":
    
 
    # プラント行列の定義 --------------
    Ap, Bp, Cp, Dp, params = generate_inverted_pendulum_model(real = False)
    # -----------------------------

    # Scilab で設計した Hinf 制御器のロード -----------
    A_k, B_k, C_k, D_k = load_scilab_hinf_controller()
    # ------------------------------------------
    

    # 入力-評価出力の伝達関数 (このゲインを上げれば,その周波数での入力の追従速度が上がる)
    fcs = 1000
    wcs = 2 * np.pi * fcs
    numws = [10, 0] # バネマスダンパ系デフォ
    denws = [1/0.001, wcs]
    Ws = signal.lti(numws, denws)
    
    # 外乱-評価出力の伝達関数 (このゲインを上げれば,その周波数での外乱の抑圧速度が上がる)
    fct = 0.01
    wct = 2 * np.pi * fct
    numwt = [0, wct] # バネマスダンパ系デフォ
    denwt = [1/1, wct]
    Wt = signal.lti(numwt, denwt)
    
    # 入れ替え -------
    #s = Ws
    #Ws = Wt
    #Wt = s
    # --------------

    # LTIシステムインスタンスの生成    
    sys_plant = signal.lti(Ap, Bp, Cp, Dp)
    sys_controller = signal.lti(A_k, B_k, C_k, D_k)
    
    #w, mag1, phase1 = signal.bode(sys_plant, np.arange(0.01, 10000, 1)) 
    #w, mag2, phase2 = signal.bode(sys_controller, np.arange(0.01, 10000, 1))
    w, mag3, phase3 = signal.bode(Ws, np.arange(0.01, 10000, 1))
    w, mag4, phase4 = signal.bode(Wt, np.arange(0.01, 10000, 1))
    
    
    # プロット
    plt.figure(1)
    plt.subplot(211)

    #plt.semilogx(w, mag1, label="Plant")
    #plt.semilogx(w, mag2, label="Scilab_Controller")
    plt.semilogx(w, mag3, label="Ws")
    plt.semilogx(w, mag4, label="Wt")
    plt.ylabel("Amplitude [dB]")
    plt.axis("tight")
    #loc='lower right'で、右下に凡例を表示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    plt.subplot(212)
    #plt.semilogx(w, phase1, label="Plant")
    #plt.semilogx(w, phase2, label="Scilab_Controller")
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
    
    
    
    # 一般化プラントの生成 -----------------------------------------------------------------
    A, B, C, D = generate_mixed_sensitivity_generalized_plant(Ap, Bp, Cp, Dp, Ws, Wt)
    # --------------------------------------------------------------------------------
    
    
    # Hoo 制御器の計算 ------------------------------------
    gamma = 1.004559375
    gamma = 0.85000
    gamma = 0.02
    gamma = 0.0286
    #gamma = 110
    n_in = Bp.shape[1] # 一般化プラントへの入力数
    n_out = Cp.shape[0] # 一般化プラントからの出力数
    Ak, Bk, Ck = hinfsyn(A, B, C, D, n_in, n_out, gamma)
    
    sys = ct.StateSpace(A, B, C, D, 0.001)
    #sysd = sys.sample(0.001, method='bilinear')
    """
    P = sys_plant
    from slycot import sb10ad
    job = 3
    ncon = 1#n_out
    nmeas = 1#n_in
    n = np.size(P.A,0)
    m = np.size(P.B,1)
    np_ = np.size(P.C,0)
    gamma = 1.e100
    out = sb10ad(job,n,m,np_,ncon,nmeas,gamma,P.A,P.B,P.C,P.D)
    """
    #K, CL, gam, rcond = rb.hinfsyn(sys_plant, 1, 1)
    Q = np.diag([1, 1, 1, 1])
    R = np.array([[1]])
    #K, S, E = rb.lqr(sys_plant, Q, R)#, [N])
    
    # ---------------------------------------------------


    # "実"プラント行列の定義 --------------
    #Ap, Bp, Cp, Dp, params = generate_inverted_pendulum_model(real = True)
    # -------------------------------

    #Ak = A_k.copy()
    #Bk = B_k.copy()
    #Ck = C_k.copy()
    
    dt = 0.001
    simtime = 16
    time = 0.0
    
    print("Ap: ")
    print(Ap)
    print("Bp: ")
    print(Bp)
    print("Cp: ")
    print(Cp)
    print("Dp: ")
    print(Dp)    

    r = 0.042
    x = np.array([[ -1.0/r ],
                  [ 0.0 ],
                  [ 0.5 ],
                  [ 0.0 ]])
    """
    x = np.array([[ 0.0 ],
                  [ 0.0 ],
                  [ 0.1 ],
                  [ 0.0 ],
                  [ 0.0 ],
                  [ 0.0 ]])
    """
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

    
    x_k = np.zeros([Bk.shape[0], 1])
    

    y_k = np.zeros([Ck.shape[0], 1])
    dx_k_history = np.zeros([0, len(x_k)])
    x_k_history = np.zeros([0, len(x_k)])
    y_k_history = np.zeros([0, len(y_k)])

    z_k = np.zeros([1, 1])
    dz_k = np.zeros([1, 1])
    
    while(time <= simtime):
        
        
        rho1 = np.array([[5*np.pi],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0]]) # 目標値の設定
        rho2 = np.array([[-5*np.pi],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0],
                         [0.0]]) # 目標値の設定
        #rho1 = np.array([[0.0]])
        #rho2 = np.array([[0.0]])
        
        if time > 5.: rho = rho2
        else: rho = rho1
        #w = w + rho
        # プラント側の計算 ------------------
        w = 0.1*np.random.randn(x.shape[0], 1) #　状態外乱の生成
        dx = (Ap @ x) + (Bp @ u) + w #+ rho[:4,:] # 状態微分
        x = x + dx * dt # 状態遷移(オイラー積分)
        v = 0.0*np.random.randn(y.shape[0], 1) # 観測ノイズ生成
        y = Cp @ x #+ v #+ Cp @ w #+ Dp @ u # 状態観測
        """
        w = 0.00*np.random.randn(B1.shape[1], 1) #　状態外乱の生成
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
        """
        
        # コントローラ側の計算 --------------------
        dx_k = Ak @ x_k + Bk @ y #- rho
        x_k = x_k + dx_k * dt
        u = Ck @ x_k
        # -----------------------------------                                          
        
        
        dx_history = np.r_[dx_history, dx.T]
        x_history = np.r_[x_history, x.T]
        y_history = np.r_[y_history, y.T]
        u_history = np.r_[u_history, u.T]
        
        dx_k_history = np.r_[dx_k_history, dx_k.T]
        x_k_history = np.r_[x_k_history, x_k.T]
        y_k_history = np.r_[y_k_history, y_k.T]
        
        t_history.append(time)
        time += dt
    
    
    r = params[1] # タイヤの半径を読み込み
        
    plt.figure()
    plt.subplot(211)
    plt.plot(t_history, u_history, color="blue", label=u"$\ u $")
    plt.ylabel(u"$u(t)$", fontsize=16)
    plt.grid(True)
    #loc='lower right'で、右下に凡例を表示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
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
    #loc='lower right'で、右下に凡例を表示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.subplot(211)
    plt.plot(t_history, u_history, color="blue", label=u"$\ u $")
    plt.ylabel(u"$u(t)$", fontsize=16)
    plt.grid(True)
    #loc='lower right'で、右下に凡例を表示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.subplot(212)
    plt.plot(t_history, r*x_k_history[:, 0], label=u"$\ x $")
    plt.plot(t_history, r*x_k_history[:, 1], label=u"$\ \dot{x} $")
    plt.plot(t_history, x_k_history[:, 2], label=u"$\ \phi $")
    plt.plot(t_history, x_k_history[:, 3], label=u"$\ \dot{\phi} $")
    plt.xlabel(u"$t [sec]$", fontsize=16)
    plt.ylabel(u"$ \hat{x}(t)$", fontsize=16)
    plt.grid(True)
    #loc='lower right'で、右下に凡例を表示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()
    
    """
    Ksys = signal.lti(Ak, Bk, Ck, 0)
    w, mag5, phase5 = signal.bode(Ksys, np.arange(0.01, 10000, 1))
    
    # プロット
    plt.figure(1)
    plt.subplot(211)
    #plt.semilogx(w, mag1, label="Plant")
    #plt.semilogx(w, mag2, label="Scilab_Controller")
    plt.semilogx(w, mag3, label="Ws")
    plt.semilogx(w, mag4, label="Wt")
    plt.semilogx(w, mag5, label="Hinf_Controller")
    plt.ylabel("Amplitude [dB]")
    plt.axis("tight")
    #loc='lower right'で、右下に凡例を表示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    """

    