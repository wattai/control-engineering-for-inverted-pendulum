#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:39:21 2017

@author: wattai
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 00:10:55 2017

@author: Watanabe
"""

from control.matlab import ctrb, obsv, place, ss, c2d, tf, bode, tf2ss
from control.robust import hinfsyn
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from invpen_moment import invpen_moments

# 可制御性のチェック
def check_ctrb(A, B):
    Uc = ctrb(A, B) # 可制御性行列の計算
    Nu = np.linalg.matrix_rank(Uc)  # Ucのランクを計算
    (N, N) = np.matrix(A).shape     # 正方行列Aのサイズ(N*N)
    # 可制御性の判別
    if Nu == N: return 0            # 可制御
    else: return -1                 # 可制御でない

# 可観測性のチェック
def check_obsv(A, C):
    Uo = obsv(A, C) # 可制御性行列の計算
    No = np.linalg.matrix_rank(Uo)  # Ucのランクを計算
    (N, N) = np.matrix(A).shape     # 正方行列Aのサイズ(N*N)
    # 可制御性の判別
    if No == N: return 0            # 可観測
    else: return -1                 # 可観測でない
    
def _clqr(A,B,Q,R):
    """Solve the continuous time lqr controller.
    x[t] = x[t] + { A x[t] + B u[t] } dt
    cost = integral{ x[t].T*Q*x[t] + u[t].T*R*u[t] } dt from 0 to INF
    """
    #first, try to solve the ricatti equation
    P = la.solve_continuous_are(A, B, Q, R)
    #compute the LQR gain
    K = la.inv(R) @ B.T @ P
    
    eigVals, eigVecs = la.eig(A - B @ K)
    
    return K, P, eigVals
    
def _dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #first, try to solve the ricatti equation
    X = la.solve_discrete_are(A, B, Q, R)
    #compute the LQR gain
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)
    
    eigVals, eigVecs = la.eig(A - B @ K)
     
    return K, X, eigVals

def FofPn_ss(dt=0.0005, omega_d=100, Km=10, Tm=0.1):
    
    #omega_d = 100.
    #Km = 10.
    #Tm = 0.10
    
    num = [Tm*omega_d/Km, omega_d/Km]
    den = [1, omega_d]
    
    sysc = tf(num, den)
    sysd = sysc.sample(Ts=dt, method='tustin', alpha=None)
    ss_d = tf2ss(sysd)
    
    print('FofPn: ', sysd)
    """
    plt.figure()
    bode(sysc)
    plt.show()
    """
    return ss_d

def Pn_ss(dt=0.0005, Km=10, Tm=0.1):
    
    num = [0., Km]
    den = [Tm, 1.]
    
    sysc = tf(num, den)
    sysd = sysc.sample(Ts=dt, method='tustin', alpha=None)
    ss_d = tf2ss(sysd)
    
    print('Pn: ', sysd)
    """
    plt.figure()
    bode(sysc)
    plt.show()
    """
    return ss_d
    
def P_ss(dt=0.0005, Km=8, Tm=0.20):
    
    num = [0., Km]
    den = [Tm, 1.]
    
    sysc = tf(num, den)
    sysd = sysc.sample(Ts=dt, method='tustin', alpha=None)
    ss_d = tf2ss(sysd)
    
    print('P: ', sysd)
    """
    plt.figure()
    bode(sysc)
    plt.show()
    """
    return ss_d  

def P_ss_2nd(dt=0.0005, Km=8, Tm=0.20, tau=0.05):
    
    num = [0., 0., Km]
    den = [Tm*tau, Tm+tau, 1.]
    
    sysc = tf(num, den)
    sysd = sysc.sample(Ts=dt, method='tustin', alpha=None)
    ss_d = tf2ss(sysd)
    
    print('P: ', sysd)
    """
    plt.figure()
    bode(sysc)
    plt.show()
    """
    return ss_d  

def FofPn_ss_2nd(dt=0.0005, omega_d=100, Km=8, Tm=0.20, tau=0.05):
    
    num = [Tm*tau*(omega_d**2), (Tm+tau)*(omega_d**2), omega_d**2]
    den = [Km, 2*Km*omega_d, Km*(omega_d**2)]
    
    sysc = tf(num, den)
    sysd = sysc.sample(Ts=dt, method='tustin', alpha=None)
    ss_d = tf2ss(sysd)
    
    print('P: ', sysd)
    """
    plt.figure()
    bode(sysc)
    plt.show()
    """
    return ss_d  
    
    

    
if __name__ == "__main__":
    
    """
    Arduino と LEGO部品 で構成される倒立振子に対して，
    x_dot = A @ x + B @ u
    y = C @ x
    で表現される状態方程式を基に，最適レギュレータにて制御を行う。
    
    状態変数の内訳は，
    x = [ dtheta1, theta1, dtheta2 ]
      =　[ 姿勢角速度[rad], 姿勢角[rad/sec], 車輪角速度[rad] ]
    出力変数は,
    u = [ 角速度目標値[rad/s] ]
    
    ジャイロセンサのみの場合，システムが可観測ではないので，
    カルマンフィルタを用いたオブザーバの設計は不可能(極配置は可能だが，任意には置けない)
    """
  
    dt = 0.006
    simtime = 5
    time = 0.0
    
    # 物理パラメータの定義 -------------
    l = 0.085 #+ 0.01 * np.random.randn() # 車軸と車体重心間の距離 [m]
    r = 0.027 #+ 0.01 * np.random.randn() # 車輪半径 [m]
    m1 = 0.279+0.030 #+ 0.10 * np.random.randn() # 車体質量 [kg]
    m2 = 0.014 #+ 0.01 * np.random.randn() # 車輪質量 [kg]
    J1, J2 = invpen_moments(m1, m2)
    #J1 = 1.46e-3 #+ 0.001 * np.random.randn() # 車体慣性モーメント [kg * m^2]
    #J2 = 1.13e-4 #+ 0.001 * np.random.randn() # 車輪慣性モーメント [kg * m^2]
    Jm = 0.02 # モータのアーマチュア慣性モーメント [kg * m^2]
    n = 1/114.7 #114.7 # ギア比
    Tm = 0.20 # モータの時定数 (0.07?)
    c = 11./15. # タイヤの減衰係数[N*m/(rad/s)]
    #c = 1.0
    g = 9.80665 # 重力加速度 [m / sec^2]
    
    delta = ( m1*(l**2) + J1 + (n**2)*Jm ) * ( (m1+m2)*(r**2) + J2 * (n**2)*Jm ) - (m1*r*l - (n**2)*Jm)**2
    a11 = - c * ( (m1+m2)*(r**2) + J2 + (n**2)*Jm ) / delta
    a12 = m1*g*l*( (m1+m2)*(r**2) + J2 + (n**2)*Jm ) / delta
    a13 = -a11 #c * ( (m1+m2)*(r**2) + J2 + (n**2)*Jm ) / delta # -a11
    a31 = c * ( m1*(l**2) + J1 + m1*r*l ) / delta
    a32 = - m1*g*l*( m1*r*l - (n**2)*Jm ) / delta
    a33 = -a31 #- c * ( m1*(l**2) + J1 + m1*r*l ) / delta # -a31
    b1 = a11/c #- ( (m1+m2)*(r**2) + J2 + (n**2)*Jm ) / delta # a11/c
    b3 = -a33/c #( m1*(l**2) + J1 + m1*r*l ) / delta # -a33/c
    A11 = a11 - b1/b3*a31
    A12 = a12 - b1/b3*a32
    #A13 = a13 - b1/b3*a32 - b1/(b3*Tm) # 論文が間違えている
    A13 = a13 - b1/b3*a33 - b1/(b3*Tm) # 論文が間違えている
    B1 = b1/(b3*Tm)
    A33 = - 1./Tm
    B3 = 1./Tm
    # -----------------------------
    
    
    # システム行列の定義 --------------
    Ac = np.array([[A11, A12, A13, 0.],
                   [1., 0., 0., 0.],
                   [0., 0., A33, 0.],
                   [0., 0., 1., 0.]])
    Bc = np.array([[B1],
                   [0.],
                   [B3],
                   [0.]])
    Cc = np.array([[1., 0., 0., 0.], # ジャイロの値
                   [0., 1., 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]]) # 車輪の角度
    #C = np.eye(len(A))
    Dc = np.array([[0.],
                   [0.],
                   [0.],
                   [0.]])
    sysd = ss(Ac, Bc, Cc, Dc).sample(Ts=dt, method='tustin', alpha=None)
    A, B, C, D = sysd.A, sysd.B, sysd.C, sysd.D
    
    
    Ace = np.array([[A11, A12, A13,  0., 0., 0., 0., 0.],
                   [ 1.,  0.,   0.,  0., 0., 0., 0., 0.],
                   [ 0.,  0.,  A33,  0., 0., 0., 0., 0.],
                   [ 0.,  0.,   1.,  0., 0., 0., 0., 0.],
                   [ -1., 0.,   0.,  0., 0., 0., 0., 0.],
                   [ 0.,  -1.,  0.,  0., 0., 0., 0., 0.],
                   [ 0.,  0.,  -1.,  0., 0., 0., 0., 0.],
                   [ 0.,  0.,   0., -1., 0., 0., 0., 0.]])
    
    Bce = np.array([[B1],
                   [0.],
                   [B3],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.]])
    Cce = np.eye(len(Ace))
    Dce = np.zeros([Cce.shape[0], Bce.shape[1]])
    sysd = ss(Ace, Bce, Cce, Dce).sample(Ts=dt, method='tustin', alpha=None)
    Ae, Be, Ce, De = sysd.A, sysd.B, sysd.C, sysd.D
    # -----------------------------
    
    A_theta = np.array([[-1./Tm]])
    B_theta = np.array([[1./Tm]])
    C_theta = np.array([[1.]])
    D_theta = np.array([[0.]])
    q = np.zeros([1, 1])
    theta = np.zeros([1, 1])
    dtheta = np.zeros([1, 1])
    sysd = ss(A_theta, B_theta, C_theta, D_theta).sample(Ts=dt, method='tustin', alpha=None)
    A_theta, B_theta, C_theta, D_theta = sysd.A, sysd.B, sysd.C, sysd.D

    # システムが可制御・可観測でなければ終了
    if check_ctrb(A, B) == -1 :
        print("システムが可制御でないので終了")
        #return 0
    if check_obsv(A, C) == -1 :
        print("システムが可観測でないので終了")
        #return 0
    
    # 最適レギュレータ重みの定義 --------------
    ##R = np.array([[1]]) # 入力の大きさに対するペナルティ
    ##Q = np.diag([1.0, 70000.0, 50.0, 20.0]) # 各変数の重要度
    #R = np.array([[200]]) # 入力の大きさに対するペナルティ Tm=0.20 n=1/38.2
    #Q = np.diag([1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 50000, 0.06]) # 各変数の重要度 Tm=0.20 n=1/38.2
    
    #R = np.array([[3000]]) # 入力の大きさに対するペナルティ Tm=0.07 n=1/38.2
    #Q = np.diag([1.0, 100.0, 10.0, 10.0, 1.0, 1.0, 100000, 0.2]) # 各変数の重要度 Tm=0.07 n=1/38.2
    #R = np.array([[3000]]) # 入力の大きさに対するペナルティ Tm=0.07 n=1/114.7
    #Q = np.diag([1.0, 100.0, 10.0, 10.0, 1.0, 1.0, 50000, 1.0]) # 各変数の重要度 Tm=0.07 n=1/114.7
    R = np.array([[100]]) # 入力の大きさに対するペナルティ Tm = 0.21 n=1/114.7
    Q = np.diag([1.0, 1.0, 15000.0, 20000.0, 100.0, 0.1]) # 各変数の重要度 Tm=0.21 n=1/114.7
    R = np.array([[300]]) # 入力の大きさに対するペナルティ Tm = 0.21 n=1/114.7
    Q = np.diag([1.0, 100.0, 10.0, 10.0, 1000.0, 20000, 0.07]) # 各変数の重要度 Tm=0.21 n=1/114.7
    R = np.array([[300]]) # 入力の大きさに対するペナルティ Tm = 0.21 n=1/114.7
    Q = np.diag([1.0, 100.0, 1.0, 1.0, 11000.0, 10000, 0.1]) # 各変数の重要度 Tm=0.10 n=1/114.7
    Q = np.diag([1.0, 0.0, 1.0, 1.0, 11000.0, 5000, 0.1]) # 各変数の重要度 Tm=0.10 n=1/114.7
    
    R = np.array([[300]]) # 入力の大きさに対するペナルティ Tm = 0.21 n=1/114.7
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 11000.0, 1.0, 100000, 0.1]) # 各変数の重要度 Tm=0.10 n=1/114.7
    
    
    #Q = np.diag([1.0, 10000.0, 1.0])
    #Q = np.sqrt(C.T @ C)
    # -----------------------------------
    
    
    # 最適レギュレータの設計
    K, P, e = _dlqr(Ae, Be, Q, R)
    Kc, Pc, ec = _clqr(Ace, Bce, Q, R)
    
    # 結果表示
    print("リカッチ方程式の解:\n", P)
    print("状態フィードバックゲイン:\n", K)
    print("閉ループ系の固有値:\n", e)
    

    
    """
    # オブザーバシステム行列の定義 ------------
    p_obs = np.array([-300e+00 +0j, -324e+00 -0j, -1.0e-00 +0j]) # 2次元配列にするとplaceでエラー出る
    #p_obs = 1 * e.real # 2次元配列にするとplaceでエラー出る
    K_obs = place(A.T, C.T, p_obs)
        
    A_obs = A - K_obs.T @ C
    B_obs = np.c_[B, K_obs.T]
    C_obs = np.eye(4, 4)
    
    print("A_obs:\n", A_obs)
    print("B_obs:\n", B_obs)
    eval_obs, evec_obs = la.eig(A_obs)
    print("オブザーバの固有値:\n", eval_obs)
    # -----------------------------------
    
    """
    
    # カルマンフィルタ型 オブザーバの重みの定義 ------
    #R_obs = np.eye(C.shape[0])
    ##R_obs = np.diag([50000])
    ##Q_obs = np.diag([10.0, 200.0, 1.0, 200.0]) # 分散値の指定
    """
    R_obs = np.diag([0.000838, 0.600591, 0.011153])
    Q_obs = np.diag([96540.0, 2000.0, 10000000.0, 400000.0])*0.00001 # 分散値の指定
    Q_obs = np.diag([156540.0, 2000.0, 10000000.0, 400000.0])*0.000004 # 分散値の指定
    """
    R_obs = np.diag([0.000838, 0.5, 0.600591, 0.011153])
    Q_obs = np.diag([1000.0, 10, 100000.0, 10000.0])*0.001 # 分散値の指定
    
    #Q_obs = C.T @ C
    #Q_obs = np.diag([1.0, 1.0, 1.0, 1.0])*0.1 # 分散値の指定
    # ---------------------------------------
    
    # カルマンフィルタ型 オブザーバシステム行列の定義 --
    Kc_obs, Pc_obs, ec_obs = _clqr(Ac.T, Cc.T, Q_obs, R_obs)
    
    K_obs, P_obs, e_obs = _dlqr(A.T, C.T, Q_obs, R_obs)
    A_obs = A - K_obs.T @ C
    B_obs = np.c_[B, K_obs.T]
    C_obs = np.eye(4, 4)
    
    print("A_obs:\n", A_obs)
    print("B_obs:\n", B_obs)
    print("Q_obs:\n", Q_obs)
    print("R_obs:\n", R_obs)
    # ---------------------------------------
    
    # 結果表示 ---------------------------
    print("オブザーバ側 リカッチ方程式の解:\n", P_obs)
    print("オブザーバ側 状態フィードバックゲイン:\n", K_obs)
    print("オブザーバ側 閉ループ系の固有値:\n", e_obs)
    # -----------------------------------
    
    
    print("A: ")
    print(A)
    print("B: ")
    print(B)
    print("C: ")
    print(C)
    print("Q: ")
    print(Q)
    print("R: ")
    print(R)
    
    #print("p_obs: ")
    #print(p_obs)
    #print("K_obs: ")
    #print(K_obs)
    

    x = np.array([[ 0.0 * np.random.randn() ],
                  [ 5 / 180. * np.pi + 0.0 * np.random.randn() ],
                  [ 0.0 * np.random.randn() ],
                  [ 0.0 ]])
    print("x0: ")
    print(x)
    
    u = np.zeros([B.shape[1], 1])
    y = np.zeros([C.shape[0], 1])
    z = np.zeros([B.shape[1], 1])
    p = np.zeros([B.shape[1], 1])

    t_history = []
    dx_history = np.zeros([0, len(x)])
    x_history = np.zeros([0, len(x)])
    y_history = np.zeros([1, len(y)])
    u_history = np.zeros([0, len(u)])
    z_history = np.zeros([0, len(u)])
    p_history = np.zeros([0, len(u)])

    
    x_hat = np.zeros([x.shape[0], 1])

    y_hat = np.zeros([C.shape[0], 1])
    dx_hat_history = np.zeros([0, len(x)])
    x_hat_history = np.zeros([0, len(x)])
    y_hat_history = np.zeros([0, len(y_hat)])
    

    from scipy import signal as sig
    fs = 1./dt
    wp = 0.025 / (fs/2) # 0.4[Hz] からパス
    ws = 0.0125 / (fs/2) # 0.2[Hz] までストップ
    gpass = 1
    gstop = 3
    b, a = sig.iirdesign(wp, ws, gpass, gstop, analog=False, ftype='butter', output='ba')
    y_ = np.zeros([C.shape[0], 1])
    print("a:", a)
    print("b:", b)
    
    
    
    # 物理パラメータの定義 -------------
    l = 0.085  #+ 0.01 * np.random.randn() # 車軸と車体重心間の距離 [m]
    r = 0.027 #+ 0.01 * np.random.randn() # 車輪半径 [m]
    m1 = 0.279+0.030 #+ 0.10 * np.random.randn() # 車体質量 [kg]
    m2 = 0.014 #+ 0.01 * np.random.randn() # 車輪質量 [kg]
    J1, J2 = invpen_moments(m1, m2)
    #J1 = 1.46e-3 #+ 0.001 * np.random.randn() # 車体慣性モーメント [kg * m^2]
    #J2 = 1.13e-4 #+ 0.001 * np.random.randn() # 車輪慣性モーメント [kg * m^2]
    Jm = 0.02 #+ 0.01 * np.random.randn() # モータのアーマチュア慣性モーメント [kg * m^2]
    n = 1./114.7 #114.7 # ギア比
    Tm = 0.20 #+ 0.01 * np.random.randn() # モータの時定数 (0.07?)
    c = 11./15. #+ 0.1 * np.random.randn() # タイヤの減衰係数[N*m/(rad/s)]
    #c = 1.0
    g = 9.80665 # 重力加速度 [m / sec^2]
    
    
    delta = ( m1*(l**2) + J1 + (n**2)*Jm ) * ( (m1+m2)*(r**2) + J2 * (n**2)*Jm ) - (m1*r*l - (n**2)*Jm)**2
    a11 = - c * ( (m1+m2)*(r**2) + J2 + (n**2)*Jm ) / delta
    a12 = m1*g*l*( (m1+m2)*(r**2) + J2 + (n**2)*Jm ) / delta
    a13 = -a11 #c * ( (m1+m2)*(r**2) + J2 + (n**2)*Jm ) / delta # -a11
    a31 = c * ( m1*(l**2) + J1 + m1*r*l ) / delta
    a32 = - m1*g*l*( m1*r*l - (n**2)*Jm ) / delta
    a33 = -a31 #- c * ( m1*(l**2) + J1 + m1*r*l ) / delta # -a31
    b1 = a11/c #- ( (m1+m2)*(r**2) + J2 + (n**2)*Jm ) / delta # a11/c
    b3 = -a33/c #( m1*(l**2) + J1 + m1*r*l ) / delta # -a33/c
    A11 = a11 - b1/b3*a31
    A12 = a12 - b1/b3*a32
    #A13 = a13 - b1/b3*a32 - b1/(b3*Tm) # 論文が分子におくべきTmを分母に置き間違えている
    A13 = a13 - b1/b3*a33 - b1/(b3*Tm) # 論文が分子におくべきTmを分母に置き間違えている
    B1 = b1/(b3*Tm)
    A33 = - 1./Tm
    B3 = 1./Tm
    # -----------------------------
    
    
    # システム行列の定義 --------------
    A = np.array([[A11, A12, A13, 0.],
                  [1., 0., 0., 0.],
                  [0., 0., A33, 0.],
                  [0., 0., 1., 0.]])
    B = np.array([[B1],
                  [0.],
                  [B3],
                  [0.]])
    C = np.array([[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]])
    #C = np.eye(len(A))
    D = np.array([[0.],
                  [0.],
                  [0.],
                  [0.]])
    sysd = ss(A, B, C, D).sample(Ts=dt, method='tustin', alpha=None)
    A, B, C, D = sysd.A, sysd.B, sysd.C, sysd.D
    # -----------------------------
    
    
    print("ec", ec)
    print("ec_obs", ec_obs)
       
    power = 0.0
    
    err = np.zeros([1, 1])
    err_1 = np.zeros([1, 1])
    err_2 = np.zeros([1, 1])
    p = np.zeros([1, 1])
    
    x_r = np.zeros([C.shape[0], 1])
    x_rr = np.zeros([C.shape[0], 1])

    err_I = np.zeros([1, 1])

    while(time <= simtime):
        
        ref = np.array([[0.0],
                        [0.0],
                        [0.0],
                        [0.0]])
        
        if time >= 2.0:
            #ref = np.zeros([2, 1])
            ref = np.array([[0.0],
                            [0 * np.sin(2*np.pi*time)],
                            [0.0],
                            [0.0]])
        
        
        
        # プラント側の計算 (実装時にはいらない) -------------------------------------------------
        w = 0.001*np.random.randn(x.shape[0], 1) # 状態外乱の生成
        x = A @ x + B @ u #+ w #+ np.r_[w, np.zeros([1, 1])] # 状態微分
        #x = x + dx * dt # 状態遷移(オイラー積分)
        v = 0.01*np.random.randn(y.shape[0], 1) # 観測ノイズ生成
        dri = -0.03 # センサの ドリフト成分生成
        y = C @ x #+ v #+ D @ u # 状態観測 (観測ノイズを 含む)
        y[0] += dri # ドリフトノイズ印加
        # --------------------------------------------------------------------------------
        
        # High pass filter to avoid drift phenomenon. --------
        #y_ = b[0]*y + b[1]*y_history[-1,:] - a[1]*y_
        # ----------------------------------------------------

        # オブザーバ側の計算 ----------------------------------------------------------------
        #x_hat = A @ x_hat + B @ u + K_obs.T @ (y - y_hat) # 推定状態の微分
        x_hat = A_obs @ x_hat + B_obs @ np.r_[u, y] # 推定状態の微分
        #x_hat = x_hat + dx_hat * dt #- rho # 状態推定

        #x_hat = A @ x_hat + B @ u + K_obs.T @ (y - C @ x_hat) # 状態推定(離散系)
        y_hat = C @ x_hat # 推定状態の観測
        # --------------------------------------------------------------------------------
    
        x_r += ref - y # 目標値追従のため, 出力と目標値との差を積分
        x_rr += x_r
    
        u = - K[:, :4] @ x_hat[:4] - K[:, 4:] @ np.r_[x_r] # + K @ (K_obs.T @ ref) # 最適ゲインによる状態フィードバック と 目標値の入力
 
         # 目標速度と速度推定値と現在のPWM信号duty比を用いた, 目標速度へ追従するためのduty比の計算 ------       
        
        ki = 0.8 # ki + kf == 1.0?
        ko = 0.8
        kf = 0.2 # ko + kb == 1.0?
        kb = 0.2
        #p = kf*u - kb*x_hat[2] + (ko*x_hat[2] - ki*p)

        
        err_2 = err_1.copy()
        err_1 = err.copy()
        err = u - x_hat[2]
        err_I += err
        
        S = 1.0
        Kp = 0.2 * S # PIDパラメータ 現場調整必須
        Ki = 0.0015 * S # PIDパラメータ 現場調整必須
        Kd = 0.002 * S # PIDパラメータ 現場調整必須
        dp = Kp*(err - err_1) + Ki*err + Kd*((err - err_1) - (err_1 - err_2))
        p += dp
        
         # ------------------------------------------------------------------------------
        
        
         # モータの速度追従の時定数に基づいた, 速度の推定入力応答値の計算 ---------------------------
        theta = A_theta @ theta + B_theta @ u
        #theta = theta + dtheta * dt
        z = C_theta @ theta
         # ------------------------------------------------------------------------------
        
        
        #dx_history = np.r_[dx_history, dx.T]
        x_history = np.r_[x_history, x.T]
        y_history = np.r_[y_history, y.T]
        u_history = np.r_[u_history, u.T]
        z_history = np.r_[z_history, z.T]
        p_history = np.r_[p_history, p.T]
        
        #dx_hat_history = np.r_[dx_hat_history, dx_hat.T]
        x_hat_history = np.r_[x_hat_history, x_hat.T]
        y_hat_history = np.r_[y_hat_history, y_hat.T]
        
        t_history.append(time)
        time += dt
    
    
    plt.figure()
    plt.subplot(311)
    plt.plot(t_history, u_history, label=u"$\ u $ [$rad/s$]")
    #plt.plot(t_history, z_history, label=u"$\ z $ [$rad/s$]")
    #plt.plot(t_history, p_history, label=u"$\ p $ [$rad/s$]")
    plt.ylabel(u"$u(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.subplot(312)
    plt.plot(t_history, x_history[:, 0], label=u"$\ \dot{θ_1} $ [$rad/s$]")
    plt.plot(t_history, x_history[:, 1], label=u"$\ θ_1 $ [$rad$]")
    plt.plot(t_history, x_history[:, 2], label=u"$\ \dot{θ_2} $ [$rad/s$]")
    plt.plot(t_history, x_history[:, 3], label=u"$\ θ_2 $ [$rad$]")
    #plt.plot(t_history, x_history[:, 4], label=u"$\ \dot{eθ_2} $ [$rad/s$]")
    #plt.xlabel(u"$t [sec]$", fontsize=16)
    plt.ylabel(u"$x(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.subplot(313)
    plt.plot(t_history, x_hat_history[:, 0], label=u"$\ \dot{θ_1} $ [$rad/s$]")
    plt.plot(t_history, x_hat_history[:, 1], label=u"$\ θ_1 $ [$rad$]")
    plt.plot(t_history, x_hat_history[:, 2], label=u"$\ \dot{θ_2} $ [$rad/s$]")
    plt.plot(t_history, x_hat_history[:, 3], label=u"$\ θ_2 $ [$rad$]")
    #plt.plot(t_history, x_hat_history[:, 4], label=u"$\ \dot{eθ_2} $ [$rad/s$]")
    plt.xlabel(u"$t [sec]$", fontsize=16)
    plt.ylabel(u"$ \hat{x}(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()
    
    
    plt.figure()
    plt.title('pole position')
    plt.xlabel('real')
    plt.ylabel('imag')
    plt.scatter(ec.real, ec.imag, label='closed-loop-system')
    plt.scatter(ec_obs.real, ec_obs.imag, label='observer')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()
    
    
    """
    from scipy import signal as sig
    fs = 1./dt
    wp = 0.5*4 / (fs/2) # [Hz] までパス
    ws = 1.0*4 / (fs/2) # [Hz] からストップ
    gpass = 1
    gstop = 3
    b, a = sig.iirdesign(wp, ws, gpass, gstop, analog=False, ftype='butter', output='ba')
    print("a:", a)
    print("b:", b)
    
    import pandas as pd
    #df = pd.read_csv('./step_response_1of4_input.csv')
    #df = pd.read_csv('./step_response_half_input.csv')
    #df = pd.read_csv('./step_response_3of4_input.csv')
    df = pd.read_csv('./step_response.csv')

    dd = df.copy()
    fs = 1./dt
    n_win = 8
    n_point = int(1 * fs) # n[秒]分のデータを読み込む
    k_charge = 0.75
    flag_charged = False
    
    response = np.convolve(-(dd.values[1:n_point+1, 1] - dd.values[:n_point, 1]), np.ones(n_win)/n_win, 'valid') / dt
    charged = response[int(np.round(len(response)*k_charge)):].mean()
    
    tt = np.zeros([n_point])
    response = np.zeros([n_point])
    for k in range(3, n_point):
        
        tt[k] = float(dd.values[k, 0])
        
        x_0 = -(3*dd.values[k, 1] - 4*dd.values[k-1, 1] + dd.values[k-2, 1]) / (2*dt)
        x_1 = -(3*dd.values[k-1, 1] - 4*dd.values[k-2, 1] + dd.values[k-3, 1]) / (2*dt)
        #x_0 = -(dd.values[k, 1] - dd.values[k-1, 1]) / dt
        #x_1 = -(dd.values[k-1, 1] - dd.values[k-2, 1]) / dt
        
        response[k] = b[0]*x_0 + b[1]*x_1 - a[1]*response[k-1]
        #response[k] = x_0
        if response[k] / charged * 100. > 63 and response[k-1] / charged * 100. <= 63 and flag_charged == False:
            print('t = %f [s] is 63 [percent] charging time!!' %(float(dd.values[k, 0])))
            T_ = float(dd.values[k, 0])
            K_ = charged
            
            k_ = k
            
            aa = (3*response[k] - 4*response[k-1] + response[k-2]) / (2*dt)
            #aa = (response[k] - response[k-1]) / (dt)
            #y = aa * t + bb
            #bb = y - aa * t
            bb = response[k] - aa * T_
            
            L_ = (0 - bb) / aa
            
            print('L_ = %f, T_ = %f, K_ = %f' %(L_, T_, K_))
            
            flag_charged = True
    
    
    width = 4
    S_xy = np.cov(tt[k_-width:k_+width], response[k_-width:k_+width])
    Sx_2 = np.var(tt[k_-width:k_+width])
    aa_ = S_xy[0, 1] / Sx_2
    bb_ = response[k_-width:k_+width].mean() - aa_ * tt[k_-width:k_+width].mean()
    L__ = (0 - bb_) / aa_
    
    print('L__ = %f' %(L__))
    
    
    Kp = [0.6*T_/(K_*L_), 0.95*T_/(K_*L_)]
    Ki = [0.6/(K_*L_), 0.7/(K_*L_)]
    Kd = [0.3*T_/K_, 0.45*T_*K_]
    
    print("Kp = %f ~ %f, Ki = %f ~ %f, Kd = %f ~ %f" %(Kp[0], Kp[1], Ki[0], Ki[1], Kd[0], Kd[1]))
    
    plt.figure()
    plt.title('Step response of DC-motor velocity [rad/s]')
    plt.xlabel('time [s]')
    plt.ylabel('DC-motor velocity [rad/s]')
    plt.plot(dd.values[1+n_point-len(response):n_point+1, 0], response)
    plt.grid(True)
    plt.show() 
    
    len(response)
    
    
    plt.figure()
    plt.xlabel('time [s]')
    plt.ylabel('DC-motor velocity percent [%]')
    plt.title('Step response of DC-motor velocity percent [%]')
    plt.plot(dd.values[1+n_point-len(response):n_point+1, 0], response / charged * 100.)
    plt.grid(True)
    plt.show()
    """
    
    
    """
    import pandas as pd
    #df = pd.read_csv('./step_response_PID_u_05.csv')
    #df = pd.read_csv('./step_response_PID_u_10.csv')
    df = pd.read_csv('./step_response_PID_u_15.csv')
    
    for k in range(n_point):
        if (df.values[k, 2]+df.values[k, 3])/2 >= df.values[0, 1] * 0.63:
            print('t = %f [s] is 63 [percent] charging time!!' %(float(df.values[k, 0])))
            break
    
    plt.figure()
    plt.plot(df.values[:n_point, 0], (df.values[:n_point, 2]+df.values[:n_point, 3])/2)
    plt.grid(True)
    plt.show()
    """
    
    # 入力分散値計測 -------------------------------------------------------------
    import pandas as pd
    arr_white_noise = pd.read_csv('./for_measure_input_white_noise.csv').values
    w_gyro = np.var(arr_white_noise[:, 1], ddof=1)
    w_velo = np.var(arr_white_noise[:, 2], ddof=1)
    t_ = arr_white_noise[:, 0]
    ip = np.poly1d(np.polyfit(t_, arr_white_noise[:, 3], 1))(t_) # Interpolation polynomial
    w_posi = np.var(arr_white_noise[:, 3] - ip, ddof=1)
    print('variance of inputs: %f, %f, %f' %(w_gyro, w_velo, w_posi))
    # -------------------------------------------------------------------------
    
    
    
    import pandas as pd
    step_response = pd.read_csv('./step_response_from_duty_to_velo_on_114_7.csv').values
    tt = step_response[:, 0]
    inputs = step_response[:, 1]
    outputs = step_response[:, 2]
    
    plt.figure()
    
    plt.subplot(311)
    plt.plot(tt, inputs)
    plt.ylabel('duty')
    plt.grid()
    
    plt.subplot(312)
    plt.plot(tt, outputs)
    plt.ylabel('posi [rad]')
    plt.grid()
    
    
    plt.figure()
    plt.plot(tt[150:300], outputs[150:300])
    a_hat, b_hat = np.polyfit(tt[200:400]-1, outputs[200:400], 1)
    Tm_hat = - b_hat / a_hat
    Km_hat = a_hat
    print('Tm_hat: %f, Km_hat: %f' %(Tm_hat, Km_hat))
    
    fitted_outputs = np.poly1d(np.polyfit(tt[200:400], outputs[200:400], 1))(tt)
    plt.plot(tt[150:300], fitted_outputs[150:300])
    plt.grid()
    plt.show()
    
    from scipy import signal as sig
    fs = 1./dt
    wp = 2 / (fs/2) # [Hz] までパス
    ws = 4 / (fs/2) # [Hz] からストップ
    gpass = 1
    gstop = 3
    b1, a1 = sig.iirdesign(wp, ws, gpass, gstop, analog=False, ftype='butter', output='ba')

    y = np.zeros([outputs.shape[0]-2])
    y_pre = 0.0
    for i, y_ in enumerate(np.convolve(outputs, np.array([1, 0, 0, 0, -1])/(4*dt), mode='valid')):
        y[i] = b1[0]*y_ + b1[1]*y_pre - a1[1]*y[i-1]
        y_pre = y_
        #y[i] = y_
    
    plt.figure()
    plt.plot(tt[2:], y[:len(y)])
    TT = 0.10
    T_shift = 1.05
    
    
    t_ = np.poly1d(np.polyfit(np.arange(len(tt)), tt, 1))(np.arange(len(tt[2:])))
    z_ = (Km_hat+1)*(1-np.exp(-t_ / (Tm_hat+0.01)) )
    z_1d = np.poly1d(np.polyfit(t_[5:10], z_[5:10], 1))(t_)

    
    plt.plot(t_ + T_shift, z_)
    plt.plot(t_ + T_shift, z_1d)
    plt.xlabel('time [s]')
    plt.ylabel('velo [rad/s]')
    plt.xlim(0.9, 2.0)
    plt.ylim(-0.2, 17)
    plt.grid()
    
    plt.show()
    
    T_waste = t_[np.min(np.where(z_1d>=0))] + T_shift - 1.0
    Km_hat_velo = y[250:300].mean()
    Tm_hat_velo = t_[np.min(np.where(z_1d>=Km_hat_velo))] + T_shift - 1.0 - T_waste
    print('wasting time: %06f [sec]' %(T_waste))
    print('Km_hat_velo: %06f' %(Km_hat_velo))
    print('Tm_hat_velo: %06f [sec]' %(Tm_hat_velo))
    
    
    
    
    Tm_aim = 0.20
    Ki = 1. / (Km_hat * Tm_aim)
    Kp = Tm_hat / (Km_hat * Tm_aim)
    ka = Tm_hat
    kb = 1. + Kp * Km_hat
    kc = Ki * Km_hat
    
    import sympy as smp
    s = smp.var('s')
    eq = smp.Eq(ka * s**2 + kb * s + kc)
    ans = smp.solve(eq)
    print('poles in closed loop system:', ans)
    
    

    A_drive = np.array([[0., 1.],
                        [-Ki*Km_hat/Tm_hat, -(1+Kp*Km_hat)/Tm_hat]])
    B_drive = np.array([[0.],
                        [1.]])
    C_drive = np.array([[Ki*Km_hat/Tm_hat, Kp*Km_hat/Tm_hat]])
    D_drive = np.zeros([1, 1])
    sysd_drive = ss(A_drive, B_drive, C_drive, D_drive).sample(Ts=dt, method='zoh', alpha=None)
    A_drive, B_drive, C_drive, D_drive = sysd_drive.A, sysd_drive.B, sysd_drive.C, sysd_drive.D
    
    A_e = np.array([[0., 1., 0.],
                    [-Ki*Km_hat/Tm_hat, -(1+Kp*Km_hat)/Tm_hat, 0.],
                    [-Ki*Km_hat/Tm_hat, -Kp*Km_hat/Tm_hat, 0.]])
    B_e = np.array([[0.],
                    [1.],
                    [0.]])
    C_e = np.eye(len(A_e))
    D_e = np.zeros([C_e.shape[0], B_e.shape[1]])
    
    print(check_ctrb(A_e, B_e), check_obsv(A_e, C_e))
    
    sysd = ss(A_e, B_e, C_e, D_e).sample(Ts=dt, method='zoh', alpha=None)
    A_e, B_e, C_e, D_e = sysd.A, sysd.B, sysd.C, sysd.D
    
    
    R_ = np.array([[200.]]) # 入力の大きさに対するペナルティ
    Q_ = np.diag([1.0, 1.0, 1.0]) # 各変数の重要度
    
    R_obs_ = np.diag([[1.]])
    Q_obs_ = np.diag([1000., 1000.]) # 分散値の指定
    
    # カルマンフィルタ型 オブザーバシステム行列の定義 --
    K_obs_, P_obs_, e_obs_ = _dlqr(A_drive.T, C_drive.T, Q_obs_, R_obs_)
    A_obs_drive = A_drive - K_obs_.T @ C_drive
    B_obs_drive = np.c_[B_drive, K_obs_.T]
    C_obs_drive = np.eye(4, 4)
    
    # 最適レギュレータの設計
    K_, P_, e_ = _dlqr(A_e, B_e, Q_, R_)
        

        
    
    Km_hat += 0. # モデル化誤差入力
    Tm_hat += 0.0 # モデル化誤差入力
    
    A_drive = np.array([[0., 1.],
                        [-Ki*Km_hat/Tm_hat, -(1+Kp*Km_hat)/Tm_hat]])
    B_drive = np.array([[0.],
                        [1.]])
    C_drive = np.array([[Ki*Km_hat/Tm_hat, Kp*Km_hat/Tm_hat]])
    D_drive = np.zeros([1, 1])
    sysd_drive = ss(A_drive, B_drive, C_drive, D_drive).sample(Ts=dt, method='zoh', alpha=None)
    A_drive, B_drive, C_drive, D_drive = sysd_drive.A, sysd_drive.B, sysd_drive.C, sysd_drive.D
    
    
    
    ttime = 0.
    x_drive = np.zeros([2, 1])
    x_hat_drive = np.zeros([2, 1])
    aim = 1. # 目標値の設定
    r_drive = np.array([[aim]])
    x_history_drive = np.zeros([2, 1]).T
    y_history_drive = np.zeros([1, 1])
    t_history_drive = np.zeros([1])
    x_hat_history_drive = np.zeros([2, 1]).T
    y_hat_history_drive = np.zeros([1, 1])
    x_r_drive = np.zeros([1, 1])
    u_drive = np.zeros([1, 1])
    
    flag = False
    simttime = 1.
    while(ttime < simttime):
        
        x_drive = A_drive @ x_drive + B_drive @ (r_drive)
        y_drive = C_drive @ x_drive
        
        x_hat_drive = A_obs_drive @ x_hat_drive + B_obs_drive @ np.r_[r_drive, y_drive] # 推定状態
        y_hat_drive = C_drive @ x_hat_drive # 推定状態の観測
        # --------------------------------------------------------------------------------
    
        x_r_drive += r_drive - y_drive # 目標値追従のため, 出力と目標値との差を積分
    
        #u_drive = - K_[:, :2] @ x_hat_[:2] - K_[:, 2:] @ np.r_[x_r_] # 最適ゲインによる状態フィードバック と 目標値の入力
        #u_drive = - K_[:, :2] @ x_hat_[:2] # 最適ゲインによる状態フィードバック と 目標値の入力
        #u_drive = - K_[:, 2:] @ np.r_[x_r_] # 最適ゲインによる状態フィードバック と 目標値の入力
 
        
        t_history_drive = np.r_[t_history_drive, ttime]
        x_history_drive = np.r_[x_history_drive, x_drive.T]
        y_history_drive = np.r_[y_history_drive, y_drive.T]
        
        if y_drive > aim * 0.632 and flag == False:
            Tm_sim = ttime
            flag = True
            
        if ttime > simttime/2:
            r_drive[0, 0] = 0.0
    
        ttime += dt
    
    plt.figure()
    plt.plot(t_history_drive[:], y_history_drive[:, 0])
    plt.xlabel('time [s]')
    plt.ylabel('velo [rad/s]')
    plt.grid()
    plt.show()
    
    print('charged time: %f' %Tm_sim)
    
    
    # 外乱オブザーバのテスト -------------------------------------------------------
    dt_velo = 0.0005
    
    Kp = 0.06 # dt_velo=0.0005 # PID
    Ki = 0.050 # dt_velo=0.0005
    Kd = 0.0

    #Kp = 0.04 # dt_velo=0.006 # PID
    #Ki = 1.00 # dt_velo=0.006
    
    #Kp = 0.030 # dt_velo=0.006 # I-PD
    #Ki = 1.00 # dt_velo=0.006

    #Kp = 0.12 # dt_velo=0.0005 # I-PD
    #Ki = 0.11 # dt_velo=0.0005
    #Kd = 0.000
    
    f = 0.0
    K = 10.0
    
    #sys_P = P_ss(dt=dt_velo, Km=14.194703 +1, Tm=0.104867)
    sys_Pn = Pn_ss(dt=dt_velo, Km=10, Tm=0.20)
    sys_dobs = FofPn_ss(dt=dt_velo, omega_d=1000, Km=10, Tm=0.20)
    sys_P = P_ss_2nd(dt=dt_velo, Km=15.131289, Tm=0.139214, tau=0.053046)
    #sys_Pn = P_ss_2nd(dt=dt_velo, Km=10, Tm=0.20, tau=0.00)
    #sys_dobs = FofPn_ss_2nd(dt=dt_velo, omega_d=1000, Km=10, Tm=0.20, tau=0.00)
    
    
    x_d = np.zeros([len(sys_dobs.A), 1])
    y_d = [0.]
    x_P = np.zeros([len(sys_P.A), 1])
    y_P = np.array([[0.]])
    duty = np.zeros([1, 1])
    x_Pn = np.zeros([len(sys_Pn.A), 1])
    y_Pn = [0.]
    omega_star = np.array([[10.]])
    
    y_temp = np.array([1, 1])

    err = np.array([0., 0., 0.])
    err_I = 0.
    
    x_d_log = np.zeros([0, len(x_d)])
    y_d_log = np.zeros([0, 1])
    x_P_log = np.zeros([0, len(x_P)])
    y_P_log = np.zeros([0, 1])
    duty_log = np.zeros([0, 1])
    x_Pn_log = np.zeros([0, len(x_Pn)])
    y_Pn_log = np.zeros([0, 1])
    omega_star_log = np.zeros([0, 1])
    time_velo_log = np.zeros([0, 1])

    
    flag = False
    Tm_controled = 0.
    
    time_velo = np.array([[0.]])
    time_stack = np.array([[0.]])

    y_P_prev = 0.0
    
    while time_velo <= 3.0:
        
        x_P = sys_P.A @ x_P + sys_P.B @ duty
        y_P = sys_P.C @ x_P + sys_P.D @ duty
        #y_P *= 1.0
        
        #if time_stack >= 0.006:
        #    y_P = y_temp #+ 0.2*np.random.randn(1, 1)
        #    time_stack *= 0.0
        
        x_Pn = sys_Pn.A @ x_Pn + sys_Pn.B @ duty
        y_Pn = sys_Pn.C @ x_Pn + sys_Pn.D @ duty
        
        x_d = sys_dobs.A @ x_d + sys_dobs.B @ (y_Pn - y_P)
        y_d = sys_dobs.C @ x_d + sys_dobs.D @ (y_Pn - y_P)
        
        omega_star = K * np.cos(2*np.pi*f*time_velo)
        
        err[2] = err[1]
        err[1] = err[0]
        err[0] = (omega_star - y_P)
        err_I += err[0] * dt
        err_I *= 0.99999
        
        #dd = Kp*(err[0]-err[1]) + Ki*err[0] + Kd*(err[0]-2*err[1]+err[2])
        #duty += dd
        duty = Kp*(err[0]) + Ki*err_I + Kd*(err[0]-err[1])/dt # PID制御器
        #duty = - Kp*y_P + Ki*err_I - Kd*(y_P-y_P_prev)/dt# I-PD制御器
        y_P_prev = y_P.copy()

        duty += y_d
        #duty = np.round(duty)
        duty = np.clip(duty, -1, 1)
        
        if y_P >= omega_star*0.63 and flag == False:
            Tm_controled = time_velo.copy()
            flag = True
            #x_P += -10.0
            
        
        x_d_log = np.r_[x_d_log, x_d.T]
        y_d_log = np.r_[y_d_log, y_d.T]
        x_P_log = np.r_[x_P_log, x_P.T]
        y_P_log = np.r_[y_P_log, y_P.T]
        duty_log = np.r_[duty_log, duty.T]
        x_Pn_log = np.r_[x_Pn_log, x_Pn.T]
        y_Pn_log = np.r_[y_Pn_log, y_Pn.T]
        omega_star_log = np.r_[omega_star_log, omega_star.T]
        time_velo_log  = np.r_[time_velo_log , time_velo.T]
        
        time_velo += dt_velo
        time_stack += dt_velo
        
    
    plt.figure()
    plt.plot(time_velo_log, y_P_log, label="actual")
    plt.plot(time_velo_log, y_Pn_log, label="nominal")
    plt.plot(time_velo_log, y_d_log, label="dobs")
    plt.plot(time_velo_log, duty_log, label="duty")
    plt.plot(time_velo_log, omega_star_log, label="aim")
    
    plt.xlabel("time [s]")
    plt.ylabel("velocity [rad/s]")
    
    plt.legend()
    plt.grid()
    
    plt.show()
    
    print("Tm_controled: %f" %Tm_controled)
    # -------------------------------------------------------------------------
    
    
    
    """
    # PWM duty - 回転数[rad/s] のグラフ点 から2次関数で fitting
    xx = [0.25, 0.50, 0.75, 1.00]
    yy = [2.54, 9.79, 12.20, 13.45]
    a_, b_, c_ = np.polyfit(xx, yy, 2)
    duty = np.linspace(0, 1, 1000)
    
    plt.figure()
    plt.plot(duty, np.poly1d(np.polyfit(xx, yy, 2))(duty), label='d=2')
    plt.legend()
    plt.grid(True)
    plt.title('PWM duty - DC motor velocity [rad/s]')
    plt.xlabel('PWM duty')
    plt.ylabel('DC-motor velocity [rad/s]')
    plt.show()
    """



    """
    plt.figure()
    plt.subplot(211)
    plt.plot(t_history, u_history, color="blue", label=u"$\ u $ [$N*m$]")
    plt.ylabel(u"$u(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.subplot(212)
    plt.plot(t_history, x_hat_history[:, 0], label=u"$\ \dot{θ_1} $ [$rad/s$]")
    plt.plot(t_history, x_hat_history[:, 1], label=u"$\ θ_1 $ [$rad$]")
    plt.plot(t_history, x_hat_history[:, 2], label=u"$\ \dot{θ_2} $ [$rad/s$]")
    plt.xlabel(u"$t [sec]$", fontsize=16)
    plt.ylabel(u"$ \hat{x}(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()
    """
    
    