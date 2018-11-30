# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:12:23 2017

@author: wattai
"""

from control.matlab import ctrb, obsv, place
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

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
    
def main():
    
    # 物理パラメータの定義 -------------
    M = 5
    m = 0.4
    l = 0.3
    D_theta = 0.001
    D_x = 0.002
    g = 9.80665
    
    N = (4.*M+m)*l/3.
    # ----------------------------
    
    # システム行列の定義 --------------
    A = np.array([[0., 0., 1., 0.],
                  [0., 0., 0., 1.],
                  [0., -m*g*l/N, -4*l*D_x/(3*N), D_theta/N],
                  [0., (M+m)*g/N, D_x/N, -(M+m)*D_theta/(N*m*l)]
                 ])
    B = np.array([[0.],
                  [0.],
                  [4.*l/(3.*N)],
                  [-1./N]
                 ])
    #C = np.eye(2, 4)
    C = np.array([[0., 0., 0., 1.]])
    D = np.array([[0]])
    # -----------------------------

    # 最適レギュレータ重みの定義 --------------
    R = np.array([[1]])
    Q = np.diag([0.0, 0.0, 0.0, 1.0])
    #Q = np.diag([1.0, 10000.0, 1.0])
    #Q = np.sqrt(C.T @ C)
    # -----------------------------------
    
    # オブザーバシステム行列の定義 ------------
    p_obs = np.array([-10, -20, -10, -10]) # 2次元配列にするとplaceでエラー出る
    #p_obs = np.array([-30, -25, -20]) # 2次元配列にするとplaceでエラー出る
    K_obs = place(A.T, C.T, p_obs)
    
    A_obs = A - K_obs.T @ C
    B_obs = np.c_[B, K_obs.T]
    C_obs = np.eye(4, 4)
    # -----------------------------------
    
    
    """
    A = np.array([[1.1, 2.0],
                  [0.0, 0.95]])
    B = np.array([[0],
                  [0.0787]])
    C = np.array([[-2, 1]])
    D = np.array([[0]])
    #Q = np.diag([5, 3])
    Q = C.T @ C
    R = np.array([[1]])
    """
    
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
    
    print("p_obs: ")
    print(p_obs)
    print("K_obs: ")
    print(K_obs)
    
    # システムが可制御・可観測でなければ終了
    if check_ctrb(A, B) == -1 :
        print("システムが可制御でないので終了")
        return 0
    #if check_obsv(A, C) == -1 :
    #    print("システムが可観測でないので終了")
    #    return 0
    # 最適レギュレータの設計
    K, P, e = _clqr(A, B, Q, R)
    # 結果表示
    print("リカッチ方程式の解:\n",P)
    print("状態フィードバックゲイン:\n",K)
    print("閉ループ系の固有値:\n",e)
    
    dt = 0.001
    simtime = 10
    time = 0.0
    
    x = np.array([[0.0],
                  [0.1],
                  [0.0],
                  [0.0]])

    u = np.zeros([B.shape[1]])
    y = np.zeros([C.shape[0], 1])

    t_history = []
    dx_history = np.zeros([0, len(x)])
    x_history = np.zeros([0, len(x)])
    y_history = np.zeros([0, len(y)])
    u_history = np.zeros([0, len(u)])

        
    x_ = np.array([[0.0],
                   [0.0],
                   [0.0],
                   [0.0]])

    y_ = np.zeros([C.shape[0], 1])
    dx__history = np.zeros([0, len(x)])
    x_history = np.zeros([0, len(x)])
    #y__history = np.zeros([0, len(y_)])
    #u__history = np.zeros([0, len(u_)])

    
    while(time <= simtime):
        
        # プラント側の計算 ------------------
        u = - K @ x_ # 最適ゲインによる状態フィードバック
        
        dx = A @ x + B @ u # 状態微分
        
        x = x + dx * dt #+ 0.01*np.random.randn(4, 1) # 状態遷移(オイラー積分)
        y = C @ x #+ D @ u # 状態観測
        # -------------------------------

        # オブザーバ側の計算 ----------------
        dx_ = A @ x_ + B @ u + K_obs.T @ (y - y_)

        x_ = x_ + dx_ * dt
        y_ = C @ x_
        # -------------------------------
        
        dx_history = np.r_[dx_history, dx.T]
        x_history = np.r_[x_history, x.T]
        y_history = np.r_[y_history, y.T]
        u_history = np.r_[u_history, u.T]
        
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
    plt.plot(t_history, x_history[:, 0], label=u"$\ x $")
    plt.plot(t_history, x_history[:, 1], label=u"$\ θ $")
    plt.plot(t_history, x_history[:, 2], label=u"$\ \.{x} $")
    plt.plot(t_history, x_history[:, 3], label=u"$\ \.{θ} $")
    plt.xlabel(u"$t [sec]$", fontsize=16)
    plt.ylabel(u"$x(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

        
if __name__ == "__main__":
    main()