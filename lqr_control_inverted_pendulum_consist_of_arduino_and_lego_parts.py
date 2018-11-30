# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 00:10:55 2017

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
    
if __name__ == "__main__":
    
    """
    Arduino と LEGO部品 で構成される倒立振子に対して，
    x_dot = A @ x + B @ u
    y = C @ x
    で表現される状態方程式を基に，最適レギュレータにて制御を行う。
    
    状態変数の内訳は，
    x = [ theta, dtheta, phi, dphi ]
      =　[ モータ回転角 [rad], モータ回転角速度[rad/sec], 車体角度[rad], 車体角速度[rad/sec] ]
    
    ジャイロセンサのみの場合，システムが可観測ではないので，
    カルマンフィルタを用いたオブザーバの設計は不可能(極配置は可能だが，任意には置けない)
    """
    
    # 物理パラメータの定義 -------------
    l = 0.08 #+ 0.01 * np.random.randn() # 車軸と車体重心間の距離 [m]
    r = 0.027 #+ 0.01 * np.random.randn() # 車輪半径 [m]
    M = 0.279 #+ 0.10 * np.random.randn() # 車体質量 [kg]
    m = 0.014 #+ 0.01 * np.random.randn() # 車輪質量 [kg]
    J_b = 1.46e-3 #+ 0.001 * np.random.randn() # 車体慣性モーメント [kg * m^2]
    J_w = 1.13e-4 #+ 0.001 * np.random.randn() # 車輪慣性モーメント [kg * m^2]
    g = 9.80665 # 重力加速度 [m / sec^2]
    n = 114.7 # ギア比
    
    c_1 = m*(r**2) + M*(r+l)**2 + J_w + J_b
    c_2 = J_w + (M+m)*(r**2) + M*l*r
    # ----------------------------
    
    # システム行列の定義 --------------
    A = np.array([[0., 1., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 1.],
                  [0., 0., M*g*l/c_1, 0.]
                 ])
    B = np.array([[0.],
                  [1.],
                  [0.],
                  [-c_2/c_1]
                 ])
    #C = np.eye(2, 4)
    C = np.array([[0., 0., 0., 1.]])
    D = np.array([[0]])
    # -----------------------------

    # システムが可制御・可観測でなければ終了
    if check_ctrb(A, B) == -1 :
        print("システムが可制御でないので終了")
        #return 0
    if check_obsv(A, C) == -1 :
        print("システムが可観測でないので終了")
        #return 0
    
    # 最適レギュレータ重みの定義 --------------
    R = np.array([[1]]) # 入力の大きさに対するペナルティ
    Q = np.diag([1.0, 1.0, 1.0, 1.0]) # 各変数の重要度
    #Q = np.diag([1.0, 10000.0, 1.0])
    #Q = np.sqrt(C.T @ C)
    # -----------------------------------
    
  
    # 最適レギュレータの設計
    K, P, e = _clqr(A, B, Q, R)
    
    # 結果表示
    print("リカッチ方程式の解:\n", P)
    print("状態フィードバックゲイン:\n", K)
    print("閉ループ系の固有値:\n", e)
    
    dt = 0.015
    simtime = 10
    time = 0.0
    
    # オブザーバシステム行列の定義 ------------
    p_obs = np.array([-17.1+0j, -17.3-0j, -17.0+0j, -17.5-0j]) # 2次元配列にするとplaceでエラー出る
    #p_obs = 1 * e.real # 2次元配列にするとplaceでエラー出る
    K_obs = place(A.T, C.T, p_obs)
        
    A_obs = A - K_obs.T @ C
    B_obs = np.c_[B, K_obs.T]
    C_obs = np.eye(4, 4)
    
    print("A_obs:\n", A_obs)
    print("B_obs:\n", B_obs)
    eval_obs, evec_obs = la.eig(A_obs)
    print("オブザーバの固有値:\n", eval_obs)
    print("オブザーバの状態フィードバックゲイン:\n", K_obs)
    # -----------------------------------
    
    """
    # カルマンフィルタ型 オブザーバの重みの定義 ------
    #V = np.array([[0.3, 0.5, 0.002, 0.1]])
    V = np.eye(C.shape[0])
    Bn = np.array([[1.0, 0.7, 2.0, 1.4]]).T
    # -----------------------------------
    
    # カルマンフィルタ型 オブザーバシステム行列の定義 --
    print((Bn @ Bn.T).shape)
    K_obs, P_obs, e_obs = _clqr(A.T, C.T, Bn @ Bn.T, V)
    # -----------------------------------
    
    # 結果表示 ---------------------------
    print("リカッチ方程式の解:\n", P_obs)
    print("状態フィードバックゲイン:\n", K_obs)
    print("閉ループ系の固有値:\n", e_obs)
    # -----------------------------------
    """
    
    # 物理パラメータの定義 -------------
    l = 0.08 #+ 0.01 * np.random.randn() # 車軸と車体重心間の距離 [m]
    r = 0.027 #+ 0.01 * np.random.randn() # 車輪半径 [m]
    M = 0.279 #+ 0.10 * np.random.randn() # 車体質量 [kg]
    m = 0.014 #+ 0.01 * np.random.randn() # 車輪質量 [kg]
    J_b = 1.46e-3 #+ 0.001 * np.random.randn() # 車体慣性モーメント [kg * m^2]
    J_w = 1.13e-4 #+ 0.001 * np.random.randn() # 車輪慣性モーメント [kg * m^2]
    g = 9.80665 # 重力加速度 [m / sec^2]
    
    c_1 = m*(r**2) + M*(r+l)**2 + J_w + J_b
    c_2 = J_w + (M+m)*(r**2) + M*l*r
    # ----------------------------
    
    # システム行列の定義 --------------
    A = np.array([[0., 1., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 1.],
                  [0., 0., M*g*l/c_1, 0.]
                 ])
    B = np.array([[0.],
                  [1.],
                  [0.],
                  [-c_2/c_1]
                 ])
    #C = np.eye(2, 4)
    C = np.array([[0., 0., 0., 1.]])
    D = np.array([[0]])
    # -----------------------------   
    
    
    
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
                  [ 0.0 * np.random.randn() ],
                  [ 5 / 180 * np.pi + 0.0 * np.random.randn() ],
                  [ 0.0 * np.random.randn() ]])
    print("x0: ")
    print(x)
    
    u = np.zeros([B.shape[1], 1])
    y = np.zeros([C.shape[0], 1])
    ui = np.zeros([B.shape[1], 1])
    uc = np.zeros([B.shape[1], 1])
    uc_ = np.zeros([B.shape[1], 1])
    uc_low = np.zeros([B.shape[1], 1])
    u_hat = np.zeros([B.shape[1], 1])
    
    t_history = []
    dx_history = np.zeros([0, len(x)])
    x_history = np.zeros([0, len(x)])
    y_history = np.zeros([1, len(y)])
    u_history = np.zeros([0, len(u)])
    ui_history = np.zeros([0, len(ui)])
    uc_history = np.zeros([0, len(u)])
    power_history = np.zeros([0, len(u)])
    u_hat_history = np.zeros([0, len(u)])

    
    x_hat = np.zeros([x.shape[0], 1])

    y_hat = np.zeros([C.shape[0], 1])
    dx_hat_history = np.zeros([0, len(x)])
    x_hat_history = np.zeros([0, len(x)])
    y_hat_history = np.zeros([0, len(y_hat)])

    from scipy import signal as sig
    fs = 1./dt
    wp = 0.1 / (fs/2) # 0.4[Hz] からパス
    ws = 0.05 / (fs/2) # 0.2[Hz] までストップ
    gpass = 1
    gstop = 3
    b, a = sig.iirdesign(wp, ws, gpass, gstop, analog=False, ftype='butter', output='ba')
    y_ = 0.0
    print("a:", a)
    print("b:", b)
    
    wp = 4 / (fs/2) # 0.4[Hz] からパス
    ws = 20 / (fs/2) # 0.2[Hz] までストップ
    gpass = 1.0
    gstop = 10
    d, c = sig.iirdesign(wp, ws, gpass, gstop, analog=False, ftype='butter', output='ba')
    print("c:", c)
    print("d:", d)
    
    y_err = 0.
    cnt = 0
    u_err = 0.
    pd = np.array([[0.]])
    x_hat1_prev1 = np.array([[0.]])
    dpd = 0.
    uc_ave = np.array([[0.]])
    power = 0.
    tau_err_prev1 = 0.0
    tau_err_prev2 = 0.0
    tau_i = 0.0
    u_hat = np.array([[0.]])
    u_hat_ = np.array([[0.]])
    y_prev = np.array([[0.]])
    
    while(time <= simtime):
        
        # プラント側の計算 ------------------
        u = - K @ x_hat # 最適ゲインによる状態フィードバック
        ui = ui + u * dt
        
        w = 0.1*np.random.randn(x.shape[0], 1) #　状態外乱の生成
        
        dx = A @ x + B @ u #+ w # 状態微分
        
        x = x + dx * dt # 状態遷移(オイラー積分)
        v = 0.1*np.random.randn(y.shape[0], 1) # 観測ノイズ生成
        dri = 0.03 # センサの ドリフト成分生成
        y = C @ x #+ v + dri #+ D @ u # 状態観測 (観測ノイズと ドリフト成分を 含む)
        # -------------------------------
        
        # High pass filter to avoid drift phenomenon. --------
        y_prev = y_
        y_ = b[0]*y + b[1]*y_history[-1] - a[1]*y_
        # ----------------------------------------------------

        
        # モーター制御のための下位制御層 ---------------------------
        uc = l/r*np.sin((y_ - y_prev)/dt) + dpd # 加えられている力の推定 1/2
        uc_low = (d[0]*uc + d[1]*uc_[-1] - c[1]*uc_low)*1.1 # 加えられている力の推定 2/2 (LPF)
        #print("u:", u, "uc:", uc)
        u_err_ = u_err
        u_err = u - uc # 指令値と現在加えられている力の差分を計算
        Kp = 2.6
        Kd = -0.9
        dpd = Kp*u_err + Kd*(u_err - u_err_) # PD制御の差分値
        pd += dpd
        
        uc_ = uc.copy()
        # -----------------------------------------------------
        
        
        K_tau = 3600.
        K_v = 9100./1.5
        tau_aim = r * (m+M) / n
        tau_now = ( - K_tau * 60./(2.*np.pi) * (x_hat[1]) + K_v * power ) * n * 1e-3
        # トルク差基準の評価式
        tau_err = tau_aim - tau_now
        # 電圧差基準の評価式
        #tau_err = tau_aim / K_v / n * 1e3 + K_tau / K_v * 60. / (2*np.pi) * x_hat[1] - power
        # 角加速度基準の評価式
        #u_hat = (x_hat[1] - x_hat1_prev1) / dt
        #tau_err = u - u_hat
                                                              
        K_p = 0.00005 # [トルク差]基準のPID制御定数
        K_i = 0.00001
        K_d = 0.00005
        
        #K_p = 0.01 # [電圧差]基準のPID制御定数
        #K_i = 0.001
        #K_d = 0.07
        
        #K_p = 1.0 # [推定角加速度差]基準のPID制御定数
        #K_i = 0.1
        #K_d = 0.1
        
        tau_i += tau_err * dt
        #tau_d = (tau_err - tau_err_prev1) / dt
        tau_d = (tau_err_prev2 - 4. * tau_err_prev1 + 3. * tau_err) / (2.*dt)
                
        dpower = K_p * tau_err + K_i * tau_i + K_d * tau_d
        power += dpower * dt
        tau_err_prev1 = tau_err
        tau_err_prev2 = tau_err_prev1
        #print(tau_err, tau_i, tau_d, power)
        
        power_history = np.r_[power_history, power.T]
        
        # ---------------------------------------------
        
        
        x_hat1_prev1[0, 0] = x_hat[1, 0].copy()
        # オブザーバ側の計算 ----------------
        #dx_hat = A @ x_hat + B @ u + K_obs.T @ (y - y_hat) # 推定状態の微分
        dx_hat = A_obs @ x_hat + B_obs @ np.r_[u, y_] # 推定状態の微分
        
        if time > 5.:
            rho = np.array([[0.0],
                          [ 0 * (r / np.pi**2)],
                          [0.0],
                          [0.0]]) # 目標値の設定
        if time > 10.:
            rho = np.zeros([4, 1])
        if time < 5.:
            rho = np.zeros([4, 1])

        x_hat = x_hat + dx_hat * dt #- rho # 状態推定
        y_hat = C @ x_hat # 推定状態の観測
        # -------------------------------
        

        

        
        
        
        
        dx_history = np.r_[dx_history, dx.T]
        x_history = np.r_[x_history, x.T]
        y_history = np.r_[y_history, y.T]
        u_history = np.r_[u_history, u.T]
        ui_history = np.r_[ui_history, ui.T]
        uc_history = np.r_[uc_history, uc_low.T]
        u_hat_history = np.r_[u_hat_history, u_hat.T]
        
        
        dx_hat_history = np.r_[dx_hat_history, dx_hat.T]
        x_hat_history = np.r_[x_hat_history, x_hat.T]
        y_hat_history = np.r_[y_hat_history, y_hat.T]
        
        t_history.append(time)
        time += dt
        cnt += 1
        
    
    plt.figure()
    plt.subplot(311)
    #plt.plot(t_history, u_history, color="blue", label=u"$\ u $ [$rad/s^2$]")
    plt.plot(t_history, power_history, color="green", label=u"$\ pow $ [$V$]")
    #plt.plot(t_history, u_hat_history, color="purple", label=u"$\ \hat{u} $ [$V$]")
    #plt.plot(t_history, uc_history, color="red", label=u"$\ u_c / 10 $ [$rad/s^2$]")
    #plt.plot(t_history, ui_history, color="green", label=u"$\ u_i $ [$rad/s$]")
    plt.ylabel(u"$u(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.subplot(312)
    plt.plot(t_history, x_history[:, 0] / (2*np.pi*r) / 100, label=u"$\ x $ [cm]")
    plt.plot(t_history, x_history[:, 1] / (2*np.pi*r) / 100, label=u"$\ \dot{x} $ [cm/s]")
    plt.plot(t_history, x_history[:, 2], label=u"$\ \phi $ [rad]")
    plt.plot(t_history, x_history[:, 3], label=u"$\ \dot{\phi} $ [rad/s]")
    #plt.xlabel(u"$t [sec]$", fontsize=16)
    plt.ylabel(u"$x(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.subplot(313)
    plt.plot(t_history, x_hat_history[:, 0] / (2*np.pi*r) / 100, label=u"$\ x $ [cm]")
    plt.plot(t_history, x_hat_history[:, 1] / (2*np.pi*r) / 100, label=u"$\ \dot{x} $ [cm/s]")
    plt.plot(t_history, x_hat_history[:, 2], label=u"$\ \phi $ [rad]")
    plt.plot(t_history, x_hat_history[:, 3], label=u"$\ \dot{\phi} $ [rad/s]")
    plt.xlabel(u"$t [sec]$", fontsize=16)
    plt.ylabel(u"$ \hat{x}(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()
    
    """
    plt.figure()
    plt.subplot(211)
    plt.plot(t_history, u_history, color="blue", label=u"$\ u $")
    plt.ylabel(u"$u(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.subplot(212)
    plt.plot(t_history, x_hat_history[:, 0] / (2*np.pi*r) / 100, label=u"$\ x $ [cm]")
    plt.plot(t_history, x_hat_history[:, 1] / (2*np.pi*r) / 100, label=u"$\ \dot{x} $ [cm/s]")
    plt.plot(t_history, x_hat_history[:, 2], label=u"$\ \phi $ [rad]")
    plt.plot(t_history, x_hat_history[:, 3], label=u"$\ \dot{\phi} $ [rad/s]")
    plt.xlabel(u"$t [sec]$", fontsize=16)
    plt.ylabel(u"$ \hat{x}(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()
    """
        
