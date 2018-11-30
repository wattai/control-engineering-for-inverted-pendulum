# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:04:59 2017

@author: wattai
"""

from control.matlab import ctrb, obsv, place
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

    
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

    # H inf 制御器 -----------------
    A_k = np.array([[-7.1991655, -9.414875], #gopt +1 by ccontrq 2*2
                      [-289.15216, -171.56731]])
    B_k = np.array([[-0.3740521],
                      [-171.96013]])
    C_k = np.array([[-0.0069560, -118.76725]])
    D_k = np.array([[4.3767059]])
    
    A_k = np.array([[0.0015839, -0.3679068], # gopt +1 by hinf 2*2
                      [-14.668575, -15.397435]])
    B_k = np.array([[1.3679068],
                      [6.5968456]])
    C_k = np.array([[224.38719, 34.140884]])
    D_k = np.array([[4.3767059]])
    
    A_k = np.array([[0.0016984, -0.3680428], # gopt +0 by hinf 2*2
                      [-16.195784, -15.630349]])
    B_k = np.array([[1.3680428],
                      [6.5974618]])
    C_k = np.array([[231.07185, 35.157585]])
    D_k = np.array([[4.3767059]])
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
    # -----------------------------------
    
    # 最適レギュレータ重みの定義 ---------------
    R = np.array([[10000]]) # 入力の大きさに対するペナルティ
    Q = np.diag([100.0, 1.0, 1.0, 1.0]) # 各変数の重要度
    # -----------------------------------
    

    dt = 0.001
    simtime = 10
    time = 0.0

    
    print("A: ")
    print(A)
    print("B: ")
    print(B)
    print("C: ")
    print(C)
    print("D: ")
    print(D)    

    x = np.array([[ 0.0 ],
                  [ 0.0 ],
                  [ 0.3 ],
                  [ 0.0 ]])

    print("x0: ")
    print(x)
    
    u = np.zeros([B.shape[1], 1])
    y = np.zeros([C.shape[0], 1])
    
    t_history = []
    dx_history = np.zeros([0, len(x)])
    x_history = np.zeros([0, len(x)])
    y_history = np.zeros([0, len(y)])
    u_history = np.zeros([0, len(u)])

    
    x_k = np.zeros([len(B_k), 1])
    

    y_k = np.zeros([C.shape[0], 1])
    dx_k_history = np.zeros([0, len(B_k)])
    x_k_history = np.zeros([0, len(B_k)])
    y_k_history = np.zeros([0, len(y_k)])

    
    while(time <= simtime):
        
        # プラント側の計算 ------------------
        w = 0*np.random.randn(x.shape[0], 1) #　状態外乱の生成
        dx = (A @ x) + (B @ u) + w # 状態微分
        x = x + dx * dt # 状態遷移(オイラー積分)
        v = 0*np.random.randn(y.shape[0], 1) # 観測ノイズ生成
        y = C @ x + v #+ D @ u # 状態観測
        # -------------------------------

        # コントローラ側の計算 ----------------
        dx_k = A_k @ x_k + B_k @ y # 推定状態の微分
        rho1 = np.array([[0.0],
                         [0.0],
                         [0.0],
                         [0.0]]) # 目標値の設定
        rho2 = np.array([[0.0],
                         [0.2],
                         [0.0],
                         [0.0]]) # 目標値の設定
        
        if time > 5.: rho = rho2.copy()
        else: rho = rho1.copy()
        x_k = x_k + dx_k * dt # 状態推定
        u = C_k @ x_k + D_k @ y # 推定状態の観測
        # -------------------------------
        
        dx_history = np.r_[dx_history, dx.T]
        x_history = np.r_[x_history, x.T]
        y_history = np.r_[y_history, y.T]
        u_history = np.r_[u_history, u.T]
        
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
    plt.plot(t_history, r * x_history[:, 0], label=u"$\ x $")
    plt.plot(t_history, r * x_history[:, 1], label=u"$\ \dot{x} $")
    plt.plot(t_history, x_history[:, 2], label=u"$\ \phi $")
    plt.plot(t_history, x_history[:, 3], label=u"$\ \dot{\phi} $")
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
    plt.plot(t_history, r * x_k_history[:, 0], label=u"$\ x $")
    plt.plot(t_history, r * x_k_history[:, 1], label=u"$\ \dot{x} $")
    plt.plot(t_history, x_k_history[:, 2], label=u"$\ \phi $")
    plt.plot(t_history, x_k_history[:, 3], label=u"$\ \dot{\phi} $")
    plt.xlabel(u"$t [sec]$", fontsize=16)
    plt.ylabel(u"$ \hat{x}(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    carwidth = r#0.30         # 台車の幅
    carheight = r#0.20        # 台車の高さ
    pendulumwidth = 0.02    # 振子の幅
    pendulumheight = 2*l    # 振子の長さ
    video(r*x_history[:, 0], x_history[:, 2],
          carwidth, carheight, pendulumwidth, pendulumheight, dt)
    
    
import matplotlib as mpl
import matplotlib.animation as animation
def video(y, theta,
          carwidth, carheight, pendulumwidth, pendulumheight, tstep):

    theta = -theta
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111,
                         xlim=(-1.0-pendulumheight,
                               +1.0+pendulumheight),
                         ylim=(-0.0-pendulumheight,
                               +2.0+pendulumheight))
    ax.set_xlabel(u"$x [m]$", fontsize=16)
    ax.grid()

    #car = plt.Rectangle((y[0] - carwidth/2, 0),
    #                     carwidth, carheight, fill=False)
    car = plt.Circle((y[0], carwidth), carwidth, fill=False)
    
    pendulum = plt.Rectangle((y[0] - pendulumwidth/2, carheight),
                              pendulumwidth, pendulumheight, fill=False)
    ts0 = ax.transData
    coords0 = [y[0], carheight/2]
    tr0 = mpl.transforms.Affine2D().rotate_deg_around(coords0[0],
                                coords0[1], 180*theta[0]/np.pi) + ts0
    pendulum.set_transform(tr0)
    
    time_text = ax.text(0.02, 0.95, 'aaaaaa', transform=ax.transAxes)
    
    def init():
        ax.add_patch(car)
        ax.add_patch(pendulum)
        time_text.set_text('initial state')
        return car, pendulum, time_text

    def anime(i):
        ts = ax.transData
        coords = [y[i], carheight]
        tr = mpl.transforms.Affine2D().rotate_deg_around(coords[0],
                                    coords[1], 180*theta[i]/np.pi) + ts

        #car.set_x(y[i] - carwidth/2)
        car.center = (y[i], carwidth)
        pendulum.set_xy((y[i] - pendulumwidth/2, carheight))
        pendulum.set_transform(tr)

        ax.add_patch(car)
        ax.add_patch(pendulum)
        time_text.set_text('time = {0:.2f}'.format(i*tstep))
        return car, pendulum, time_text

    ani = animation.FuncAnimation(fig, anime, np.arange(1, len(y)),
                                  interval=tstep*1.0e+3, blit=True,
                                  init_func=init)
    
    #ani.save('pydulum_x264_robust.mp4',
    #         fps=1/tstep,
    #         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    #ani.save('pydulum_x265_guruguru.mp4',
    #         fps=1/tstep,
    #         extra_args=['-vcodec', 'libx265', '-pix_fmt', 'yuv420p'])
    
    plt.show()
        
if __name__ == "__main__":
    main()