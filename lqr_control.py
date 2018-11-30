# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:36:48 2017

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
    
    M = 5
    m = 0.4
    l = 0.1
    D_theta = 0.001
    D_x = 0.002
    g = 9.80665
    
    N = (4.*M+m)*l/3.
        
    # システム行列の定義
    A = np.array([[0., 0., 1., 0.],
                  [0., 0., 0., 1.],
                  [0., -m*g*l/N, -4*l*D_x/(3*N), D_theta/N],
                  [0., (M+m)*g/N, D_x/N, -(M+m)*D_theta/(N*m*l)],
                 ])
    B = np.array([[0.],
                  [0.],
                  [4.*l/(3.*N)],
                  [-1./N]
                 ])
    #C = np.eye(2, 4)
    C = np.array([[0, 0, 0, 1]])
    D = np.array([[0]])
    
    Q = np.diag([100, 1.0, 1.0, 1.0])
    #Q = np.sqrt(C.T @ C)
    R = np.array([[1]])
    
    
    #A = A[1:, 1:]
    #B = B[1:, :]
    #C = np.array([[0, 0, 1]])
    #Q = np.diag([1.0, 10.0, 1.0])
    
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
    """
    # システムが可制御・可観測でなければ終了
    if check_ctrb(A, B) == -1 :
        print("システムが可制御でないので終了")
        return 0
    if check_obsv(A, C) == -1 :
        print("システムが可観測でないので終了")
        return 0
    """
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
    
    u = np.zeros([B.shape[1], 1])
    y = np.zeros([C.shape[0], 1])

    t_history = []
    dx_history = np.zeros([0, len(x)])
    x_history = np.zeros([0, len(x)])
    y_history = np.zeros([0, len(y)])
    u_history = np.zeros([0, len(u)])
    
    while(time <= simtime):
        
        u = - K @ x # 最適ゲインによる状態フィードバック
        
        dx = A @ x + B @ u
        
        x = x + dx * dt # 状態遷移(オイラー積分)
        y = C @ x + D @ u # 状態観測
        
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.subplot(212)
    plt.plot(t_history, x_history[:, 0], label=u"$\ x $")
    plt.plot(t_history, x_history[:, 2], label=u"$\ \.{x} $")
    plt.plot(t_history, x_history[:, 1], label=u"$\ θ $")
    plt.plot(t_history, x_history[:, 3], label=u"$\ \.{θ} $")
    plt.xlabel(u"$t [sec]$", fontsize=16)
    plt.ylabel(u"$x(t)$", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()
    
    
    carwidth = 0.30         # 台車の幅
    carheight = 0.20        # 台車の高さ
    pendulumwidth = 0.04    # 振子の幅
    pendulumheight = 2*l    # 振子の長さ
    video(x_history[:, 0], x_history[:, 1],
          carwidth, carheight, pendulumwidth, pendulumheight, dt)
    
    
import matplotlib as mpl
import matplotlib.animation as animation
def video(y, theta,
          carwidth, carheight, pendulumwidth, pendulumheight, tstep):

    theta = -theta
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111,
                         xlim=(np.min(y)-pendulumheight,
                               np.max(y)+pendulumheight),
                         ylim=(np.min(y)-pendulumheight,
                               np.max(y)+pendulumheight))
    ax.set_xlabel(u"$x [m]$", fontsize=16)
    ax.grid()

    car = plt.Rectangle((y[0] - carwidth/2, 0),
                         carwidth, carheight, fill=False)

    pendulum = plt.Rectangle((y[0] - pendulumwidth/2, carheight/2),
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
        coords = [y[i], carheight/2]
        tr = mpl.transforms.Affine2D().rotate_deg_around(coords[0],
                                    coords[1], 180*theta[i]/np.pi) + ts

        car.set_x(y[i] - carwidth/2)
        pendulum.set_xy((y[i] - pendulumwidth/2, carheight/2))
        pendulum.set_transform(tr)

        ax.add_patch(car)
        ax.add_patch(pendulum)
        time_text.set_text('time = {0:.2f}'.format(i*tstep))
        return car, pendulum, time_text

    ani = animation.FuncAnimation(fig, anime, np.arange(1, len(y)),
                                  interval=tstep*1.0e+3, blit=True,
                                  init_func=init)
    
    #ani.save('pydulum_x264.mp4',
    #         fps=1/tstep,
    #         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    #ani.save('pydulum_x265_guruguru.mp4',
    #         fps=1/tstep,
    #         extra_args=['-vcodec', 'libx265', '-pix_fmt', 'yuv420p'])
    
    plt.show()

    
if __name__ == "__main__":
    main()