# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:22:40 2017

@author: wattai
"""

import numpy as np

def invpen_moments(mb_, mw):

    # 本体(body)の慣性モーメントの計測 -------------------------------------------------
    # 正面から見て糸が平行になった時間系列 (半周期毎の時間) [s]
    measures_b = [14.8, 15.4, 16.0, 16.6, 17.2, 17.8, 18.4, 19.0,
                  19.6, 20.1, 20.7, 21.3, 21.9, 22.5, 22.9, 23.6,
                  24.1, 24.7, 25.3, 25.8, 26.3, 26.9, 27.5, 27.9, 28.5]
    
    Tb = np.convolve(measures_b, [2, -2], mode='valid').mean() # 本体の振動周期 [s]
    fb = 1./Tb # 本体の振動周波数 [Hz]
    
    g = 9.80665 # 重力加速度 [m/s^2]
    ab = 4.8 * 1e-2 # 重心より @左側@ の糸と結合している箇所までの距離 [m] (重心から見て車軸側)
    bb = 5.7 * 1e-2 # 重心より @右側@ の糸と結合している箇所までの距離 [m] (重心から見て頭側)
    mb  = mb_ + mw # 293 * 1e-3# 本体(タイヤ含む)の質量[kg]
    lb = 24 * 1e-2 # 本体計測時の糸の長さ[m]
    Jb = mb * g * ab * bb / ( 4 * np.pi**2 * fb**2 * lb )
   # ---------------------------------------------------------------------------    
   
   # タイヤ(wheel)の慣性モーメントの計測 ------------------------------------------------
    # 正面から見て糸が平行になった時間系列 (半周期毎の時間) [s]
    measures_w = [11.8, 12.2, 12.7, 13.1, 13.5, 14.0, 14.4, 14.8,
                  15.2, 15.7, 16.1, 16.5, 16.9, 17.4, 17.8, 18.2,
                  18.6, 19.0, 19.4, 19.8, 20.2, 20.6, 21.0, 21.4, 21.8]
    
    Tw = np.convolve(measures_w, [2, -2], mode='valid').mean() # タイヤ1個での振動周期 [s]
    fw = 1./Tw # タイヤ1個での振動周波数 [Hz]
    
    g = 9.80665 # 重力加速度 [m/s^2]
    rw = 2.75 * 1e-2 # タイヤの半径 [m]
    aw = rw # 重心より @左側@ の糸と結合している箇所までの距離 [m]
    bw = rw # 重心より @右側@ の糸と結合している箇所までの距離 [m]
    #mw = 7 * 1e-3 * 2 # タイヤの質量 [kg] (2個分)
    lw = 32 * 1e-2 # タイヤ計測時の糸の長さ [m]
    Jw = mw * g * aw * bw / ( 4 * np.pi**2 * fw**2 * lw ) # 求める慣性モーメント [kg*m^2]
   # ---------------------------------------------------------------------------

    return Jb, Jw    
    

if __name__ == '__main__':

    Jb, Jw = invpen_moments()
    
    print('本体の慣性モーメント:', Jb, '[kg*m^2]')
    print('タイヤ2個分まとめた慣性モーメント [kg*m^2]:', Jw, '[kg*m^2]')
    