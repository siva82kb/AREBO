"""Module containing a set of function for simulating AREBO's
kinematics.

Author: Sivakumar Balasubramanian
Date: 09 July 2018
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from ipywidgets import interactive
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import widgets
from ipywidgets import interactive
from mpl_toolkits.mplot3d import Axes3D
import itertools
import functools

# From my libraries
from myrobotics import rotx, roty, rotz, HTMat, HTMat4DH
from myrobotics import forward_kinematics

def sincos(t):
    return np.sin(t), np.cos(t)


def arebo_inv_kinematics(Xh, Rh, dh):
    # Inverse Kinematics
    # Step 1. Find the orientation of the plane in the robot.
    ta1 = np.arctan2(Xh[1] , Xh[0])
    
    # Step 2. Find new position for the 3-link planar arm.
    _s1, _c1 = sincos(ta1)
    if np.isclose(_s1, 0.0):
        xd = Xh[0] / _c1
    elif np.isclose(_c1, 0.0):
        xd = Xh[1] / _s1
    else:
        xd = 0.5 * ((Xh[0] / _c1) + (Xh[1] / _s1))
    yd = Xh[2]
    
    # Step 3.
    _R1 = np.array([[_c1, 0, _s1],
                    [_s1, 0, -_c1],
                    [0, 1, 0]])
    _Rh = np.matmul(_R1.T, Rh)

    # Determine t2 + t3 + t4
    _ta4 = np.arctan2(_Rh[0, 2], -_Rh[1, 2])
    
    # Determine t6
    ta6 = np.arctan2(_Rh[2, 0], _Rh[2, 1])

    # Determine t5
    _s4t, _c4t = sincos(_ta4)
    _s6, _c6 = sincos(ta6)
    
    # Find _r1 and _r2
    # _r1
    if np.isclose(_c4t, 0.0):
        _r1 = _Rh[0, 2] / _s4t
    elif np.isclose(_s4t, 0.0):
        _r1 = - _Rh[1, 2] / _c4t
    else:
        _r1 = 0.5 * (_Rh[0, 2] / _s4t - _Rh[1, 2] / _c4t)
        
    # _r2
    if np.isclose(_c6, 0.0):
        _r2 = _Rh[2, 0] / _s6
    elif np.isclose(_s6, 0.0):
        _r2 = _Rh[2, 1] / _c6
    else:
        _r2 = 0.5 * ((_Rh[2, 0] / _s6) + (_Rh[2, 1] / _c6))
    
    ta5 = np.arctan2(-_Rh[2, 2], 0.5 * (_r1 + _r2))
    _s5, _c5 = sincos(ta5)
    
    # Convert to 2-link planar arm problem
    a, b, c = dh[1]['a'], dh[2]['a'], dh[4]['d']
    _xd = xd - c * _c4t
    _yd = yd - c * _s4t    
    _n = _xd**2 + _yd**2 - a**2 - b**2
    _d = 2 * a * b
    ta3 = np.arccos(_n / _d)

    alpha = np.arctan2(_yd, _xd)

    _n = _xd**2 + _yd**2 + a**2 - b**2
    _d = 2 * a * np.sqrt(_xd**2 + _yd**2)
    beta = np.arccos(_n / _d)

    ta2 = alpha - beta
    ta4 = _ta4 - ta2 - ta3

    return np.array([ta1, ta2, ta3, ta4, ta5, ta6])


def a1(tv, th):
    return np.cos(tv) * np.cos(th)


def a2(tv, th):
    return np.sin(tv) * np.cos(th)


def a3(tv, th):
    return np.sin(th)


def b1(ta1, ta2, ta3, ta4, a, b, c):
    _temp = (a * np.cos(ta2) +
             b * np.cos(ta2 + ta3) +
             c * np.cos(ta2 + ta3 + ta4))
    return np.cos(ta1) * _temp


def b2(ta1, ta2, ta3, ta4, a, b, c):
    _temp = (a * np.cos(ta2) +
             b * np.cos(ta2 + ta3) +
             c * np.cos(ta2 + ta3 + ta4))
    return np.sin(ta1) * _temp


def b3(ta1, ta2, ta3, ta4, a, b, c):
    _temp = (a * np.sin(ta2) +
             b * np.sin(ta2 + ta3) +
             c * np.sin(ta2 + ta3 + ta4))
    return _temp


def plot_link_rot(ax, o0, H, color='k', frame=False):
    o = H[:, 3]
    x = H[:, 0] + o
    y = H[:, 1] + o
    z = H[:, 2] + o
    
    # Plot the link
    ax.plot([o0[2], o[2]], [o0[1], o[1]], [-o0[0], -o[0]], color=color, lw=2)
    # Plot the link frame.
    if frame:
        ax.plot([o[2], x[2]], [o[1], x[1]], [-o[0], -x[0]], c=[1, 0, 0, 0.5],
                lw=4.0)
        ax.plot([o[2], z[2]], [o[1], z[1]], [-o[0], -z[0]], c=[0, 0, 1, 0.5],
                lw=4.0)


def show_arebo_human(dh_arebo, t_a, dh_human, t_h, sh_pos):
    # Find Homogenrous transformation matrices.
    _Ha = [HTMat4DH(_t + _dh['t'], _dh['d'], _dh['a'], _dh['al'])
           for _t, _dh in zip(t_a, dh_arebo)]
    _Harebo = functools.reduce(np.dot, _Ha)
    _Xarebo = _Harebo[:3, 3]
    _Rarebo = _Harebo[:3, :3]
    
    _Hh = [HTMat4DH(_t + _dh['t'], _dh['d'], _dh['a'], _dh['al'])
           for _t, _dh in zip(t_h, dh_human)]
    _Hhuman = functools.reduce(np.dot, _Hh)
    _Xhuman = _Hhuman[:3, 3] + sh_pos
    _Rhuman = _Hhuman[:3, :3]
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    L = 14
    ax.plot([-L, L], [0, 0], [0, 0], '0.5', lw=0.4)
    ax.plot([0, 0], [-L, L], [0, 0], '0.5', lw=0.4)
    ax.plot([0, 0], [0, 0], [-L, L], '0.5', lw=0.4)
    
    # Plot AREBO links and frames
    # Link and Frame 0
    _Hc = np.eye(4)
    plot_link_rot(ax, [0, 0, 0], _Hc, color='0.6', frame=True)
    # Link and Frame 1
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Ha[0])
    plot_link_rot(ax, _o, _Hc, color='0.6', frame=False)
    # Link and Frame 2
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Ha[1])
    plot_link_rot(ax, _o, _Hc, color='0.6', frame=False)
    # Link and Frame 3
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Ha[2])
    plot_link_rot(ax, _o, _Hc, color='0.6', frame=False)
    # Link and Frame 4
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Ha[3])
    plot_link_rot(ax, _o, _Hc, color='0.6', frame=False)
    # Link and Frame 5
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Ha[4])
    plot_link_rot(ax, _o, _Hc, color='0.6', frame=False)
    # Link and Frame 6
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Ha[5])
    plot_link_rot(ax, _o, _Hc, color='0.6', frame=True)

    # Plot HUMAN links and frames
    # Link and Frame 1
    _Hc = np.eye(4)
    _Hc[0, 3] = sh_pos[0]
    _Hc[1, 3] = sh_pos[1]
    _Hc[2, 3] = sh_pos[2]
    plot_link_rot(ax, sh_pos, _Hc, frame=True)
    # Link and Frame 2
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Hh[0])
    plot_link_rot(ax, _o, _Hc, frame=False)
    # Link and Frame 3
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Hh[1])
    plot_link_rot(ax, _o, _Hc, frame=False)
    # Link and Frame 4
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Hh[2])
    plot_link_rot(ax, _o, _Hc, frame=True)
    
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    print("-")
    print("Position Error: ", np.round(np.linalg.norm(_Xarebo - _Xhuman), 6))
    print("Orientation error: ", np.round(np.linalg.norm(_Rhuman - _Rarebo), 6)   )
    print("AREBO Angles: ", np.round(t_a * (180.0 / np.pi), 1))


def show_arebo(dh_arebo, t_a):
    # Find Homogenrous transformation matrices.
    _Ha = [HTMat4DH(_t + _dh['t'], _dh['d'], _dh['a'], _dh['al'])
           for _t, _dh in zip(t_a, dh_arebo)]
    _Harebo = functools.reduce(np.dot, _Ha)
    _Xarebo = _Harebo[:3, 3]
    _Rarebo = _Harebo[:3, :3]
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    L = 14
    ax.plot([-L, L], [0, 0], [0, 0], '0.5', lw=0.4)
    ax.plot([0, 0], [-L, L], [0, 0], '0.5', lw=0.4)
    ax.plot([0, 0], [0, 0], [-L, L], '0.5', lw=0.4)
    
    # Plot AREBO links and frames
    # Link and Frame 0
    _Hc = np.eye(4)
    plot_link_rot(ax, [0, 0, 0], _Hc, color='0.6', frame=True)
    # Link and Frame 1
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Ha[0])
    plot_link_rot(ax, _o, _Hc, color='0.6', frame=True)
    # Link and Frame 2
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Ha[1])
    plot_link_rot(ax, _o, _Hc, color='0.6', frame=True)
    # Link and Frame 3
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Ha[2])
    plot_link_rot(ax, _o, _Hc, color='0.6', frame=True)
    # Link and Frame 4
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Ha[3])
    plot_link_rot(ax, _o, _Hc, color='0.6', frame=True)
    # Link and Frame 5
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Ha[4])
    plot_link_rot(ax, _o, _Hc, color='0.6', frame=True)
    # Link and Frame 6
    _o = _Hc[:, 3]
    _Hc = np.matmul(_Hc, _Ha[5])
    plot_link_rot(ax, _o, _Hc, color='0.6', frame=True)

    
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
