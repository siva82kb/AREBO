"""Module containing robotics related functions and classes.

Author: Sivakumar Balasubramanian
Date: 24 May 2018
"""

import numpy as np
import functools

def rotx(t):
    ct, st = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  ct, -st],
                     [0.0,  st,  ct]])


def roty(t):
    ct, st = np.cos(t), np.sin(t)
    return np.array([[ ct, 0.0,  st],
                     [0.0, 1.0, 0.0],
                     [-st, 0.0,  ct]])


def rotz(t):
    ct, st = np.cos(t), np.sin(t)
    return np.array([[ ct, -st, 0.0],
                     [ st,  ct, 0.0],
                     [0.0, 0.0, 1.0]])


def HTMat(R, d):
    _R = np.hstack((R, d))
    return np.vstack((_R, np.array([[0, 0, 0, 1]])))


def HTMat4DH(t, d, a, al):
    _Hx = HTMat(rotz(t), np.zeros((3, 1)))
    _Hd = HTMat(np.eye(3), np.array([[0, 0, d]]).T)
    _Ha = HTMat(np.eye(3), np.array([[a, 0, 0]]).T)
    _Hal = HTMat(rotx(al), np.zeros((3, 1)))
    return functools.reduce(np.dot, [_Hx, _Hd, _Ha, _Hal])


def forward_kinematics(dhparam, t):
    """Returns the location and orientation of the different framess with 
    respect to the base frame the given configuation.
    """
    _H = [HTMat4DH(_t + _dh['t'], _dh['d'], _dh['a'], _dh['al'])
          for _t, _dh in zip(t, dhparam)]
    H = [_H[0]]
    for i in range(1, len(_H)):
        H.append(np.matmul(H[i-1], _H[i]))
    return H, _H