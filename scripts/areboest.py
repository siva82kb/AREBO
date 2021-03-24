"""Module containing a set of functions for estimating human limb parameters
with AREBO.

Author: Sivakumar Balasubramanian
Date: 25 July 2018
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools
import itertools
import functools
import pandas as pd
from scipy.signal import savgol_filter

# From my libraries
from myrobotics import rotx, roty, rotz, HTMat, HTMat4DH
from arebo import arebo_inv_kinematics
from arebo import a1, a2, a3, b1, b2 , b3

def run_calibration():
    # Set human joint angles.
    tv = np.linspace(-90, 0, 20) * np.pi / 180.0
    th = np.array([0])  # np.linspace(-30, 60, 20)
    tr = np.array([0])

    calib = pd.DataFrame(columns=["th", "tv", "tr", "ta1",
                                  "ta2", "ta3", "ta4",
                                  "a1", "a2", "a3", "b1", "b2", "b3"])
    for _tv, _th, _tr in itertools.product(tv, th, tr):
        # Perform forward kinematics of the arm
        # t_h = [np.pi * _t / 180. for _t in [_tv, _th, _tr]]
        t_h = [_tv, _th, _tr]
        
        # Human forward kinematics
        # Human HT matices 
        # Find Homogenrous transformation matrices.
        _Hh = [HTMat4DH(_t + _dh['t'], _dh['d'], _dh['a'], _dh['al'])
              for _t, _dh in zip(t_h, human_dh)]
        _Hhuman = functools.reduce(np.dot, _Hh)
        _Xhuman = _Hhuman[:3, 3] + sh_pos
        _Rhuman = _Hhuman[:3, :3]

        # AREBO position inverse kinematics
        ta = arebo_inv_kinematics(Xh=_Xhuman, Rh=_Rhuman, dh=arebo_dh)
        _datapoint = {"th": [_th], "tv": [_tv], "tr":[_tr],
                      "ta1": [ta[0]], "ta2": [ta[1]],
                      "ta3": [ta[2]], "ta4": [ta[3]],
                      "a1": [a1(_tv, _th)],
                      "a2": [a2(_tv, _th)],
                      "a3": [a3(_tv, _th)],
                      "b1": [b1(ta[0], ta[1], ta[2], ta[3], a, b, c)],
                      "b2": [b2(ta[0], ta[1], ta[2], ta[3], a, b, c)],
                      "b3": [b3(ta[0], ta[1], ta[2], ta[3], a, b, c)],
                    }
        calib = calib.append(pd.DataFrame.from_dict(_datapoint), ignore_index=True)


def normalize(x, rmin, rmax):
    return (rmax - rmin) * ((x - np.min(x)) / (np.max(x) - np.min(x))) + rmin


def generate_calib_data(tv, th, tr, sh_pos, uh, arebo_dh, human_dh, noise=0.):
    calib = pd.DataFrame(columns=["th", "tv", "tr", "ta1",
                                  "ta2", "ta3", "ta4", "ta5", "ta6",
                                  "a1", "a2", "a3", "b1", "b2", "b3",
                                  "shx", "shy", "shz", "l"])
    N = len(tv)
    for _n in range(N):
        # Human limb angles
        _tv, _th, _tr = tv[_n], th[_n], tr[_n]

        # Perform forward kinematics of the arm
        # t_h = [np.pi * _t / 180. for _t in [_tv, _th, _tr]]
        t_h = [_tv, _th, _tr]
        
        # Human forward kinematics
        # Human HT matices 
        # Find Homogenrous transformation matrices.
        _Hh = [HTMat4DH(_t + _dh['t'], _dh['d'], _dh['a'], _dh['al'])
              for _t, _dh in zip(t_h, human_dh)]
        _Hhuman = functools.reduce(np.dot, _Hh)
        _Xhuman = _Hhuman[:3, 3] + sh_pos[:, _n]
        _Rhuman = _Hhuman[:3, :3]

        # AREBO position inverse kinematics
        ta = arebo_inv_kinematics(Xh=_Xhuman, Rh=_Rhuman, dh=arebo_dh)
        # Add noise
        _noise = np.random.normal(0, noise, 9)
        # Update angles
        _th += _noise[0]
        _tv += _noise[1]
        _tr += _noise[2]
        ta[0] += _noise[3]
        ta[1] += _noise[4]
        ta[2] += _noise[5]
        ta[3] += _noise[6]
        ta[4] += _noise[2]
        ta[5] += _noise[8]
        _datapoint = {"th": [_th], "tv": [_tv], "tr": [_tr],
                      "ta1": [ta[0]], "ta2": [ta[1]], "ta3": [ta[2]],
                      "ta4": [ta[3]], "ta5": [ta[4]], "ta6": [ta[5]],
                      "a1": [a1(_tv, _th)],
                      "a2": [a2(_tv, _th)],
                      "a3": [a3(_tv, _th)],
                      "b1": [b1(ta[0], ta[1], ta[2], ta[3],
                                arebo_dh[1]['a'], arebo_dh[2]['a'],
                                arebo_dh[4]['d'])],
                      "b2": [b2(ta[0], ta[1], ta[2], ta[3],
                                arebo_dh[1]['a'], arebo_dh[2]['a'],
                                arebo_dh[4]['d'])],
                      "b3": [b3(ta[0], ta[1], ta[2], ta[3],
                                arebo_dh[1]['a'], arebo_dh[2]['a'],
                                arebo_dh[4]['d'])],
                      "shx": [sh_pos[0, _n]],
                      "shy": [sh_pos[1, _n]],
                      "shz": [sh_pos[2, _n]],
                      "l": [uh]
                      }
        calib = calib.append(pd.DataFrame.from_dict(_datapoint), ignore_index=True)
    return calib


def estimate_limb_param(calib, N, M):
    _A = np.array(calib.loc[:, ["a1", "a2", "a3"]])
    _B = np.array(calib.loc[:, ["b1", "b2", "b3"]])

    # Parameter estimate
    p_hat = np.zeros((4, N - M + 1))
    for _n in range(N - M + 1):
        _nnan = np.sum(np.isnan(_B[_n:_n + M])) / 3
        L = int(M - _nnan)

        # Create A and b matrices
        A = np.zeros((3 * L, 4))
        B = np.zeros((3 * L, 1))
        i = 0
        for _a, _b in zip(_A[_n:_n + M, :], _B[_n:_n + M]):
            if np.sum(np.isnan(_b)) == 0:
                A[3 * i:3 * i + 3, 0] = _a
                A[3 * i:3 * i + 3, 1:] = np.eye(3)
                B[3 * i:3 * i + 3,  0] = _b
                i += 1

        # Solve Ax = b
        p_hat[:, _n] = np.matmul(np.linalg.pinv(A), B).T[0]
    
    return p_hat


def gen_polysine(t, f=[0.1, 0.2, 0.5], a=[1.0, 0.5, 0.1]):
    phi = np.random.uniform(0, 2 * np.pi, size=(4,1))
    x = np.zeros(len(t))
    for i in range(len(f)):
        x += a[i] * np.sin(2 * np.pi * f[i] * t + phi[i])
    return x


def get_random_humanparams(shpos_range, l_range, Np):
    # Generate random human parameter sets.
    sh_posx = np.random.uniform(shpos_range[0, 0], shpos_range[0, 1], Np)
    sh_posy = np.random.uniform(shpos_range[1, 0], shpos_range[1, 1], Np)
    sh_posz = np.random.uniform(shpos_range[2, 0], shpos_range[2, 1], Np)
    sh_pos = np.array([sh_posx, sh_posy, sh_posz])
    l = np.random.uniform(l_range[0], l_range[1], Np)
    return sh_pos, l


# Generate calibration movement sequences
def get_subject_calib_angles(t, pvrange, phrange):
    tv = (np.pi / 180.) * normalize(gen_polysine(t, f=[0.2, 0.5, 1.0],
                                                 a=[1.0, 0.5, 0.1]),
                                    pvrange[0], pvrange[1])
    th = (np.pi / 180.) * normalize(gen_polysine(t, f=[0.2, 0.5, 1.0],
                                                 a=[1.0, 0.5, 0.1]),
                                    phrange[0], phrange[1])
    tr = np.zeros(len(t))
    return np.array([tv, th, tr])


def get_all_params(hparam, ang_calib, angnoise):
    _N = (len(hparam), len(ang_calib), len(angnoise))
    for v1 in enumerate(hparam):
        for v2 in enumerate(ang_calib):
            for v3 in enumerate(angnoise):
                yield v1, v2, v3, _N
