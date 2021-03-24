"""Module containing functions for AREBO link lenght optimization.

Author: Sivakumar Balasubramanian
Date: 13 June 2020
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
from scipy.interpolate import interp2d

# From my libraries
from myrobotics import rotx, roty, rotz, HTMat, HTMat4DH
from myrobotics import forward_kinematics
from arebo import arebo_inv_kinematics
from arebo import a1, a2, a3, b1, b2 , b3
from arebo import show_arebo
from arebo import show_arebo_human
from arebo import sincos

# From my libraries
from myrobotics import rotx, roty, rotz, HTMat, HTMat4DH
from myrobotics import forward_kinematics

def get_arebo_dh(r1, r2, r3):
    """Returns the AREBO DH parameter dictionary for the given
    r1, r2, and r3."""
    return [{'a': 0,  'al': +np.pi/2, 'd': 0,  't': 0},
            {'a': r1, 'al': 0,        'd': 0,  't': 0},
            {'a': r2, 'al': 0,        'd': 0,  't': 0},
            {'a': 0,  'al': -np.pi/2, 'd': 0,  't': -np.pi/2},
            {'a': 0,  'al': +np.pi/2, 'd': r3, 't': +np.pi/2},
            {'a': 0,  'al': 0,        'd': 0,  't': +np.pi/2}]


def get_human_dh(uh):
    """Returns the Human limb DH parameter dictionary for the given
    uh."""
    return [{'a': 0, 'al': +np.pi/2, 'd': 0,  't': 0},
            {'a': 0, 'al': -np.pi/2, 'd': 0,  't': -np.pi/2},
            {'a': 0, 'al': 0,        'd': uh, 't': -np.pi}]


def iterate_over_hlimb_param(hlimb_param):
    # Go through each possible human parameter 
    # one after the other.
    # 1. Human shoulder position
    shpos = get_possible_position(hlimb_param['loc'])
    # 2. Possible human limb length
    uh = np.arange(hlimb_param['links']['uh'][0],
                  hlimb_param['links']['uh'][1] + hlimb_param['links']['dr'],
                  hlimb_param['links']['dr'])
    return [[_shp[0], _shp[1], _shp[2], _uh] for _shp in shpos for _uh in uh]


def iterate_over_robotparam(rparam):
    _r1range = get_from_range(rparam['links']['r1'][0],
                              rparam['links']['r1'][1],
                              rparam['links']['dr'])
    _r2range = get_from_range(rparam['links']['r2'][0],
                              rparam['links']['r2'][1],
                              rparam['links']['dr'])
    _r3range = get_from_range(rparam['links']['r3'][0],
                              rparam['links']['r3'][1],
                              rparam['links']['dr'])
    return [(_r1, _r2, _r3) for _r1 in _r1range
            for _r2 in _r2range for _r3 in _r3range]


def get_from_range(strt, stp, step):
    return np.arange(strt, stp + step, step)


def get_possible_position(locs):
    """Get the combinations of the different locations
    for the human limb."""
    for _x in locs['x']:
      for _y  in locs['y']:
        for _z in locs['z']:
          yield [_x, _y, _z]


def all_hlimb_angles(hparam):
    _pa1 = get_from_range(hparam['angles']['t1'][0],
                          hparam['angles']['t1'][1],
                          hparam['angles']['dt'])
    _pa2 = get_from_range(hparam['angles']['t2'][0],
                          hparam['angles']['t2'][1],
                          hparam['angles']['dt'])
    return [((i1, i2), (_p1, _p2, 0)) for i1, _p1 in enumerate(_pa1)
             for i2, _p2 in enumerate(_pa2)], _pa1, _pa2


def human_limb_forward_kinematics(p_h, human_dh, shpos):
    # Human forward kinematics
    # Human HT matices 
    # Find Homogenrous transformation matrices.
    _Hh = [HTMat4DH(_t + _dh['t'], _dh['d'], _dh['a'], _dh['al'])
           for _t, _dh in zip(p_h, human_dh)]
    _Hhuman = functools.reduce(np.dot, _Hh)
    _Xhuman = _Hhuman[:3, 3] + shpos
    _Rhuman = _Hhuman[:3, :3]

    return _Rhuman, _Xhuman


def arebo_forward_kinematics(t_a, arebo_dh):
    # Find Homogenrous transformation matrices.
    _Ha = [HTMat4DH(_t + _dh['t'], _dh['d'], _dh['a'], _dh['al'])
           for _t, _dh in zip(t_a, arebo_dh)]
    _Harebo = functools.reduce(np.dot, _Ha)
    _Xarebo = _Harebo[:3, 3]
    _Rarebo = _Harebo[:3, :3]
    return _Rarebo, _Xarebo, _Ha


def get_frame5_comps(HTMrob):
    _H5 = HTMrob[0] @ HTMrob[1] @ HTMrob[2] @ HTMrob[3] @ HTMrob[4]
    return _H5[:3, 0], _H5[:3, 1], _H5[:3, 2]


def get_frame3_jacobian(ta1, ta2, ta3, arebo_dh):
    s1, c1 = sincos(ta1)
    s2, c2 = sincos(ta2)
    s23, c23 = sincos(ta2 + ta3)

    r1, r2 = arebo_dh[1]['a'], arebo_dh[2]['a']
    # Components of the jacobian matrix
    j = np.zeros((3, 3))
    j[0, 0] = - (r1 * c2 + r2 * c23) * s1
    j[0, 1] = (r1 * c2 + r2 * c23) * c1
    j[0, 2] = 0
    j[1, 0] = - (r1 * s2 + r2 * s23) * c1
    j[1, 1] = - (r1 * s2 + r2 * s23) * s1
    j[1, 2] = r1 * c2 + r2 * c23
    j[2, 0] = - r2 * c1 * s23
    j[2, 1] = - r2 * s1 * s23
    j[2, 2] = r2 * c23

    return j


def wrap_angles(angles):
    return np.array([(a + np.pi) % (2 * np.pi) - np.pi for a in angles])


def is_reachable(rangs, robot_param):
    # Make sure none of the anges are nan.
    _isnotnan = (not np.any(np.isnan(rangs)))
    return _isnotnan and is_in_range(rangs, robot_param)


def is_in_range(rangs, robot_param):
    # Check if all the angles are within the allow range.
    return np.all([(_ra >= robot_param['angles'][f't{i+1}'][0] and
                    _ra <= robot_param['angles'][f't{i+1}'][1])
                   for i, _ra in enumerate(rangs)])


def get_invkinmaps(rparam, uh, shpos, all_hangles, Nha1, Nha2, robot_param):
    invkin_maps = {'reachable': np.zeros((Nha1, Nha2)),
                   'fratio': np.zeros((Nha1, Nha2)),
                   'rangles': np.zeros((Nha1, Nha2, 6))}
    # Human and robot DH parameters
    _hDH = get_human_dh(uh)
    _rDH = get_arebo_dh(*rparam)
    for _inx, _pang in all_hangles:
        # Human Forward Kinematics
        _Rh, _Xh = human_limb_forward_kinematics(_pang, _hDH, shpos)

        # Inverse Kinematics for the 
        _rangs = wrap_angles(arebo_inv_kinematics(_Xh, _Rh, _rDH))

        # Robot forward kinematics
        _, _, _Hr = arebo_forward_kinematics(_rangs, _rDH)

        # Frame3 Jacobian.
        _J3 = get_frame3_jacobian(_rangs[0], _rangs[1], _rangs[2], _rDH)
        
        # Frame5 components and their projection matrices.
        _x5, _y5, _z5 = get_frame5_comps(_Hr)
        _xy5 = np.array([_x5, _y5]).T
        _z5 = np.array([_z5]).T
        _Pxy5 = _xy5 @ _xy5.T
        _Pz5 = _z5 @ _z5.T

        # Update inverse kinematics maps.
        # 1. Reachablility map.
        _isrchbl = is_reachable(_rangs, robot_param)
        invkin_maps['reachable'][_inx[0], _inx[1]] = 1 * _isrchbl
        
        # 2. Force ration map.
        if (not np.any(np.isnan(_rangs)) and _isrchbl and
            np.linalg.det(_J3.T) != 0):
            _fxy = np.linalg.norm(_Pxy5 @ np.linalg.inv(_J3.T), ord=2)
            _fz = np.linalg.norm(_Pz5 @ np.linalg.inv(_J3.T), ord=2)
            invkin_maps['fratio'][_inx[0], _inx[1]] = _fxy / _fz
        else:
            invkin_maps['fratio'][_inx[0], _inx[1]] = np.nan

        # 3. Robot angles
        invkin_maps['rangles'][_inx[0], _inx[1], :] = _rangs

    return invkin_maps


def visualize_invkinmap(invkin_maps, pha1, pha2, rparam, uh, shpos, gvals):
    r2d = 180 / np.pi
    fig = figure(figsize=(10, 3))

    ax = fig.add_subplot(121)
    ax.grid(color='0.85', linestyle='-', linewidth=0.5)
    ax.set_xlim(r2d * (pha1[0] - 0.0), r2d * (pha1[-1] + 0.0))
    ax.set_ylim(r2d * (pha2[0] - 0.0), r2d * (pha2[-1] + 0.0))
    r2d = (180 / np.pi)
    for _i1 in range(Nha1):
        for _i2 in range(Nha2):
            if invkin_maps['reachable'][_i1, _i2] == 1:
                ax.plot(r2d * pha1[_i1], r2d * pha2[_i2], 's', color='0',
                        markersize=8)
    xticks(fontsize=14)
    yticks(fontsize=14)
    ax.set_xlabel("$\phi_1$", fontsize=16)
    ax.set_ylabel("$\phi_2$", fontsize=16)
    ax.set_title("Reachable Points", fontsize=16)

    ax = fig.add_subplot(122)
    ax.grid(color='0.85', linestyle='-', linewidth=0.5)
    ax.set_xlim(r2d * (pha1[0] - 0.0), r2d * (pha1[-1] + 0.0))
    ax.set_ylim(r2d * (pha2[0] - 0.0), r2d * (pha2[-1] + 0.0))
    # Color of the points
    _col = (invkin_maps['fratio'] - 1)
    _col[_col < 0] = 0
    _col[_col > 1] = 1
    for _i1 in range(Nha1):
        for _i2 in range(Nha2):
            if not np.isnan(invkin_maps['fratio'][_i1, _i2]):
                # Convert fratio to a gray scale value.
                ax.plot(r2d * pha1[_i1], r2d * pha2[_i2], 's',
                        color=f'{1 - _col[_i1, _i2]}', markersize=8)
    xticks(fontsize=14)
    yticks(fontsize=14)
    ax.set_xlabel("$\phi_1$", fontsize=16)
    ax.set_title("Force ratio", fontsize=16)
    _str1 = f"Robot: [{rparam[0]}, {rparam[1]}, {rparam[2]}]"
    _str2 = f"Human: [{shpos[0]}, {shpos[1]}, {shpos[2]}, {uh}]"
    _str3 = f"G scorres: [{gvals[0]:0.3f}, {gvals[1]:0.3f}, {gvals[2]:0.3f}]"
    plt.suptitle("\n".join((f"{_str1} {_str2}", _str3)), x=0.55, y=1.125,
                 fontsize=16)
    plt.tight_layout()

    return fig


def get_goodness_score(invkin_maps, w=0.5):
    # 1. Normalized workspace
    _g1 = np.nanmean(invkin_maps['reachable'])
    
    # 2. Noralized force ratio
    _fr = np.copy(invkin_maps['fratio'])
    _fr[_fr < 1] = 0
    _fr[_fr >= 1] = 1
    _g2 = np.nanmean(_fr)
    
    return (_g1, _g2)


def get_interped_data(data, objective, xcol, ycol, x, y, dorig, dnew):
    xvals = get_from_range(x[0], x[1], dorig)
    yvals = get_from_range(y[0], y[1], dorig)
    xgrid, ygrid = np.meshgrid(xvals, yvals)
    dataxy = np.zeros(np.shape(xgrid))
    for i1, _r1 in enumerate(xvals):
        for i2, _r2 in enumerate(yvals):
            _rinx = ((data[xcol] == _r1) &
                     (data[ycol] == _r2))
            dataxy[i2, i1] = np.mean(data[_rinx][objective])
    
    # Interpolate values.
    intdataxy = interp2d(xgrid, ygrid, dataxy, kind='linear')
    xvalsnew = get_from_range(x[0], x[1], dnew)
    yvalsnew = get_from_range(y[0], y[1], dnew)
    xgridnew, ygridnew = np.meshgrid(xvalsnew, yvalsnew)
    return xvalsnew, yvalsnew, xgridnew, ygridnew, intdataxy(xvalsnew, yvalsnew)


def plot_objective_heatmap(allgood, robot_param, objective="O12", title="Overall Objective"):
    r1vals = get_from_range(robot_param['links']['r1'][0],
                            robot_param['links']['r1'][1],
                            robot_param['links']['dr'])
    r2vals = get_from_range(robot_param['links']['r2'][0],
                            robot_param['links']['r2'][1],
                            robot_param['links']['dr'])
    r3vals = get_from_range(robot_param['links']['r3'][0],
                            robot_param['links']['r3'][1],
                            robot_param['links']['dr'])

    vmin = np.min(allgood[objective])
    vmax = np.max(allgood[objective])

    fig = plt.figure(figsize=(12, 3.5))
    ax = fig.add_subplot(131)
    r1grid, r2grid = np.meshgrid(r1vals, r2vals)
    val12 = np.zeros(np.shape(r1grid))
    for i1, _r1 in enumerate(r1vals):
        for i2, _r2 in enumerate(r2vals):
            _rinx = ((allgood['r1'] == _r1) &
                    (allgood['r2'] == _r2))
            val12[i2, i1] = np.mean(allgood[_rinx][objective])

    c = ax.pcolormesh(r1grid, r2grid, val12, cmap='RdBu', vmin=vmin,
                    vmax=vmax)
    ax.set_title(title, fontsize=16)
    ax.axis([r1grid.min(), r1grid.max(), r2grid.min(), r2grid.max()])
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("r1 (cm)", fontsize=16)
    ax.set_ylabel("r2 (cm)", fontsize=16)
    plt.xticks(np.arange(r1vals[0], r1vals[-1] + 5, 5), fontsize=16)
    plt.yticks(np.arange(r2vals[0], r2vals[-1] + 5, 5), fontsize=16)
    
    ax = fig.add_subplot(132)
    r2grid, r3grid = np.meshgrid(r2vals, r3vals)
    val23 = np.zeros(np.shape(r2grid))
    for i1, _r2 in enumerate(r2vals):
        for i2, _r3 in enumerate(r3vals):
            _rinx = ((allgood['r2'] == _r2) &
                    (allgood['r3'] == _r3))
            val23[i2, i1] = np.mean(allgood[_rinx][objective])

    c = ax.pcolormesh(r2grid, r3grid, val23, cmap='RdBu', vmin=vmin,
                    vmax=vmax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("r2 (cm)", fontsize=16)
    ax.set_ylabel("r3 (cm)", fontsize=16)
    ax.axis([r2grid.min(), r2grid.max(), r3grid.min(), r3grid.max()])
    fig.colorbar(c, ax=ax)
    plt.xticks(np.arange(r2vals[0], r2vals[-1] + 5, 5), fontsize=16)
    plt.yticks(np.arange(r3vals[0], r3vals[-1] + 1, 1), fontsize=16)

    ax = fig.add_subplot(133)
    r1grid, r3grid = np.meshgrid(r1vals, r3vals)
    val13 = np.zeros(np.shape(r1grid))
    for i1, _r1 in enumerate(r1vals):
        for i2, _r3 in enumerate(r3vals):
            _rinx = ((allgood['r1'] == _r1) &
                    (allgood['r3'] == _r3))
            val13[i2, i1] = np.mean(allgood[_rinx][objective])

    c = ax.pcolormesh(r1grid, r3grid, val13, cmap='RdBu', vmin=vmin,
                    vmax=vmax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("r1 (cm)", fontsize=16)
    ax.set_ylabel("r3 (cm)", fontsize=16)
    ax.axis([r1grid.min(), r1grid.max(), r3grid.min(), r3grid.max()])
    fig.colorbar(c, ax=ax)
    plt.xticks(np.arange(r1vals[0], r1vals[-1] + 5, 5), fontsize=16)
    plt.yticks(np.arange(r3vals[0], r3vals[-1] + 1, 1), fontsize=16)

    plt.tight_layout()

    return fig


def plot_objective_heatmap_interp(allgood, robot_param, objective="O12",
                                  title="Overall Objective", dr=0.1):
    vmin = np.min(allgood[objective])
    vmax = np.max(allgood[objective])

    fig = plt.figure(figsize=(12, 3.5))
    ax = fig.add_subplot(131)
    intdata = get_interped_data(allgood, objective, xcol='r1', ycol='r2',
                                x=robot_param['links']['r1'],
                                y=robot_param['links']['r2'],
                                dorig=robot_param['links']['dr'],
                                dnew=0.1)
    xvals, yvals, xgrid, ygrid, dataxy = intdata
    c = ax.pcolormesh(xgrid, ygrid, dataxy, cmap='RdBu',
                      vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=16)
    ax.axis([xvals.min(), xvals.max(), yvals.min(), yvals.max()])
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("r1 (cm)", fontsize=16)
    ax.set_ylabel("r2 (cm)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    ax = fig.add_subplot(132)
    intdata = get_interped_data(allgood, objective, xcol='r2', ycol='r3',
                                x=robot_param['links']['r2'],
                                y=robot_param['links']['r3'],
                                dorig=robot_param['links']['dr'],
                                dnew=0.1)
    xvals, yvals, xgrid, ygrid, dataxy = intdata
    c = ax.pcolormesh(xgrid, ygrid, dataxy, cmap='RdBu',
                      vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=16)
    ax.axis([xvals.min(), xvals.max(), yvals.min(), yvals.max()])
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("r2 (cm)", fontsize=16)
    ax.set_ylabel("r3 (cm)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    ax = fig.add_subplot(133)
    intdata = get_interped_data(allgood, objective, xcol='r1', ycol='r3',
                                x=robot_param['links']['r1'],
                                y=robot_param['links']['r3'],
                                dorig=robot_param['links']['dr'],
                                dnew=0.1)
    xvals, yvals, xgrid, ygrid, dataxy = intdata
    c = ax.pcolormesh(xgrid, ygrid, dataxy, cmap='RdBu',
                      vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=16)
    ax.axis([xvals.min(), xvals.max(), yvals.min(), yvals.max()])
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("r1 (cm)", fontsize=16)
    ax.set_ylabel("r3 (cm)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()

    return fig


def get_val_ranges(data, col1, col2):
    _xvals = data[col1].unique()
    _yvals = np.array([[data[data[col1] == _x][col2].min(),
                        data[data[col1] == _x][col2].median(),
                        data[data[col1] == _x][col2].max()]
                       for _x in _xvals])
    return _xvals, _yvals