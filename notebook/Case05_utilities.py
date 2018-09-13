import numpy as np
import scipy as sp
import scipy.sparse as spm
from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import sys, time, os.path
from itertools import permutations
sys.path.append('../module/')
from SHUtil import SphCoord_to_CartCoord, CartCoord_to_SphCoord

import pyshtools
from SHUtil import SHCilmToVector, SHVectorToCilm, lmk2K, K2lmk
from SHUtil import plotfv, TransMat, l_coeffs, m_coeffs
from ShElastic import calSmode, calUmode
from SHBV import generate_submat, visualize_Cmat, print_SH_mode

from scipy.sparse.linalg import lsqr, spsolve
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

# procedures for transformation between Uvec and Tvec
def Uvec2Tvec(Uvec, Cmat, Dmat, disp=False):
    tic = time.time()
    #B = lsqr(Dmat, Uvec.T)
    B_sol = spsolve(Dmat, Uvec.T)
    toc = time.time()
    #B_sol = B[0]
    #print('Residual:', B[3], 'Time:', toc-tic, 'Solution:', B_sol.size)
    print('Time: %.4fs'%(toc-tic))
    if disp:
        disp_index_sol = print_SH_mode(B_sol, m_dir=3, etol=1e-8)
    return Cmat.dot(B_sol)

def Tvec2Uvec(Tvec, Cmat, Dmat, disp=False):
    tic = time.time()
    A = lsqr(Cmat, Tvec.T, atol=0, btol=0, conlim=0)
    #A_sol = spsolve(Cmat, Tvec.T)
    toc = time.time()
    A_sol = A[0]
    print('Residual:', A[3], 'Time:', toc-tic, 'Solution:', A_sol.size)
    print('Time: %.4fs'%(toc-tic))
    if disp:
        disp_index_sol = print_SH_mode(A_sol, m_dir=3, etol=1e-8)
    return Dmat.dot(A_sol)

def SHVec2mesh(xvec, lmax=None, SphCoord=True, Complex=False):
    if lmax is None:
        lmax = (np.sqrt(xvec.size/3) - 1).astype(np.int)
    cvec = xvec.reshape(3, -1)
    nvec = cvec.shape[1]
    xmesh= [None for _ in range(3)]
    for k in range(3):
        if Complex:
            cext = np.zeros((lmax+1)**2, dtype=np.complex)
            cext[:nvec] = cvec[k, :(lmax+1)**2]
            cilm = SHVectorToCilm(cext)
        else:
            cext = np.zeros((lmax+1)**2)
            cext[:nvec] = cvec[k, :(lmax+1)**2]
            cilm = pyshtools.shio.SHVectorToCilm(cext)
        coeffs = pyshtools.SHCoeffs.from_array(cilm)
        grid = coeffs.expand('GLQ')
        xmesh[k] = grid.to_array().real
    xmesh = np.stack(xmesh, axis=-1)
    if SphCoord:
        Q = TransMat(lJmax=lmax)
        xmesh = np.sum(Q*xmesh[...,np.newaxis,:], axis=-1)
    return xmesh

# visualizing SHvectors in 2D
def visSHVec(xvec, lmax_plot=None, cmap='viridis', show=True, 
             SphCoord=True, config_quiver=(2, 4, 'k', 1), n_vrange=None, s_vrange=None,
             lonshift=0, Complex=False, figsize=(10, 5)):
    xmesh = SHVec2mesh(xvec, lmax=lmax_plot, SphCoord=SphCoord, Complex=Complex)
    if SphCoord:
        fig = [None for _ in range(2)]
        ax = [None for _ in range(2)]
        xshear= np.linalg.norm(xmesh[...,1:], axis=-1)
        
        fig[0], ax[0] = plotfv(xmesh[...,0], show=False, cmap=cmap,vrange=n_vrange,
                               lonshift=lonshift, figsize=figsize)
        ax[0].set_title('norm')
        
        fig[1], ax[1] = plotfv(xshear, show=False, cmap='Reds', lonshift=lonshift, figsize=figsize, vrange=s_vrange)
        latsdeg, lonsdeg = pyshtools.expand.GLQGridCoord(lmax_plot)
        lons, lats = np.meshgrid(lonsdeg, latsdeg)
        xshift = np.roll(xmesh, np.round(lons.shape[1]*lonshift/360).astype(np.int), axis=1)
        st, dq, color, scale = config_quiver
        ax[1].quiver(lons[::dq,st::dq], lats[::dq,st::dq], 
                     xshift[::dq,st::dq,1], xshift[::dq,st::dq,2], 
                     color=color, scale=scale)
        ax[1].set_title('shear')
    else:
        fig = [None for _ in range(3)]
        ax = [None for _ in range(3)]
        titlestr = ('x', 'y', 'z')
        for k in range(3):
            fig[k], ax[k] = plotfv(xmesh[...,k], show=False, cmap=cmap, lonshift=lonshift, figsize=figsize)
            ax[k].set_title('$'+titlestr[k]+'$')
    if show:
        plt.show()
    return fig, ax

# visualizing SHvectors in 3D
def visSH3d(xmesh, cmesh=None, r0=1, lmax_plot=None,
            figsize=(16,16), show=True, filename=None,
            elevation=0, azimuth=0, surface=False, color=None):
    if lmax_plot is None:
        lmax_plot = xmesh.shape[0] - 1
    lats, lons = pyshtools.expand.GLQGridCoord(lmax_plot)
    nlat = lats.size; nlon = lons.size;

    lats_circular = np.hstack(([90.], lats, [-90.]))
    lons_circular = np.append(lons, [lons[0]])
    u = np.radians(lons_circular)
    v = np.radians(90. - lats_circular)
    normvec = np.zeros((nlat+2, nlon+1, 3))
    normvec[...,0] = np.sin(v)[:, None] * np.cos(u)[None, :]
    normvec[...,1] = np.sin(v)[:, None] * np.sin(u)[None, :]
    normvec[...,2] = np.cos(v)[:, None] * np.ones_like(lons_circular)[None, :]

    upoints = np.zeros((nlat + 2, nlon + 1, 3))
    upoints[1:-1, :-1, :] = xmesh
    upoints[0, :, :] = np.mean(xmesh[0,:,:], axis=0)  # not exact !
    upoints[-1, :, :] = np.mean(xmesh[-1,:,:], axis=0)  # not exact !
    upoints[1:-1, -1, :] = xmesh[:, 0, :]
    upoints *= r0
    
    x = r0 * np.sin(v)[:, None] * np.cos(u)[None, :]  + upoints[..., 0]
    y = r0 * np.sin(v)[:, None] * np.sin(u)[None, :] + upoints[..., 1]
    z = r0 * np.cos(v)[:, None] * np.ones_like(lons_circular)[None, :] + upoints[...,2]

    if color is None:
        if cmesh is None:
            magn_point = np.sum(normvec * upoints, axis=-1)
        else:
            tpoints = np.zeros((nlat + 2, nlon + 1, 3))
            tpoints[1:-1, :-1, :] = cmesh
            tpoints[0, :, :] = np.mean(cmesh[0,:,:], axis=0)  # not exact !
            tpoints[-1, :, :] = np.mean(cmesh[-1,:,:], axis=0)  # not exact !
            tpoints[1:-1, -1, :] = cmesh[:, 0, :]
            magn_point = np.sum(normvec * tpoints, axis=-1)
        magn_face = 1./4. * (magn_point[1:, 1:] + magn_point[:-1, 1:] +
                             magn_point[1:, :-1] + magn_point[:-1, :-1])
        magnmax_face = np.max(np.abs(magn_face))
        magnmax_point = np.max(np.abs(magn_point))
        norm = plt.Normalize(-magnmax_face / 2., magnmax_face / 2., clip=True)
        cmap = plt.get_cmap('RdBu_r')
        colors = cmap(norm(magn_face.flatten()))
        colors = colors.reshape(nlat + 1, nlon, 4)

    fig = plt.figure(figsize=figsize)
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')
    if surface:
        if color is None:
            ax3d.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors)
        else:
            ax3d.plot_surface(x, y, z, rstride=1, cstride=1, color=color)
    else:
        ax3d.scatter(x, y, z)
    ax3d.view_init(elev=elevation, azim=azimuth)
    ax3d.tick_params(labelsize=16)

    if filename is not None:
        plt.tight_layout()
        fig.savefig(filename,transparent=True)
    if show:
        plt.show(block=True)
    return fig, ax3d

# convert between complex and real SH vectors

def SHvec_rtoc(xvec):
    lmax = np.int(np.sqrt(xvec.size/3)-1)
    rcilm = pyshtools.shio.SHVectorToCilm(xvec)
    ccilm = pyshtools.shio.SHrtoc(rcilm)
    cilm = np.zeros_like(ccilm, dtype=np.complex)
    cilm[0,...] = ccilm[0,...] + 1.j*ccilm[1,...]
    cilm[1,:,1:] = (-1)**m_coeffs(lmax)[0,:,1:] * cilm[0,:,1:].conj()
    return SHCilmToVector(cilm)

def SHvec_ctor(xvec):
    cilm = SHVectorToCilm(xvec)
    ccilm = np.zeros_like(cilm, dtype=np.float)
    ccilm[0,...] = cilm[0,...].real
    ccilm[1,...] = cilm[0,...].imag
    rcilm = pyshtools.shio.SHctor(ccilm)
    return pyshtools.shio.SHCilmToVector(rcilm)

# coefficients to shape-distance

def coeffs2dr(uvec, f_interp=None, lmax=None, X0=None, Complex=False,
              lat_weights=1, vert_weight=1, l_weight=None, norm_order=1, debug=False):
    #### shape difference from SH vectors
    ## uvec: Input real SH vector (3x(lmax+1)^2)
    ## f_interp: interpolation function that calculates r(theta,phi)
    ## lmax, X0: pre-calculated parameters for speed-up. Explained in the code.
    ## Complex: Default False, uvec is the real SH vector using pyshtools.shio.SHVectorToCilm;
    ##          otherwise, it will be complex SH vector using SHUtil.SHVectorToCilm;
    ## lat_weights: weighing function for different densities of nodes on different lattitudes.
    ## vert_weight: weighing function for different error in different directions
    ## l_weight: weighing function for coefficients (anti-aliasing)
    if lmax is None:
        lmax = np.sqrt(uvec.size/3).astype(np.int) - 1
    nvec = (lmax+1)**2; 
    if norm_order > 1: #== np.inf:
        nmesh = 1
    else:
        nmesh = (lmax+1)*(2*lmax+1)
    cvec = uvec.reshape(3, nvec)
    # expand the displacement field onto a mesh
    umesh= [None for _ in range(3)]
    for k in range(3):
        if Complex:
            cilm = SHVectorToCilm(cvec[k, :])
        else:
            cilm = pyshtools.shio.SHVectorToCilm(cvec[k, :])
        coeffs = pyshtools.SHCoeffs.from_array(cilm)
        grid = coeffs.expand('GLQ')
        umesh[k] = grid.to_array().real
    umesh = np.stack(umesh, axis=-1)
    # calculate the shape
    if X0 is None:
        lon = np.deg2rad(grid.lons())
        colat = np.deg2rad(90-grid.lats())
        PHI, THETA = np.meshgrid(lon, colat)
        R = np.ones_like(PHI)
        X,Y,Z = SphCoord_to_CartCoord(R, THETA, PHI)
        X0 = np.stack([X,Y,Z], axis=-1)
    Xs = X0 + umesh
    rs, ts, ps = CartCoord_to_SphCoord(Xs[...,0], Xs[...,1], Xs[...,2])
    lats = 90 - np.rad2deg(ts); lons = np.rad2deg(ps); lons[lons < 0] += 360;
    rref = f_interp.ev(lats, lons) + 1
    Xref = np.stack(SphCoord_to_CartCoord(rref, ts, ps), axis=-1)
    if debug:
        return Xs - Xref

    norm2 = np.sum(((Xs - Xref)*vert_weight)**2, axis=-1)*(lat_weights**2)
    return np.linalg.norm(norm2.flatten(), ord=norm_order)/nmesh

def coeffs2dist(uvec, Xt=None, f_cached=None, lmax=None, X0=None, Complex=False, 
                lat_weights=1, vert_weight=1, l_weight=None, debug=False):
    #### shape difference from SH vectors
    ## uvec: Input real SH vector (3x(lmax+1)^2)
    ## Xt: Input shape data (nv x 3) or (nf x 3 x 3); can be full list or neighbor list
    ## f_cached: Default None, Xt is the coordinates of vertices; otherwise, Xt is the
    ##          coordinates of faces, and f_cached provides the pre-calculated values.
    ## lmax, X0: pre-calculated parameters for speed-up. Explained in the code.
    ## Complex: Default False, uvec is the real SH vector using pyshtools.shio.SHVectorToCilm;
    ##          otherwise, it will be complex SH vector using SHUtil.SHVectorToCilm;
    ## lat_weights: weighing function for different densities of nodes on different lattitudes.
    ## vert_weight: weighing function for different error in different directions
    ## l_weight: weighing function for coefficients (anti-aliasing)
    if lmax is None:
        lmax = np.sqrt(uvec.size/3).astype(np.int) - 1
    nvec = (lmax+1)**2
    cvec = uvec.reshape(3, nvec)
    # expand the displacement field onto a mesh
    umesh= [None for _ in range(3)]
    for k in range(3):
        if Complex:
            cilm = SHVectorToCilm(cvec[k, :])
        else:
            cilm = pyshtools.shio.SHVectorToCilm(cvec[k, :])
        coeffs = pyshtools.SHCoeffs.from_array(cilm)
        grid = coeffs.expand('GLQ')
        umesh[k] = grid.to_array().real
    umesh = np.stack(umesh, axis=-1)
    # calculate the shape
    if X0 is None:
        lon = np.deg2rad(grid.lons())
        colat = np.deg2rad(90-grid.lats())
        PHI, THETA = np.meshgrid(lon, colat)
        R = np.ones_like(PHI)
        X,Y,Z = SphCoord_to_CartCoord(R, THETA, PHI)
        X0 = np.stack([X,Y,Z], axis=-1)
    Xs = X0 + umesh

    if f_cached is None:
        d2surfsum = np.sum(((Xs[..., np.newaxis, :]-Xt))**2*vert_weight, axis=-1)
        d2surf = d2surfsum.min(axis=-1)
        if debug:
            return d2surfsum
    else:
        #d2surf = d2f(Xs, Xt, f_cached=f_cached, avg_dist=False, vert_weight=vert_weight).min(axis=-1)
        r0, r1, r2, nf, r00, r11, r01, d = f_cached; Xf = Xt;
        pq = np.sum((Xs[...,np.newaxis,:]-r0)*nf, axis=-1)             # n x m
        q = Xs[...,np.newaxis,:] - pq[...,np.newaxis]*nf               # n x m x 3, projection point
        d2fmat = np.sum(((Xs[...,np.newaxis,:]-r0)*nf)**2*vert_weight, axis=-1)   # n x m

        # determine the barycentric coordinate of q
        r12  = np.sum((r2-r0)*(q-r0), axis=-1)                         # n x m
        r02  = np.sum((r1-r0)*(q-r0), axis=-1)                         # n x m
        bary = np.zeros_like(q)                                        # n x m x 3
        bary[...,2] = (r00*r12-r01*r02)/d
        bary[...,1] = (r11*r02-r01*r12)/d
        bary[...,0] = 1 - bary[...,1] - bary[...,2]
        out = np.any(bary < 0, axis=-1)                                # n x m

        # determine the closest point on the edges
        Xfv= np.broadcast_to(Xf, q.shape+(3,))[out]                    # n_out x 3 x 3
        Xp = np.broadcast_to(Xs[...,np.newaxis,:], q.shape)[out]       # n_out x 3
        ve = np.roll(Xfv, 1, axis=-2)-Xfv                              # n_out x 3 x 3
        le = np.linalg.norm(ve, axis=-1)                               # n_out x 3
        ts = np.sum(ve*(Xp[...,np.newaxis,:]-Xfv), axis=-1)/le**2      # n_out x 3
        ts[ts > 1] = 1; ts[ts < 0] = 0;
        qs = Xfv+(ts[...,np.newaxis]*ve)
        dq = np.sum(((Xp[...,np.newaxis,:]-qs))**2*vert_weight, axis=-1)       # n_out x 3
        d2fmat[out] = np.min(dq, axis=-1)
        d2surf = d2fmat.min(axis=-1)
        if debug:
            return d2fmat
    
    if l_weight is None:
        regularization = 0
    else:
        regularization = np.linalg.norm(np.tile(l_weight, 3)*uvec)
    return np.mean(d2surf*lat_weights) + regularization*0.05

# target functions

def sol2dr(aK, Cmat, Dmat, alpha = 0.05, beta=0.05, isTfv=None,
           f_interp=None, lmax=None, X0=None,
           lat_weights=None, vert_weight=1, l_weight=None, norm_order=1, separate=False):
    if lat_weights is None:
        lat_weights = np.ones((lmax+1, 2*lmax+1))
    nvec = (lmax+1)**2
    if np.nonzero(isTfv)[0].size == 0:
        Tdist = 0
        Tvec = 0
    else:
        Tvec = Cmat.dot(aK)
        tcvec = Tvec.reshape(3, -1)
        # expand the displacement and traction field onto a mesh
        tmesh = np.empty((lmax+1, 2*lmax+1, 3))
        for k in range(3):
            tcilm = SHVectorToCilm(tcvec[k, :])
            tgrid = pyshtools.SHCoeffs.from_array(tcilm).expand('GLQ')
            tmesh[..., k] = tgrid.to_array().real
        if isTfv.dtype == np.bool:
            tvalues = np.sum((tmesh[isTfv, :]*vert_weight)**2, axis=-1)
            Tdist = np.mean(tvalues*lat_weights[isTfv]**2)
        else:
            tvalues = np.sum((tmesh * vert_weight)**2, axis=-1)*isTfv
            Tdist = np.mean(tvalues*lat_weights**2)
    Uvec = Dmat.dot(aK)    
    Udist = coeffs2dr(Uvec, f_interp=f_interp, lmax=lmax, X0=X0, Complex=True,
                      lat_weights=lat_weights, vert_weight=vert_weight, norm_order=norm_order)
    #regularization = np.vdot(Uvec, Tvec).real*2*np.pi
    regularization = np.vdot(Uvec, Uvec*l_weight).real + alpha*np.vdot(Tvec, Tvec*l_weight).real
    #regularization = np.vdot(aK, aK*l_weight).real
    if separate:
        E = np.vdot(Uvec, Tvec).real*2*np.pi
        return (Udist, Tdist, regularization, E)
    return (Udist + alpha*Tdist + beta*regularization)

# target function
def sol2dist(aK, Cmat, Dmat, alpha = 0.05, beta=0.05, isTfv=None, 
             f_cached=None, lmax=None, X0=None, Xt=None,
             lat_weights=None, vert_weight=1, l_weight=None, separate=False):
    if lat_weights is None:
        lat_weights = np.ones((lmax+1, 2*lmax+1))
    if np.nonzero(isTfv)[0].size == 0:
        Tdist = 0
    else:
        Tvec = Cmat.dot(aK)
        nvec = (lmax+1)**2
        tcvec = Tvec.reshape(3, -1)
        # expand the displacement and traction field onto a mesh
        tmesh = np.empty((lmax+1, 2*lmax+1, 3))
        for k in range(3):
            tcilm = SHVectorToCilm(tcvec[k, :])
            tgrid = pyshtools.SHCoeffs.from_array(tcilm).expand('GLQ')
            tmesh[..., k] = tgrid.to_array().real
        if isTfv.dtype == np.bool:
            tvalues = np.sum((tmesh[isTfv, :]*vert_weight)**2, axis=-1)
            Tdist = np.mean(tvalues*lat_weights[isTfv]**2)
        else:
            tvalues = np.sum((tmesh * vert_weight)**2, axis=-1)*isTfv
            Tdist = np.mean(tvalues*lat_weights**2)
    Uvec = Dmat.dot(aK)
    Udist = coeffs2dist(Uvec, Xt=Xt, f_cached=f_cached, lmax=lmax, X0=X0, Complex=True,
                        lat_weights=lat_weights, vert_weight=vert_weight)
    regularization = np.vdot(aK, aK*l_weight).real
    if separate:
        return (Udist, Tdist, regularization)
    return (Udist + alpha*Tdist + beta*regularization)

# output of the target function
def sol2dist_verbose(Asol, r0=1, mu0=1):
    mean_dist = np.sqrt(Asol[0])*r0
    mean_T = np.sqrt(Asol[1])*mu0
    print('  mean shape difference = %.4fum'%mean_dist)
    print('  mean |T| in free surface = %.4fPa'%mean_T)
    print('  regularization = %e'%Asol[2])
    print('  Energy = %epJ'%(Asol[3]*(r0/1e6)**3*mu0*1e12))
    return (mean_dist, mean_T)

# update the arguments of the target function
def sol2dist_update(AK_iter, args, file_neigh):
    print('        recalculate neighbor list')
    Cmat, Dmat, myalpha, mybeta, f_cached, lJmax, X0, XFneigh, lat_weights, vert_weight, A_weight = args
    Uvec_iter = Dmat.dot(AK_iter)
    Xt = X0 + SHVec2mesh(Uvec_iter, lmax=lJmax, SphCoord=False, Complex=True)
    if f_cached is None: # node-node dist
        Xn = np.load(file_neigh+'.npz')['Xneigh']
        XFneigh = generate_Xneigh(Xt, Xn, n_list=n_load, filename='tmp')
    else:               # node-face dist
        Fn = np.load(file_neigh+'.npz')['Fneigh']
        XFneigh = generate_Fneigh(Xt, Fn, n_list=n_load, filename='tmp')
        f_cached = generate_fcache(XFneigh)
    return (Cmat, Dmat, myalpha, mybeta, f_cached, lJmax, X0, XFneigh, lat_weights, vert_weight, A_weight)

# minimization algorithm

def minimize_AK(AK_init, target, args, iter_config, verbose=None, verbose_args=(), AKfile='', fvfile='', file_neigh=''):
    #### optimization iteration wrapper
    ## AK_init: Initial guess of the AK solution
    ## target: target function, arguments (AK_init, *args, verbose)
    ## args: arguments for target function
    ## iter_config: {'N_period', 'minimizer', 'minimizer_config', 
    ##               'n_update', 'update'} # settings for periodic updating (e.g. neighbor list)
    ## verbose: print details
    ## AKfile, fvfile: save current AK and iteration history.
    ## f_neigh: neighbor list file

    magnify = 1e4
    # print initial settings
    if os.path.exists(AKfile):
        AK_iter = np.load(AKfile)
        print('loading AK from file, target function value: %.4e'%(target(AK_iter, *args)))
    else:
        AK_iter = AK_init.copy()
        print('initial AK, target function value: %.4e'%(target(AK_iter, *args)))
    if verbose is not None:
        verbose(target(AK_iter, *args, True), *verbose_args)

    if os.path.exists(fvfile):
        tmp = np.loadtxt(fvfile)
        funval = tmp.reshape(-1, tmp.shape[-1])[:, 1:]
    else:
        funval = None
    
    # optimization iterations
    N_period = iter_config['N_period']
    minimizer = iter_config['minimizer']
    minimizer_config = iter_config['minimizer_config']
    n_update = iter_config['n_update']
    if n_update > 0:
        print('update args per %d iterations'%n_update)
    for i in range(N_period):
        if n_update > 0:
            if np.remainder(i, n_update) == np.remainder(N_period-1, n_update):
                print('      update args...')
                args = iter_config['update'](AK_iter, args, file_neigh)

        print(' period %d/%d'%(i, N_period))
        tic = time.time()
        magnified_target = lambda AK : target(AK, *args)*magnify
        AK_min = minimize(magnified_target, AK_iter, #args=args,
                          method=minimizer, options=minimizer_config)
        toc = time.time()
        print('  period %d: F = %.4e, time: %.4fs'%(i, AK_min.fun/magnify, toc-tic))
        
        fparts = target(AK_min.x, *args, True)
        if verbose is not None:
            fverb = verbose(fparts, *verbose_args)
            fvalue = [AK_min.fun/magnify, *fverb, *fparts]
        else:
            fvalue = [AK_min.fun/magnify, *fparts]
        if funval is None:
            funval = np.array(fvalue).reshape(1, -1)
        else:
            funval = np.vstack([funval, fvalue])
        AK_iter = AK_min.x
        if not (AKfile == ''):
            np.save(AKfile, AK_min.x)
        if not (fvfile == ''):
            fvfmt = '%8d'+(' %e'*(funval.shape[1]))
            np.savetxt(fvfile, np.hstack([np.arange(funval.shape[0])[:, np.newaxis], funval]), fmt=fvfmt)
    return AK_min.x, funval

# generating neighboring list

def d2v(Xs, Xv, avg_dist=True, vert_weight=1):
    #### point-to-point distance
    ## Xs: points for testing (..., 3); 
    ## Xv: m data points (m, 3);
    ## avg_dist: if True, calculate the average shape difference per node
    ## vert_weight: the weighing function for different coordinates (3, )
    d2vmat = np.linalg.norm((Xs[..., np.newaxis, :]-Xv)*vert_weight, axis=-1)   # pair-wise distances (..., m)
    if avg_dist:
        return d2vmat.min(axis=-1).mean()
    else:
        return d2vmat

def d2e(Xs, Xe, e_cached=None, avg_dist=True, infval=False, debug=False):
    #### point-to-edge distance
    ## Xs: points for testing (..., 3); 
    ## Xe: m edges from data (m, 2, 3);
    ## avg_dist: if True, calculate the average shape difference per node
    ## infval: if True, the projection outside the edge will be consider infinite distance
    ## debug: output info for debug/testing
    if e_cached is None:
        r1 = Xe[..., 0, :]; r2 = Xe[..., 1, :]; r12 = r2 - r1;        # m x 3, edge vertices
        l12 = np.linalg.norm(r12, axis=-1)                            # m,     edge length
    else:
        r1, r12, l12 = e_cached
    t = np.sum((Xs[..., np.newaxis, :]-r1)*r12, axis=-1)/l12**2   # n x m,     projection ratio
    t_cal = t.copy(); t_cal[t<0] = 0; t_cal[t>1] = 1;             # a copy of t for calculation
    q = r1 + (t_cal[...,np.newaxis]*r12)                          # n x m x 3, projection point
    d2emat = np.linalg.norm(Xs[..., np.newaxis, :] - q, axis=-1)  # n x m
    if infval:
        d2emat[np.logical_or(t<0, t>1)] = np.inf
    if avg_dist:
        return d2emat.min(axis=-1).mean()
    else:
        if debug:
            dr1 = np.linalg.norm(Xs[..., np.newaxis, :] - r1, axis=-1)
            dr2 = np.linalg.norm(Xs[..., np.newaxis, :] - r2, axis=-1)
            return (d2emat, q, dr1, dr2)
        else:
            return d2emat
        
def generate_fcache(Xf):
    r0 = Xf[..., 0, :]; r1 = Xf[..., 1, :]; r2 = Xf[..., 2, :];    # m x 3
    nf = np.cross(r1-r0, r2-r0)                                    # normal vector
    nf = nf / np.linalg.norm(nf, axis=-1)[...,np.newaxis]          # m x 3
    r11 = np.sum((r2 - r0)**2, axis=-1)                            # m
    r00 = np.sum((r1 - r0)**2, axis=-1)                            # m
    r01 = np.sum((r1-r0)*(r2-r0), axis=-1)                         # m
    d = r11*r00 - r01*r01                                          # m
    return r0, r1, r2, nf, r00, r11, r01, d

def d2f(Xs, Xf, f_cached=None, avg_dist=True, infval=False, debug=False, vert_weight=1, fasteval=False):
    #### point-to-face distance
    ## Xs: points for testing (..., 3); 
    ## Xf: m faces from data (m, 3, 3);
    ## avg_dist: if True, calculate the average shape difference per node
    ## infval: if True, the projection outside the face will be consider infinite distance
    ## debug: output info for debug/testing
    ## vert_weight: the weighing function for different coordinates (3, )
    ## fasteval: if True, only calculate the point-vertex distances for estimation.

    if fasteval:
        d2fmat = np.linalg.norm(Xs[...,np.newaxis,np.newaxis,:] - Xf, axis=-1).min(axis=-1)
    else:
        if f_cached is None:
            f_cached = generate_fcache(Xf)
        r0, r1, r2, nf, r00, r11, r01, d = f_cached
        pq = np.sum((Xs[...,np.newaxis,:]-r0)*nf*vert_weight, axis=-1)     # n x m
        q = Xs[...,np.newaxis,:] - pq[...,np.newaxis]*nf               # n x m x 3, projection point
        d2fmat = np.abs(pq)                                            # n x m

        # determine the barycentric coordinate of q
        r12  = np.sum((r2-r0)*(q-r0), axis=-1)                         # n x m
        r02  = np.sum((r1-r0)*(q-r0), axis=-1)                         # n x m
        bary = np.zeros_like(q)                                        # n x m x 3
        bary[...,2] = (r00*r12-r01*r02)/d
        bary[...,1] = (r11*r02-r01*r12)/d
        bary[...,0] = 1 - bary[...,1] - bary[...,2]
        out = np.any(bary < 0, axis=-1)                                # n x m

        # determine the closest point on the edges
        Xfv= np.broadcast_to(Xf, q.shape+(3,))[out]                    # n_out x 3 x 3
        Xp = np.broadcast_to(Xs[...,np.newaxis,:], q.shape)[out]       # n_out x 3
        ve = np.roll(Xfv, 1, axis=-2)-Xfv                              # n_out x 3 x 3
        le = np.linalg.norm(ve, axis=-1)                               # n_out x 3
        ts = np.sum(ve*(Xp[...,np.newaxis,:]-Xfv), axis=-1)/le**2      # n_out x 3
        ts[ts > 1] = 1; ts[ts < 0] = 0;
        qs = Xfv+(ts[...,np.newaxis]*ve)
        dq = np.linalg.norm(((Xp[...,np.newaxis,:]-qs)*vert_weight), axis=-1)       # n_out x 3
        d2fmat[out] = np.min(dq, axis=-1)
        if debug:
            return (d2fmat, q)
        else:
            return d2fmat

    if avg_dist:
        return d2fmat.min(axis=-1).mean()
    else:
        return d2fmat

def generate_Xneigh(Xt, Xref, n_list=200, filename=None):
    d2vmat = d2v(Xt, Xref, avg_dist=False)
    d2varg = np.argsort(d2vmat)[...,:n_list]
    if len(Xref.shape) > 2:
        d2varg0 =np.broadcast_to(np.arange(d2varg.shape[0])[:,np.newaxis,np.newaxis], d2varg.shape)
        d2varg1 =np.broadcast_to(np.arange(d2varg.shape[1])[np.newaxis,:,np.newaxis], d2varg.shape)
        Xneigh = Xref[(d2varg0, d2varg1, d2varg)]
    else:
        Xneigh = Xref[d2varg]
    if filename is not None:
        np.savez(filename+'_Xneigh', Xneigh=Xneigh)
    return Xneigh

def generate_Eneigh(Xt, Eref, n_list=200, filename=None):
    d2emat = d2e(Xt, Eref, avg_dist=False)
    d2earg = np.argsort(d2emat)[...,:n_list]
    if len(Eref.shape) > 3:
        d2earg0 =np.broadcast_to(np.arange(d2earg.shape[0])[:,np.newaxis,np.newaxis], d2earg.shape)
        d2earg1 =np.broadcast_to(np.arange(d2earg.shape[1])[np.newaxis,:,np.newaxis], d2earg.shape)
        Eneigh = Eref[(d2earg0, d2earg1, d2earg)]
    else:
        Eneigh = Eref[d2earg]
    if filename is not None:
        np.savez(filename+'_Eneigh', Eneigh=Eneigh)
    return Eneigh

def generate_Fneigh(Xt, Fref, n_list=200, filename=None):
    if Fref.shape[-3] > 1000:
        d2fmat = d2f(Xt, Fref, avg_dist=False, fasteval=True)
        d2farg = np.argsort(d2fmat)[...,:n_list]
        Fn = Fref[d2farg]
        d2fmat = d2f(Xt, Fn, avg_dist=False)
        d2farg = np.argsort(d2fmat)[...,:n_list]
        d2farg0 =np.broadcast_to(np.arange(d2farg.shape[0])[:,np.newaxis,np.newaxis], d2farg.shape)
        d2farg1 =np.broadcast_to(np.arange(d2farg.shape[1])[np.newaxis,:,np.newaxis], d2farg.shape)
        Fneigh = Fn[(d2farg0, d2farg1, d2farg)]
    else:
        d2fmat = d2f(Xt, Fref, avg_dist=False)
        d2farg = np.argsort(d2fmat)[...,:n_list]
        if len(Fref.shape) > 3:
            d2farg0 =np.broadcast_to(np.arange(d2farg.shape[0])[:,np.newaxis,np.newaxis], d2farg.shape)
            d2farg1 =np.broadcast_to(np.arange(d2farg.shape[1])[np.newaxis,:,np.newaxis], d2farg.shape)
            Fneigh = Fref[(d2farg0, d2farg1, d2farg)]
        else:
            Fneigh = Fref[d2farg]
    np.savez(filename+'_Fneigh', Fneigh=Fneigh)
    return Fneigh

def generate_neighbor_list(Xt, Xref=None, Eref=None, Fref=None, n_list=200, filename=None):
    return_value = []
    if Xref is not None:
        d2vmat = d2v(Xt, Xref, avg_dist=False)
        d2varg = np.argsort(d2vmat)[...,:n_list]
        if len(Xref.shape) > 2:
            d2varg0 =np.broadcast_to(np.arange(d2varg.shape[0])[:,np.newaxis,np.newaxis], d2varg.shape)
            d2varg1 =np.broadcast_to(np.arange(d2varg.shape[1])[np.newaxis,:,np.newaxis], d2varg.shape)
            Xneigh = Xref[(d2varg0, d2varg1, d2varg)]
        else:
            Xneigh = Xref[d2varg]
        return_value.append(Xneigh)
        if filename is not None:
            np.savez(filename+'_Xneigh', Xneigh=Xneigh)
    if Eref is not None:
        d2emat = d2e(Xt, Eref, avg_dist=False)
        d2earg = np.argsort(d2emat)[...,:n_list]
        if len(Eref.shape) > 3:
            d2earg0 =np.broadcast_to(np.arange(d2earg.shape[0])[:,np.newaxis,np.newaxis], d2earg.shape)
            d2earg1 =np.broadcast_to(np.arange(d2earg.shape[1])[np.newaxis,:,np.newaxis], d2earg.shape)
            Eneigh = Eref[(d2earg0, d2earg1, d2earg)]
        else:
            Eneigh = Eref[d2earg]
        return_value.append(Eneigh)
        if filename is not None:
            np.savez(filename+'_Eneigh', Eneigh=Eneigh)
    if Fref is not None:
        if Fref.shape[-3] > 1000:
            d2fmat = d2f(Xt, Fref, avg_dist=False, fasteval=True)
            d2farg = np.argsort(d2fmat)[...,:n_list]
            Fn = Fref[d2farg]
            d2fmat = d2f(Xt, Fn, avg_dist=False)
            d2farg = np.argsort(d2fmat)[...,:n_list]
            d2farg0 =np.broadcast_to(np.arange(d2farg.shape[0])[:,np.newaxis,np.newaxis], d2farg.shape)
            d2farg1 =np.broadcast_to(np.arange(d2farg.shape[1])[np.newaxis,:,np.newaxis], d2farg.shape)
            Fneigh = Fn[(d2farg0, d2farg1, d2farg)]
        else:
            d2fmat = d2f(Xt, Fref, avg_dist=False)
            d2farg = np.argsort(d2fmat)[...,:n_list]
            if len(Fref.shape) > 3:
                d2farg0 =np.broadcast_to(np.arange(d2farg.shape[0])[:,np.newaxis,np.newaxis], d2farg.shape)
                d2farg1 =np.broadcast_to(np.arange(d2farg.shape[1])[np.newaxis,:,np.newaxis], d2farg.shape)
                Fneigh = Fref[(d2farg0, d2farg1, d2farg)]
            else:
                Fneigh = Fref[d2farg]
        np.savez(filename+'_Fneigh', Fneigh=Fneigh)
        return_value.append(Fneigh)
    return tuple(return_value)
