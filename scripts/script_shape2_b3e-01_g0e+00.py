#!/usr/bin/env python
# coding: utf-8

# # Hydrogel-Cell Interaction Test Case
# 
# In this case, we only have the deformed shape of the spherical hydrogel as shown in the next few cells.

# In[ ]:


import numpy as np
import scipy as sp
import scipy.sparse as spm
from scipy.sparse.linalg import lsqr, spsolve
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import sys, time, os.path
from itertools import permutations
import pyshtools

sys.path.append('..')
from shelastic.shutil import SphCoord_to_CartCoord, CartCoord_to_SphCoord, GLQCartCoord
from shelastic.shutil import SHCilmToVector, SHVectorToCilm, SHVec2mesh, SHmesh2Vec
from shelastic.shutil import TransMat, l_coeffs, m_coeffs, LM_list, lmk2K, K2lmk
from shelastic.shbv import generate_submat, Uvec2Tvec, Tvec2Uvec
from shelastic.shvis  import plotfv, vismesh, visSHVec, visSH3d
from sphere_utils import loadCoeffs, arbitrary_force, Ur_interp
from sphere_utils import usurf2umesh, dumesh_dus, usurf2vec
from sphere_utils import Tvec2Tres, usurf2dr2, calculateTfv, genSmat, genLmat, grad_usurf2dr2


# ## Input information

# In[ ]:


############################# change inputs here #################################
# Data file name
datadir = '../testdata'
smoothed = 'smoothed_3'
dilated = '_1um_dilated'
#dilated = '_softedge'
shapename = 'Shape2'
datafile = os.path.join(datadir, shapename+'_Coordinates_Cart_'+smoothed+'.csv')
connfile = os.path.join(datadir, shapename+'_Connectivity.csv')
maskfile = os.path.join(datadir, shapename+'_Mask'+dilated+'.csv')

# Material properties
mu0 = 300/3; nu0 = 0.499;

# Spherical Harmonics Analysis Settings
lJmax = 20; lKmax = lJmax; lmax_plot = 60;

# initial guess settings
init_guess_type = 'ur-only' #'ur-only' or 'true'

# regularizations
myalpha = 1     # traction magnitude
mybeta  = 3e-01 # 1  # coefficient magnitude
mygamma = 0e+00 # 1  # penalty magnitude

# program switches
plot_figure = False
myord = 1; # myord: p-norm order (1-mean value)

# minimization settings
N_period = 2000
maxiter_per_period = 5
CG_gtol = 1e-5
eps = 1e-5               # for jacobian

minimizer = 'CG'
minimizer_config = {'maxiter': maxiter_per_period, 'disp': True, 'gtol': CG_gtol}

# dump files for minimization
savename = shapename+('_b%.0e'%(mybeta))+('_lmax%d'%lJmax)+smoothed+dilated+'_g%.0e'%mygamma

# settings for loading \hat{U}, \hat{T} coefficients
Cmat, Dmat = loadCoeffs(mu0, nu0, lJmax, 'reg')


# ## Load the geometry

# In[ ]:


#### load the geometry ####
# Vs, Vp: list of nodes (nV, trivial), coordinates of the nodes (nVx3)
# Es, Ep: list of edges (nEx2), list of points on the edges (nEx2x3)
# Fs, Fp: list of facets (nFx3), list of points on the facets (nFx3x3)
# Tfv: traction free boundary map of the node list
# Tfe: traction free boundary map of the edge list
# Tff: traction free boundary map of the face list
# Tf_diluted: diluted traction free boundary map

data = np.genfromtxt(datafile, delimiter=',')
conn = np.genfromtxt(connfile, delimiter=',', dtype=np.int)
if dilated == '_softedge':
    masktype = np.float
else:
    masktype = np.int
if shapename == 'Shape4':
    mask = np.zeros_like(data[:,0]).astype(masktype)
else:
    mask = np.genfromtxt(maskfile, dtype=masktype)
print('data, connectivity:', data.shape, conn.shape)

Fs = conn - 1
Np = data.shape[0]
Vs = np.arange(Np)
edge_conn = spm.lil_matrix((Np, Np), dtype=bool)
for i, j in permutations(range(3), 2):
    edge_conn[Fs[:, i], Fs[:, j]] = True
Es = spm.triu(edge_conn).tocoo()
Es = np.vstack([Es.row, Es.col]).T
print('id of nodes, edges, facets:', Vs.shape, Es.shape, Fs.shape)
Vp = data[..., :3]
Ep = Vp[Es, :]; Fp = Vp[Fs, :];
print('coord of nodes, edges, facets:', Vp.shape, Ep.shape, Fp.shape)

if dilated == '_softedge':
    Tfv = (mask > 0.5)
else:
    Tfv = mask.astype(np.bool)

#### Plot the geometry (Vp) ####
if plot_figure:
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d')

    nTfv = np.logical_not(Tfv)
    ax.scatter3D(Vp[Tfv, 0], Vp[Tfv, 1], Vp[Tfv, 2])
    ax.scatter3D(Vp[nTfv, 0], Vp[nTfv, 1], Vp[nTfv, 2])

    ax.view_init(azim=0, elev=0)
    plt.show()


# Then we determine the original radius of the particle, assuming the particle is incompressible. The total volume can be estimated by adding the volume of the tetrahedrons. The volume of a tetrahedron is calculated as:
# 
# $$
# V_{0123}=\frac{1}{6}
# \begin{vmatrix}
#  x_1 & y_1 & z_1 & 1\\ 
#  x_2 & y_2 & z_2 & 1\\ 
#  x_3 & y_3 & z_3 & 1\\ 
#  0 & 0 & 0 & 1\\ 
# \end{vmatrix}
# $$

# In[ ]:


tet = np.zeros((Fs.shape[0], 4, 4))
tet[:,:-1,:-1] = Fp
tet[:,:,-1] = 1
vol = np.sum(np.linalg.det(tet)/6, axis=0)
r0 = np.cbrt(vol/(4/3*np.pi))
print('V = %.4f, r0 = %.4f'%(vol, r0))


# We need to solve the reverse problem of a deformed shape. We will try the following methods to tackle this problem:
# 
# 1. Assume $r$-direction deformation only, using the solution as initial guess to the optimization
# 2. LSQ solving coeffs of the SH solutions for fitting both the traction-free boundary and the shape

# Conversion between complex and real spherical harmonics, for $m>0$
# 
# $$
# f_{lm} = \left[f_l^m+(-1)^mf_l^{-m}\right]/\sqrt{2}\\
# f_{l-m}=i\left[f_l^m-(-1)^mf_l^{-m}\right]/\sqrt{2}\\
# f_l^m = (f_{lm}-if_{l-m})/\sqrt{2}\\
# f_l^{-m} = (-1)^m(f_{lm}+if_{l-m})/\sqrt{2}
# $$
# 
# for $m=0$:
# 
# $$
# f_{l0} = f_l^0
# $$
# 

# ## 2. LSQ solving SH coeffs for displacement field
# 
# Obviously, the decomposition is not satisfactory. It is not reasonable to assume the deformation is only on $r$-direction. In this section, we will try to optimize SH coeffs, so that the deformed shape is closest to the data. Notice that the integral of a spherical harmonic function on the sphere surface is:
# 
# $$
# \int_0^{2\pi}\!\int_0^{\pi}Y_l^m(\theta,\varphi)\sin\theta d\theta d\varphi = 4\pi\delta_{l0}\delta_{m0}
# $$
# 
# Therefore, only the $Y_0^0$ term controls the rigid body translation (constant). If we only impose higher mode spherical harmonics, there will be no rigid body motion.

# ### Develop the interpolation function for $u_r(\theta,\varphi)$ from data

# In[ ]:


f_interp = Ur_interp(Vp/r0, lmax=lJmax+20, plot_figure=plot_figure)
# Define shape reference
latsdeg, lonsdeg = pyshtools.expand.GLQGridCoord(lJmax)
lon0, lat0 = np.meshgrid(lonsdeg, latsdeg)
X0surf = np.stack([lat0, lon0], axis=-1)
X0 = GLQCartCoord(lJmax)


# ### Define the initial guess

# In[ ]:


import glob
filelist = glob.glob('AK_'+savename+'_??.npz')
print(filelist)
nfile = len(filelist)
if nfile > 0:
    u0_surf = np.load('AK_'+savename+'_%02d.npz'%(nfile-1))['u_surf_list'][-1, ...]
elif init_guess_type == 'ur-only':
    u0_surf = np.zeros_like(X0surf)

U0vec, aK, T0vec = usurf2vec(u0_surf, f_interp, lJmax)
if plot_figure:
    fig, ax = plt.subplots()
    ax.plot(np.abs(T0vec))
    plt.show()
    print('Displacement in Spherical Coordinates...')
    fig, ax = visSHVec(U0vec*r0, lmax_plot=lmax_plot, SphCoord=True, Complex=True,
                       n_vrange=(-1, 1), s_vrange=(0, 1),
                       config_quiver=(2, 3, 'k', 10), lonshift=180, figsize=(6,3))
    print('Traction in Spherical Coordinates...')
    fig, ax = visSHVec((T0vec)*mu0, lmax_plot=lmax_plot, SphCoord=True, Complex=True, 
                       n_vrange=(-100, 100), s_vrange=(0, 50),
                       config_quiver=(2, 3, 'k', 500), lonshift=180, figsize=(6,3))


# ### Define the traction free region based on the initial guess

# In[ ]:


# Define weights and traction free region
if dilated == '_softedge':
    isTfv = calculateTfv(U0vec, lJmax, Vp/r0, mask, lat_weight=True)
else:
    isTfv = calculateTfv(U0vec, lJmax, Vp/r0, Tfv, lat_weight=True)
print(isTfv.shape)


# penalty function:
# $$
# p=|\mathbf{Q}\hat{T}^{(K)}|^2=(\mathbf{QCD}^{-1}\mathbf{S}U_{mesh})^H\mathbf{QCD}^{-1}\mathbf{S}U_{mesh}
# $$

# In[ ]:


lmax = lJmax
ldamp_hi = lmax; ldamp_lo = lmax - 5;
lv, _ = LM_list(lmax); lv_ones = np.ones_like(lv);
lv_lim = np.minimum(np.maximum(lv, ldamp_lo), ldamp_hi)
ldamp = (np.maximum(lv_lim-ldamp_lo, 0) / (ldamp_hi - ldamp_lo))**1
Q = spm.csr_matrix(np.diag(np.tile(ldamp, 3)).astype(np.complex))
#print(Q.shape, CDmat.shape)
#plt.plot(ldamp)
#plt.show()


# In[ ]:


# Calculating L, S, P matrices for Jacobian evaluation
tic = time.time()
Lmat = genLmat(lJmax, Cmat=Cmat, Dmat=Dmat)
print('Time for generating L matrix: %.2fs'%(time.time() - tic))
tic = time.time()
Smat = genSmat(lJmax, Cmat=Cmat, Dmat=Dmat)
print('Time for generating S matrix: %.2fs'%(time.time() - tic))
Dinv = spm.linalg.inv(Dmat)
CDmat = Cmat.dot(Dinv)
tic = time.time()
CDSmat = np.asmatrix(CDmat.dot(Smat))
SHCDS = np.asmatrix(Smat).H.dot(CDSmat)
print('Time for generating S^HCD^{-1}S matrix: %.2fs'%(time.time() - tic))
tic = time.time()
QCDSmat = Q.dot(CDSmat)
QCDSHQCDS = QCDSmat.H.dot(QCDSmat).real
print('Time for generating QCDSHQCDS matrix: %.2fs'%(time.time() - tic))
tic = time.time()
P = np.diag(np.stack([isTfv]*3, axis=-1).flatten())
TresJac = 2*np.asmatrix(np.dot(np.dot(Lmat.T, P), Lmat))/(lJmax+1)/(lJmax*2+1) 
EelJac  = 2*np.pi*(SHCDS+SHCDS.H)
penJac  = QCDSHQCDS+QCDSHQCDS.H
print('time of matrix build:', time.time() - tic)


# In[ ]:


T0dist = Tvec2Tres(T0vec, lJmax, isTfv=isTfv, norm_order=myord)
E0el = np.vdot(U0vec, T0vec).real*2*np.pi
pen0 = np.vdot(Q.dot(T0vec), Q.dot(T0vec)).real
print('Traction residual: %.4e Pa'%(np.sqrt(T0dist)*mu0))
print('Elastic energy: %.4e pJ'%(E0el*(r0/1e6)**3*mu0*1e12))
print('funval: %.4e %.4e %.4e'%(T0dist, E0el, pen0))


# In[ ]:


target_args = (f_interp, lJmax, mybeta, myord, X0surf, X0, isTfv, Cmat, Dmat, 
               mu0, nu0, np.array([1]), np.array([1]), eps, '2-point', (TresJac, EelJac, penJac), mygamma)

def print_iter(xk):
    Uvec, aK, Tvec = usurf2vec(xk, f_interp, lJmax, X0surf=X0surf, X0=X0, Cmat=Cmat, Dmat=Dmat)
    Tdist = Tvec2Tres(Tvec, lJmax, isTfv=isTfv, norm_order=myord)
    Eel = np.vdot(Uvec, Tvec).real*2*np.pi
    pen = np.vdot(Q.dot(Tvec), Q.dot(Tvec)).real
    dr  = usurf2dr2(xk, *target_args)
    print('%13.4ePa%13.4epJ%13.4e%13.4e%13.4e%13.4e'%(np.sqrt(Tdist)*mu0, Eel*(r0/1e6)**3*mu0*1e12, Tdist, Eel, pen, dr))

u_surf = u0_surf.flatten().copy()
u_surf_list = [u_surf, ]
loss_count = 0
tic_start = time.time()
for i in range(N_period):
    print('Period %4d  Tr'%i, ' '*10, 'Eel', ' '*9, 'f0',' '*9, 'f1', ' '*9, 'f2', ' '*9, 'f')
    tic = time.time()
    u_res = minimize(usurf2dr2, u_surf.flatten(), args=target_args, jac=grad_usurf2dr2,
                     method = minimizer, options=minimizer_config, callback=print_iter)
    print('Iteration Time: %.2fs'%(time.time() - tic))
    u_surf = u_res.x.copy()
    u_surf_list.append(u_surf)
    if u_res.success:
        break
    if u_res.message == 'Warning: Desired error not necessarily achieved due to precision loss.':
        loss_count += 1
        if loss_count > 10:
            break
    if i%10 == 9:
        print('update isTfv at step %d'%i)
        Uvec, aK, Tvec = usurf2vec(u_surf, f_interp, lJmax, X0surf=X0surf, X0=X0, Cmat=Cmat, Dmat=Dmat)
        if dilated == '_softedge':
            isTfv = calculateTfv(Uvec, lJmax, Vp/r0, mask, lat_weight=True)
        else:
            isTfv = calculateTfv(Uvec, lJmax, Vp/r0, Tfv, lat_weight=True)
        P = np.diag(np.stack([isTfv]*3, axis=-1).flatten())
        TresJac = 2*np.asmatrix(np.dot(np.dot(Lmat.T, P), Lmat))/(lJmax+1)/(lJmax*2+1)
        target_args = (f_interp, lJmax, mybeta, myord, X0surf, X0, isTfv, Cmat, Dmat, 
               mu0, nu0, np.array([1]), np.array([1]), eps, '2-point', (TresJac, EelJac, penJac), mygamma)
print('Total Wall Time: %.2fs'%(time.time() - tic_start))


# In[ ]:


print(u_res)
u_surf = u_res.x

Usurfvec, aK, Tsurfvec = usurf2vec(u_surf, f_interp, lJmax, X0surf=X0surf, X0=X0, Cmat=Cmat, Dmat=Dmat)
Tsurfdist = Tvec2Tres(Tsurfvec, lJmax, isTfv=isTfv, norm_order=myord)
Eelsurf = np.vdot(Usurfvec, Tsurfvec).real*2*np.pi
print('Traction residual: %.4e Pa'%(np.sqrt(Tsurfdist)*mu0))
print('Elastic energy: %.4e pJ'%(Eelsurf*(r0/1e6)**3*mu0*1e12))
print('funval: %.4e %.4e'%(Tsurfdist, Eelsurf))

if plot_figure:
    fig, ax = plt.subplots()
    # ax.plot(np.abs(T_usr_vec))
    ax.plot(np.abs(Tsurfvec))
    plt.show()

    print('Displacement in Spherical Coordinates...')
    fig, ax = visSHVec(Usurfvec*r0, lmax_plot=lmax_plot, SphCoord=True, Complex=True, #s_vrange=(0,0.01),
                       config_quiver=(2, 3, 'k', 10), lonshift=180, figsize=(6,3))
    print('Traction in Spherical Coordinates...')
    fig, ax = visSHVec((Tsurfvec)*mu0, lmax_plot=lmax_plot, SphCoord=True, Complex=True, 
                       n_vrange=(-100, 100), s_vrange=(0, 50),
                       config_quiver=(2, 3, 'k', 500), lonshift=180, figsize=(6,3))


# In[ ]:


# %matplotlib notebook
if plot_figure:
    # umesh_fine_scaled = SHVec2mesh(U0vec, lmax=lmax_plot, SphCoord=False, Complex=True)
    # tmesh_fine = SHVec2mesh(T0vec*mu0, lmax=lmax_plot, SphCoord=False, Complex=True)
    umesh_fine_scaled = SHVec2mesh(Usurfvec, lmax=lmax_plot, SphCoord=False, Complex=True)
    tmesh_fine = SHVec2mesh(Tsurfvec*mu0, lmax=lmax_plot, SphCoord=False, Complex=True)
    print('Visualize the shape in 3D...')
    fig, ax = visSH3d(umesh_fine_scaled, cmesh=tmesh_fine, 
                      r0=r0, show=False, vmin=-100, vmax=100,
                      elevation=0, azimuth=0, 
                      surface=True, figsize=(6,6))

    # reference data
    nTfv = np.logical_not(Tfv)
    #ax.scatter3D(Vp[Tfv, 0], Vp[Tfv, 1], Vp[Tfv, 2], marker='^', s=10)
    #ax.scatter3D(Vp[nTfv, 0], Vp[nTfv, 1], Vp[nTfv, 2], marker='^', s=10)
    ax.set_title('Front view')
    plt.show()


# In[ ]:


import glob
filelist = glob.glob('AK_'+savename+'_??.npz')
nfile = len(filelist)
np.savez('AK_'+savename+'_%02d.npz'%(nfile), 
         AK_iter=aK, u_surf_list=np.stack(u_surf_list, axis=0), 
         beta=mybeta, N=N_period)

