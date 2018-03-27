import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spm
from scipy.io import loadmat
import pyshtools
import sys, time
sys.path.append('../module/')
from SHUtil import SphCoord_to_CartCoord, CartCoord_to_SphCoord
from SHUtil import SHCilmToVector, lmk2K, K2lmk
from SHBV import fast_stress_solution, generate_submat, print_SH_mode
from ShElastic import calSmode

def img_stress(mu, nu, r0, b0, x01, x02, sigma_0):
    a = 1.0; b1 = b0/r0; x1 = x01/r0; x2 = x02/r0; sigma_inf = sigma_0;
    b = np.linalg.norm(b1, axis=1).mean()

    #### generate meshing on the void surface
    tic = time.time()
    Ngrid = sigma_0.shape[1]
    theta = np.arange(0, np.pi, np.pi/Ngrid)
    phi = np.arange(0, 2*np.pi, np.pi/Ngrid)
    THETA, PHI = np.meshgrid(theta, phi)

    Z = a*np.cos(THETA)
    Y = a*np.sin(THETA)*np.sin(PHI)
    X = a*np.sin(THETA)*np.cos(PHI)
    N = -np.stack([X/a, Y/a, Z/a], axis=-1)
    toc = time.time()
    print('generate meshing on the void surface', toc-tic)

    #### decompose traction boundary condition
    tic = time.time()

    T_inf = np.einsum('ijkl,ijl->ijk', sigma_0, N)
    T_usr_mesh = T_inf.astype(np.complex)

    T_usr_vec = np.array([])
    lJmax = np.int(Ngrid/2) - 1
    lKmax = lJmax - 3
    LJ = (lJmax+1)**2

    T_usr_vec = np.empty(3*LJ, dtype=np.complex)
    for k in range(3):
        T_usr_grid = pyshtools.SHGrid.from_array(T_usr_mesh[:,:,k].T, grid='DH')
        #T_usr_grid.plot3d(elevation=20, azimuth=45)
        T_usr_cilm = T_usr_grid.expand()
        T_usr_vec[LJ*k:LJ*(k+1)] = SHCilmToVector(T_usr_cilm.to_array(), lmax = lJmax)
    toc = time.time()
    print('decompose traction boundary condition', toc-tic)

    #### load traction mode matrix
    tic = time.time()

    shtype = 'irr'
    Tmodes = loadmat('Tmodes.mat')
    Tmodes = (Tmodes['T1'+shtype], Tmodes['T2'+shtype], Tmodes['T3'+shtype], Tmodes['T0'+shtype])
    print(Tmodes[0].shape, Tmodes[1].shape, Tmodes[2].shape, Tmodes[3].shape, nu, mu, lJmax, lKmax)
    fullCmat = calSmode(Tmodes, mu, nu)
    lJfull = 23; lKfull = 20;
    Cmat = generate_submat(mu, nu, fullCmat, lKmax, lJmax, lJfull=lJfull, lKfull=lKfull)

    toc = time.time()
    print('load traction mode matrix', toc-tic)

    #### determine the mode coefficients
    tic = time.time()
    A = spm.linalg.lsqr(Cmat, T_usr_vec.transpose())
    A_sol = A[0]
    index_sol = print_SH_mode(A_sol, m_dir=3, etol=1e-8, verbose=False)
    toc = time.time()
    print('determine the mode coefficients ( mode', len(index_sol), '):', toc-tic)

    #### evaluate stress solution and nodal force
    tic = time.time()
    nseg, m = b1.shape
    f1 = np.zeros(x1.shape)
    f2 = np.zeros(x2.shape)
    xi = x2 - x1
    norm_xi = np.tile(np.linalg.norm(xi, axis=1), (3, 1)).T
    xi = xi/norm_xi

    Smodes = loadmat('Smodes.mat')
    Smodes = (Smodes['S1'+shtype], Smodes['S2'+shtype], Smodes['S3'+shtype], Smodes['S0'+shtype])
    fullSmodes = calSmode(Smodes, mu, nu)
    Smodes = generate_submat(mu, nu, fullSmodes, lKmax, lJmax, kJ=9)

    sigma_1 = fast_stress_solution(A_sol, x1[:, 0], x1[:, 1], x1[:, 2], Smodes, lKmax, lJmax)

    f1 = np.cross(np.sum(sigma_1*b1[:,np.newaxis,:], axis=-1), xi)
    #print(f1)

    toc = time.time()
    print('evaluate stress solution and nodal force ( segs', nseg ,')', toc-tic)
    print(np.mean(f1[:,2])/b)

    return f1, f1
'''
#### load the image stress problem setting
tic = time.time()
data = loadmat('Susr1.mat')
sigma_inf = np.swapaxes(data['S'], 0, 1)
a = data['R'][0,0]
mu = data['MU'][0,0]
nu = data['NU'][0,0]
b1 = data['b1']
x1 = data['x1']
x2 = data['x2']

r0 = a; b0 = b1; x01 = x1; x02 = x2; sigma_0 = sigma_inf; # mu0 = mu; mu = 1.0;
#a = a/r0; b1 = b0/r0; x1 = x01/r0; x2 = x02/r0; sigma_inf = sigma_0;

#b = np.linalg.norm(b1, axis=1).mean()
toc = time.time()
print('load the image stress problem setting', toc-tic)

f = img_stress(mu, nu, r0, b0, x01, x02, sigma_0)
'''