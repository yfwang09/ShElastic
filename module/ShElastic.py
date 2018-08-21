import numpy as _np
import scipy as _sp
import scipy.sparse as _spm
from scipy.io import savemat
import matplotlib.pyplot as _plt
import pyshtools as _psh
import sys, os
sys.path.append('../module/')
from SHUtil import SHCilmToVector, sparse_mode, lmk2K
from SHGrad import VSH1, VSH2

def genUmode(l, m, k, shtype='irr', lmax=None):
    if shtype == 'irr':
        C_vsh1 = -l
        if lmax is None:
            lmax = l + 3
    elif shtype == 'reg':
        C_vsh1 = l+1
        if lmax is None:
            lmax = l + 3
    else:
        print('U_mode: invalid shtype (irr, reg)')
        return 0
    Ylm = _psh.SHCoeffs.from_zeros(lmax=lmax, kind='complex')
    Ylm.set_coeffs(1.0, l, m)
    U_nu = _np.zeros(Ylm.to_array().shape+(3,), dtype=_np.complex)
    U_nu[..., k] = Ylm.to_array()
    rdotPsi = VSH1(Ylm.to_array())[...,k]
    U_0 = C_vsh1*VSH1(rdotPsi) + VSH2(rdotPsi)
    return (U_nu, U_0)

def calUmode(Umode, mu, nu, c_omega=1.0):
    U_nu, U_0 = Umode
    C_nu = -4*(1-nu)
    return c_omega*(U_nu * C_nu + U_0)/2.0/mu

def genGradU(U, c1, c2):
    gradu = [None for _ in range(3)]
    for i in range(3):
        gradu[i] = c1 * VSH1(U[...,i]) + c2 * VSH2(U[...,i])
    return _np.stack(gradu, axis=-1)

def genSmode(l, m, k, shtype='irr', lmax=None, returnU=False):
    if shtype == 'irr':
        c1 = -(l+1)
        if lmax is None:
            lmax = l + 3
    elif shtype == 'reg':
        c1 = l
        if lmax is None:
            lmax = l + 3
    else: # if (shtype != 'irr') and (shtype != 'reg'):
        print('genGradU: invalid shtype (irr, reg)')
        return 0
    U_nu, U_0 = genUmode(l, m, k, shtype=shtype, lmax=lmax)
    gradU_nu = genGradU(U_nu, c1, 1.0)
    gradU_0 = genGradU(U_0, c1, 1.0)
    ukk_nu = _np.trace(gradU_nu, axis1=-1, axis2=-2)
    ukk_0 = _np.trace(gradU_0, axis1=-1, axis2=-2)
    S_nu1 = ukk_nu[...,_np.newaxis,_np.newaxis]*_np.eye(3)
    S_nu2 = 0.5*(gradU_nu+_np.swapaxes(gradU_nu, -1, -2))
    S_nu3 = ukk_0[...,_np.newaxis,_np.newaxis] *_np.eye(3)
    S_0 = 0.5*(gradU_0+_np.swapaxes(gradU_0, -1, -2))
    if returnU:
        return (U_nu, U_0, S_nu1, S_nu2, S_nu3, S_0)
    else:
        return (S_nu1, S_nu2, S_nu3, S_0)

def calSmode(Smode, mu, nu, c_omega=1.0):
    S_nu1, S_nu2, S_nu3, S_0 = Smode
    c_nu1 = -4*(1-nu)
    c_nu2 = nu/(1-2*nu)
    return c_omega*(c_nu1*c_nu2*S_nu1 + c_nu1*S_nu2 + c_nu2*S_nu3 + S_0)

def calTmode(S_K):
    T_K = _np.zeros(S_K.shape[:-1], dtype=_np.complex)
    for i in range(3):
        T_K[...,i] = VSH1(S_K[...,i,0])[...,0]+VSH1(S_K[...,i,1])[...,1]+VSH1(S_K[...,i,2])[...,2]
    return T_K

def calEmode(U_K, T_K):
    pass

def generate_modes(lmax, etol=1e-8, save_lmax=50):
    if save_lmax < lmax + 3:
        save_lmax = lmax + 3
    M = 3*(save_lmax+1)**2
    N = 3*(lmax+1)**2
    U1irr = _spm.lil_matrix((M,N), dtype=_np.complex); U1reg = _spm.lil_matrix((M,N), dtype=_np.complex);
    U0irr = _spm.lil_matrix((M,N), dtype=_np.complex); U0reg = _spm.lil_matrix((M,N), dtype=_np.complex);
    S1irr = _spm.lil_matrix((3*M,N), dtype=_np.complex); S1reg = _spm.lil_matrix((3*M,N), dtype=_np.complex);
    S2irr = _spm.lil_matrix((3*M,N), dtype=_np.complex); S2reg = _spm.lil_matrix((3*M,N), dtype=_np.complex);
    S3irr = _spm.lil_matrix((3*M,N), dtype=_np.complex); S3reg = _spm.lil_matrix((3*M,N), dtype=_np.complex);
    S0irr = _spm.lil_matrix((3*M,N), dtype=_np.complex); S0reg = _spm.lil_matrix((3*M,N), dtype=_np.complex);
    T1irr = _spm.lil_matrix((M,N), dtype=_np.complex); T1reg = _spm.lil_matrix((M,N), dtype=_np.complex);
    T2irr = _spm.lil_matrix((M,N), dtype=_np.complex); T2reg = _spm.lil_matrix((M,N), dtype=_np.complex);
    T3irr = _spm.lil_matrix((M,N), dtype=_np.complex); T3reg = _spm.lil_matrix((M,N), dtype=_np.complex);
    T0irr = _spm.lil_matrix((M,N), dtype=_np.complex); T0reg = _spm.lil_matrix((M,N), dtype=_np.complex);
    for l in range(lmax+1):
        for m in range(-l, l+1):
            for k in range(3):
                K = lmk2K(l,m,k,lmax=lmax)
                print(l, m, k)
                Unu, U0, Snu1, Snu2, Snu3, S0 = genSmode(l, m, k, shtype='irr',returnU=True)
                Tnu1 = calTmode(Snu1); Tnu2 = calTmode(Snu2); Tnu3 = calTmode(Snu3); T0 = calTmode(S0)
                print('irregular solid harmonic modes...')
                U1irr[:, K] = sparse_mode(Unu, lmax=save_lmax, etol=etol); 
                U0irr[:, K] = sparse_mode(U0, lmax=save_lmax, etol=etol);
                S1irr[:, K] = sparse_mode(Snu1, lmax=save_lmax, etol=etol); 
                S2irr[:, K] = sparse_mode(Snu2, lmax=save_lmax, etol=etol);
                S3irr[:, K] = sparse_mode(Snu3, lmax=save_lmax, etol=etol); 
                S0irr[:, K] = sparse_mode(S0, lmax=save_lmax, etol=etol);
                T1irr[:, K] = sparse_mode(Tnu1, lmax=save_lmax, etol=etol); 
                T2irr[:, K] = sparse_mode(Tnu2, lmax=save_lmax, etol=etol);
                T3irr[:, K] = sparse_mode(Tnu3, lmax=save_lmax, etol=etol); 
                T0irr[:, K] = sparse_mode(T0, lmax=save_lmax, etol=etol);
                Unu, U0, Snu1, Snu2, Snu3, S0 = genSmode(l, m, k, shtype='reg',returnU=True)
                Tnu1 = -calTmode(Snu1); Tnu2 = -calTmode(Snu2); Tnu3 = -calTmode(Snu3); T0 = -calTmode(S0)
                print('regular solid harmonic modes...')
                U1reg[:, K] = sparse_mode(Unu, lmax=save_lmax, etol=etol); 
                U0reg[:, K] = sparse_mode(U0, lmax=save_lmax, etol=etol);
                S1reg[:, K] = sparse_mode(Snu1, lmax=save_lmax, etol=etol); 
                S2reg[:, K] = sparse_mode(Snu2, lmax=save_lmax, etol=etol);
                S3reg[:, K] = sparse_mode(Snu3, lmax=save_lmax, etol=etol); 
                S0reg[:, K] = sparse_mode(S0, lmax=save_lmax, etol=etol);
                T1reg[:, K] = sparse_mode(Tnu1, lmax=save_lmax, etol=etol); 
                T2reg[:, K] = sparse_mode(Tnu2, lmax=save_lmax, etol=etol);
                T3reg[:, K] = sparse_mode(Tnu3, lmax=save_lmax, etol=etol); 
                T0reg[:, K] = sparse_mode(T0, lmax=save_lmax, etol=etol);
    savemat('Umodes.mat', {'U1irr': U1irr, 'U1reg': U1reg, 'U0irr': U0irr, 'U0reg': U0reg})
    savemat('Smodes.mat', {'S1irr': S1irr, 'S1reg': S1reg, 'S2irr': S2irr, 'S2reg': S2reg,
                           'S3irr': S3irr, 'S3reg': S3reg, 'S0irr': S0irr, 'S0reg': S0reg})
    savemat('Tmodes.mat', {'T1irr': T1irr, 'T1reg': T1reg, 'T2irr': T2irr, 'T2reg': T2reg,
                           'T3irr': T3irr, 'T3reg': T3reg, 'T0irr': T0irr, 'T0reg': T0reg})
    print('save success')