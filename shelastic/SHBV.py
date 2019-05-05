import numpy as _np
import scipy.sparse as _spm
from scipy.special import sph_harm
import matplotlib.pyplot as _plt
from shelastic.shelastic import calUmode, calSmode
from shelastic.shutil import SHCilmToVector, CartCoord_to_SphCoord, K2lmk
from time import time

def generate_submat(modes, mu, nu, lmax_col, lmax_row, shtype='irr', verbose=False):
    '''Obtain the sub-matrix of spherical harmonic modes
    
    Parameters
    ----------
    modes : dict
        one of Umodes, Smodes, or Tmodes created by generate_modes
    mu,nu : float
        shear modulus and Poisson's ratio
    lmax_col,lmax_row : int
        lmax for column number (K), and row number (J)
    shtype : string, ['irr' or 'reg']
        'irr' represents irregular spherical harmonics for spherical void
        'reg' represents regular spherical harmonics for solid sphere
    verbose: bool
        if True, print sub-matrix size
    
    Returns
    -------
    complex lil_matrix, dimension (kJ*(lmax_row+1)^2, kK*(lmax_col+1)^2)
        sub-matrix of spherical harmonic mode. kK = 3 for 3 directions (i,j,k)
        kJ = 3 for vector (U and T), 9 for tensor (S)

    See Also
    --------
    generate_modes : generate spherical harmonic modes

    '''
    if 'U0'+shtype in modes.keys():   # obtain displacement mode full matrix
        U1 = modes['U1'+shtype]; U0 = modes['U0'+shtype];
        fullmat = calUmode((U1, U0), mu, nu)
        kK, kJ = 3, 3
    elif 'S0'+shtype in modes.keys(): # obtain stress mode full matrix
        S1 = modes['S1'+shtype]; S2 = modes['S2'+shtype];
        S3 = modes['S3'+shtype]; S0 = modes['S0'+shtype];
        fullmat = calSmode((S1, S2, S3, S0), mu, nu)
        kK, kJ = 3, 9
    elif 'T0'+shtype in modes.keys(): # obtain traction mode full matrix
        T1 = modes['T1'+shtype]; T2 = modes['T2'+shtype];
        T3 = modes['T3'+shtype]; T0 = modes['T0'+shtype];
        fullmat = calSmode((T1, T2, T3, T0), mu, nu)
        kK, kJ = 3, 3
    else:
        print('input is not SH mode created by generate_modes()')
        return -1

    M, N = fullmat.shape
    lKfull = _np.sqrt(N/kK).astype(_np.int)-1;
    lJfull = _np.sqrt(M/kJ).astype(_np.int)-1;

    LJfull = (lJfull+1)**2; LKfull = (lKfull+1)**2;
    LJmax = (lmax_row+1)**2;LKmax = (lmax_col+1)**2;
    full_row = kJ*LJfull;   full_col = kK*LKfull;
    size_row = kJ*LJmax;    size_col = kK*LKmax;
    if verbose:
        print('Integrating modes to a matrix')
        print(size_row, size_col)

    # Combine sparse matrix using block matrices
    mat_blocks = [[None for _ in range(kK)] for _ in range(kJ)]
    for kj in range(kJ):
        for kk in range(kK):
            r1 = kj*LJmax; r2 = r1+LJmax; c1 = kk*LKmax; c2 = c1+LKmax;
            R1 = kj*LJfull;R2 = R1+LJmax; C1 = kk*LKfull;C2 = C1+LKmax;
            mat_blocks[kj][kk] = fullmat[R1:R2, C1:C2]
    submat = _spm.bmat(mat_blocks)

    return submat

def print_SH_mode(vec, m_dir=3, etol=1e-8, verbose=True):
    # vec is a *complex* spherical harmonic vector
    # with m different directions
    idx_type = [('index', '<i4', 3), ('coeff', _np.complex_)]
    idx_mode = _np.array([], dtype=idx_type)
    lmax = _np.int(_np.sqrt(vec.size / m_dir)) - 1
    idx = [K2lmk(i, lmax) for i in range(vec.size)]
    for i in range(vec.size):
        if (_np.abs(_np.real(vec[i])) > etol or \
            _np.abs(_np.imag(vec[i])) > etol):
            if verbose:
                print('index:',i, idx[i], 'coeff:', vec[i])
            new_idx = _np.array((idx[i], vec[i]), dtype=idx_type)
            idx_mode = _np.append(idx_mode, new_idx)
    return idx_mode

def fast_displacement_solution(aK, X, Y, Z, Umodes, lKmax=50, lJmax=53, shtype='irr', verbose=True):
    R, THETA, PHI = CartCoord_to_SphCoord(X, Y, Z)
    disp = _np.zeros(X.shape+(3,lKmax+1), dtype=_np.complex)
    lats = _np.pi/2-THETA; lat_d = _np.rad2deg(lats); 
    lons = PHI;            lon_d = _np.rad2deg(lons); 
    ## Multiply the modes U_K by a_K
    M, N = Umodes.shape;
    UK = Umodes*_spm.diags(aK, shape=(aK.size, aK.size))
    ## Combine the columns with same l
    mode_l, mode_m, mode_k = K2lmk(_np.arange(N, dtype=_np.int), lmax=lKmax)
    lmodes = _np.arange(lKmax+1, dtype=_np.int)
    map_L = _spm.csr_matrix(mode_l[:,_np.newaxis] == lmodes, dtype=_np.int)
    Ul = UK.dot(map_L).tocsc()
    for lmode in lmodes:
        Js = Ul[:,lmode].nonzero()[0]
        if Js.size > 0:
            ls, ms, ks = K2lmk(Js, lmax=lJmax)
            us = ks%3; # us = ((ks-vs)/3).astype(_np.int)
            for u in range(3):
                modes_u = (u == us)
                values = Ul[Js,lmode].toarray().flatten()[modes_u]
                disp_u = sph_harm(ms[modes_u], ls[modes_u], PHI[...,_np.newaxis], THETA[...,_np.newaxis])
                disp[...,u,lmode] = _np.sum(disp_u*((-1.)**ms[modes_u])*values*_np.sqrt(4*_np.pi), axis=-1)
    if shtype == 'irr':
        disp /= R[...,_np.newaxis,_np.newaxis]**(lmodes+1)
    else:
        disp *= R[...,_np.newaxis,_np.newaxis]**(lmodes)
    return disp.sum(axis=-1)

def fast_stress_solution(aK, X, Y, Z, Smodes, lKmax=50, lJmax=53, shtype='irr', verbose=True):
    R, THETA, PHI = CartCoord_to_SphCoord(X, Y, Z)
    sigma = _np.zeros(X.shape+(3,3,lKmax+1), dtype=_np.complex)
    lats = _np.pi/2-THETA; lat_d = _np.rad2deg(lats); 
    lons = PHI;            lon_d = _np.rad2deg(lons); 
    ## Multiply the modes S_K by a_K
    M, N = Smodes.shape;
    SK = Smodes*_spm.diags(aK, shape=(aK.size, aK.size))
    ## Combine the columns with same l
    mode_l, mode_m, mode_k = K2lmk(_np.arange(N, dtype=_np.int), lmax=lKmax)
    lmodes = _np.arange(lKmax+1, dtype=_np.int)
    map_L = _spm.csr_matrix(mode_l[:,_np.newaxis] == lmodes, dtype=_np.int)
    Sl = SK.dot(map_L).tocsc()
    for lmode in lmodes:
        Js = Sl[:,lmode].nonzero()[0]
        if Js.size > 0:
            ls, ms, ks = K2lmk(Js, lmax=lJmax)
            vs = ks%3; us = ((ks-vs)/3).astype(_np.int)
            for u in range(3):
                for v in range(3):
                    modes_uv = _np.logical_and(u == us, v == vs)
                    values = Sl[Js,lmode].toarray().flatten()[modes_uv]
                    sigma_uv = sph_harm(ms[modes_uv], ls[modes_uv], PHI[...,_np.newaxis], THETA[...,_np.newaxis])
                    sigma[...,u,v,lmode] = _np.sum(sigma_uv*((-1.)**ms[modes_uv])*values*_np.sqrt(4*_np.pi), axis=-1)
    if shtype == 'irr':
        sigma /= R[...,_np.newaxis,_np.newaxis,_np.newaxis]**(lmodes+2)
    else:
        sigma *= R[...,_np.newaxis,_np.newaxis,_np.newaxis]**(lmodes-1)
    return sigma.sum(axis=-1).real

def fast_energy_solution(A_sol, Dmat, Cmat, Ac_sol=None, Dcmat=None, Ccmat=None):
    Uvec = Dmat.dot(A_sol)
    Tvec = Cmat.dot(A_sol)
    if (Dcmat is not None) and (Ccmat is not None):
        Ucvec = Dcmat.dot(Ac_sol)
        Tcvec = Ccmat.dot(Ac_sol)
        Ecore = 2*_np.pi*_np.vdot(Ucvec, Tcvec).real
    else:
        Ecore = 0

    return -Ecore + 2*_np.pi*_np.vdot(Uvec, Tvec).real

def visualize_Cmat(Csub, precision=1e-8, m_max=3):
    _plt.spy(Csub, precision=precision, markersize = 3)
    mode_sub = _np.int(_np.sqrt(Csub.shape[1]/m_max))-1
    lmax_sub = _np.int(_np.sqrt(Csub.shape[0]/3))-1
    print(mode_sub, lmax_sub)
    lsy = _np.arange(0, lmax_sub+1)
    lsx = _np.arange(0, mode_sub+1)

    x_range = [0, Csub.shape[1]]
    y_range = [0, Csub.shape[0]]
    # traction direction dividing line:
    for i in range(1, 3+1):
        y_divide = y_range[1]*i/3
        _plt.plot(x_range, _np.ones(2)*y_divide-0.5)
        y_text = y_divide - y_range[1]/m_max/2
        #_plt.text(-mode_sub-5, y_text, '$T_'+str(i)+'$', fontsize=24)
    # mode dividing line:
    for i in range(1, m_max+1):
        x_divide = x_range[1]*i/m_max
        _plt.plot(_np.ones(2)*x_divide-0.5, y_range)
        x_text = x_divide - x_range[1]/3/2 - 1
        #_plt.text(x_text, -lmax_sub, '$\\psi_'+str(i)+'$', fontsize=24)

    # ticks for different traction directions
    ticks_y = _np.array([])
    for i in range(3):
        ticks_Ti = i*(lmax_sub+1)**2+lsy**2
        ticks_y = _np.hstack((ticks_y, ticks_Ti))
    _plt.yticks(ticks_y, _np.tile(lsy, 3))
    # ticks for different modes
    ticks_x = _np.array([])
    for i in range(m_max):
        ticks_Psi_i = i*(mode_sub+1)**2+lsx**2
        ticks_x = _np.hstack((ticks_x, ticks_Psi_i))
    _plt.xticks(ticks_x, _np.tile(lsx, m_max))
    if m_max == 4:
        _plt.title('Solution A+B', fontsize=24)
    else:
        _plt.title('Solution B', fontsize=24)
