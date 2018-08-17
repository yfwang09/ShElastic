import numpy as _np
import scipy as _sp
import scipy.sparse as _spm
import matplotlib.pyplot as _plt
import pyshtools as _psh

# routines for coordinate transformation
def CartCoord_to_SphCoord(X, Y, Z):
    # translate Cartesian coordinates into spherical coordinates
    # make sure X, Y, Z have same dimensions
    R = _np.sqrt(X**2 + Y**2 + Z**2)
    THETA = _np.arccos(Z/R)
    PHI = _np.arctan2(Y, X)
    PHI[PHI < 0] += 2*_np.pi
    return (R, THETA, PHI)
def SphCoord_to_CartCoord(R, THETA, PHI):
    # translate spherical coordinates into Cartesian coordinates
    # make sure R, THETA, PHI have same dimensions
    Z = R * _np.cos(THETA)
    X = R * _np.sin(THETA) * _np.cos(PHI)
    Y = R * _np.sin(THETA) * _np.sin(PHI)
    return (X, Y, Z)
def TransMat(t_mesh=None, p_mesh=None, lJmax=None):
    if (t_mesh is None) and (p_mesh is None) and (lJmax is not None):
        latglq, longlq = _psh.expand.GLQGridCoord(lJmax)
        theta = _np.radians(90-latglq)
        phi = _np.radians(longlq)
        p_mesh, t_mesh = _np.meshgrid(phi, theta)
    Q = _np.zeros(p_mesh.shape+(3,3))
    Q[...,0,0] = _np.sin(t_mesh)*_np.cos(p_mesh)
    Q[...,0,1] = _np.sin(t_mesh)*_np.sin(p_mesh)
    Q[...,0,2] = _np.cos(t_mesh)
    Q[...,1,0] = _np.cos(t_mesh)*_np.cos(p_mesh)
    Q[...,1,1] = _np.cos(t_mesh)*_np.sin(p_mesh)
    Q[...,1,2] =-_np.sin(t_mesh)
    Q[...,2,0] =-_np.sin(p_mesh)
    Q[...,2,1] = _np.cos(p_mesh) 
    return Q

# routines for index transformation
def lm2L(l, m):
    return l**2 + (l + m)
def L2lm(L):
    l = _np.floor(_np.sqrt(L)).astype(_np.int)
    m = L - l**2 - l
    return (l, m)
def LM_list(lmax):
    return L2lm(_np.arange((lmax+1)**2, dtype=_np.int))
def ILM_list(lmax):
    l, m = LM_list(lmax)
    i = (m < 0).astype(_np.int)
    return (i, l, _np.abs(m).astype(_np.int))

###  lmk2K, K2lmk ###
# Given lmax, transform between (l,m,k) degrees and 1d index K
def lmk2K(l, m, k, lmax):
    n = (lmax+1)**2
    return lm2L(l, m) + k*n
def K2lmk(K, lmax):
    n = (lmax+1)**2
    k = _np.floor(K/n).astype(_np.int)
    l, m = L2lm(K%n)
    return l, m, k

def l_coeffs(lmax):
    # This function to create the l degrees corresponding to the coefficient matrix shape
    l_list = _np.arange(lmax + 1)
    return _np.broadcast_to(l_list[_np.newaxis,:,_np.newaxis], (2, lmax+1, lmax+1))
def m_coeffs(lmax):
    # This function to create the m degrees corresponding to the coefficient matrix shape
    m_list = _np.arange(lmax + 1)
    return _np.broadcast_to(m_list[_np.newaxis,_np.newaxis, :], (2, lmax+1, lmax+1))

# routines for matrix shape transformation
def SHCilmToVector(cilm, lmax=None):
    clmax = cilm.shape[1] - 1
    if lmax is None:
        lmax = clmax
    i, l, m = ILM_list(clmax)
    vec = _np.zeros((lmax+1)**2, dtype=cilm.dtype)
    vec[:(clmax+1)**2] = cilm[i,l,m]
    return vec
def SHVectorToCilm(vec, lmax=None):
    # check whether vec has a valid length
    if lmax is None:
        lmax = _np.around(_np.sqrt(len(vec))).astype(_np.int) - 1
        if (lmax+1)**2 != len(vec):
            print('invalid length')
            return
    Lmax = (lmax+1)**2
    cilm = _np.zeros((2, lmax+1, lmax+1), dtype=vec.dtype)
    i, l, m = ILM_list(lmax)
    cilm[i, l, m] = vec[:Lmax]
    return cilm

def sparse_mode(M_K, lmax=None, etol=1e-8):
    clmax = M_K.shape[1] - 1
    if lmax is None:
        lmax = clmax
    MK_mod = M_K.reshape(*M_K.shape[:3], -1)
    d = MK_mod.shape[-1]
    mode = _np.empty((d, (lmax+1)**2), dtype=M_K.dtype)
    for i in range(d):
        mode[i, :] = SHCilmToVector(MK_mod[..., i], lmax=lmax)
    mode = mode.reshape((mode.size, 1))
    SPM = _spm.lil_matrix((mode.size, 1), dtype=mode.dtype)
    filter_idx = _np.abs(mode)>etol
    SPM[filter_idx] = mode[filter_idx]
    return SPM

def dense_mode(mode, d, lmax):
    Lmax = (lmax+1)**2
    if d==0:
        M = SHVectorToCilm(mode)
    elif d==1:
        M = []
        for i in range(3):
            M.append(SHVectorToCilm(mode[Lmax*i:Lmax*(i+1)]))
        M = _np.stack(M, axis=-1)
    elif d==2:
        M = []
        for i in range(3):
            Mi = []
            for j in range(3):
                Mi.append(SHVectorToCilm(mode[Lmax*(i*3+j):Lmax*(i*3+j+1)]))
            M.append(_np.stack(Mi, axis=-1))
        M = _np.stack(M, axis=-2)

    return M

########### visualization

def plotfv(fv, figsize=(10,5), colorbar=True, show=True, vrange=None, cmap='viridis', lonshift=0):
    """
    Initialize the class instance from an input array.

    Usage
    -----
    fig, ax = SHUtil.plotfv(fv, [figsize, colorbar, show, vrange, cmap, lonshift])

    Returns
    -------
    fig, ax : matplotlib figure and axis instances

    Parameters
    ----------
    fv : ndarray, shape (nlat, nlon)
        2-D numpy array of the gridded data, where nlat and nlon are the
        number of latitudinal and longitudinal bands, respectively.
    figsize : size of the figure, optional, default = (10, 5)
    colorbar : bool, optional, default = True
        If True (default), plot the colorbar along with the map
    show : bool, optional, default = True
        If True, plot the image to the screen.
    vrange : 2-element tuple, optional
        The range of the colormap, default is (min, max)
    cmap : string, default = 'viridis'
        Name of the colormap, see matplotlib
    lonshift : float, in degree, default = 0
        Shift the map along longitude direction by lonshift degree
    """
    if lonshift is not None:
        fv = _np.roll(fv, _np.round(fv.shape[1]*lonshift/360).astype(_np.int), axis=1)
    if vrange is None:
        fmax, fmin = fv.max(), fv.min()
    else:
        fmin, fmax = vrange
    fcolors = (fv - fmin)/(fmax - fmin)    # normalize the values into range [0, 1]
    fcolors[fcolors<0]=0
    fig0 = _plt.figure(figsize=figsize)
    ax0 = fig0.add_subplot(111)
    cax0 = ax0.imshow(fv, extent=(0, 360, -90, 90), cmap=cmap, vmin=fmin, vmax=fmax)
    ax0.set(xlabel='longitude', ylabel='latitude')
    if colorbar:
        fig0.colorbar(cax0)
    if show:
        _plt.show()
    return fig0, ax0

########### savemode, loadmode: save and load vectorized spherical harmonic mode (l, m, value)

def loadmode(mode, d, lmax):
    M = _psh.SHCoeffs.from_zeros(lmax, kind='complex')
    if d==0:
        M.set_coeffs(mode['c'], mode['l'], mode['m'])
    elif d==1:
        M = _np.zeros(M.to_array().shape+(3,), dtype=_np.complex)
        for i in range(3):
            Mi = _psh.SHCoeffs.from_zeros(lmax, kind='complex')
            colidx = (mode['i'] == i)
            ls = mode['l'][colidx].astype(_np.int)
            ms = mode['m'][colidx].astype(_np.int)
            values = mode['c'][colidx].astype(_np.complex)
            Mi.set_coeffs(values, ls, ms)
            M[..., i] = Mi.to_array()
    elif d==2:
        M = _np.zeros(M.to_array().shape+(3,3), dtype=_np.complex)
        for i in range(3):
            for j in range(3):
                Mij = _psh.SHCoeffs.from_zeros(lmax, kind='complex')
                colidx = _np.logical_and(mode['i'] == i, mode['j'] == j)
                ls = mode['l'][colidx].astype(_np.int)
                ms = mode['m'][colidx].astype(_np.int)
                values = mode['c'][colidx].astype(_np.complex)
                Mij.set_coeffs(values, ls, ms)
                M[...,i,j] = Mij.to_array()
    else:
        print('loadmode: invalid dimension d=', d)
        return 0

    return M
