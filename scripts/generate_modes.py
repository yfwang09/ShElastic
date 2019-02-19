import sys, os
sys.path.append('../module/')
from ShElastic import generate_modes

# generate modes for future use
lKfull = 5
lJfull = lKfull + 3
savepath = '../module/lmax%dmodes'%lKfull
os.makedirs(savepath, exist_ok=True)
generate_modes(lKfull, save_lmax=lJfull, path=savepath)

# test the generated modes

from ShElastic import genUmode, calUmode, genSmode, calTmode
from SHUtil import lmk2K, K2lmk
from SHUtil import sparse_mode, dense_mode
from SHBV import generate_submat
import numpy as np
from scipy.io import loadmat

refpath = savepath #'../module/lmax60modes'
Umodes = loadmat(os.path.join(refpath, 'Umodes.mat'))
Smodes = loadmat(os.path.join(refpath, 'Smodes.mat'))
Tmodes = loadmat(os.path.join(refpath, 'Tmodes.mat'))

lKmax = lKfull
lJmax = lJfull
ntest = 5

for itest in range(ntest):
    l = np.random.randint(lKfull + 1)
    m = np.random.randint(-l, l+1)
    k = np.random.randint(3)
    if np.random.randint(2):
        shtype = 'reg'
    else:
        shtype = 'irr'
    K = lmk2K(l, m, k, lmax=lKmax)
    print('############### Test %d ###############'%itest)
    print('(l, m, k) = (%d, %d, %d)'%(l,m,k), shtype, 'spherical harmonic bases')
    print('K =', K)

    # load U,S,T mode from file
    U0modes = generate_submat(1, 1/3, Umodes['U0'+shtype], lKmax, lJmax, lKfull=lKfull, lJfull=lJfull).tocsc()
    S0modes = generate_submat(1, 1/3, Smodes['S0'+shtype], lKmax, lJmax, lKfull=lKfull, lJfull=lJfull, kJ=9).tocsc()
    T0modes = generate_submat(1, 1/3, Tmodes['T0'+shtype], lKmax, lJmax, lKfull=lKfull, lJfull=lJfull).tocsc()
    U0_exam = U0modes[:, K]
    S0_exam = S0modes[:, K]
    T0_exam = T0modes[:, K]

    # generate U,S,T from scratch
    Unu, U0, Snu1, Snu2, Snu3, S0 = genSmode(l, m, k, shtype=shtype, returnU=True)
    if shtype == 'irr':
        Tnu1 = calTmode(Snu1)
        Tnu2 = calTmode(Snu2)
        Tnu3 = calTmode(Snu3)
        T0 = calTmode(S0)
    else:
        Tnu1 = -calTmode(Snu1)
        Tnu2 = -calTmode(Snu2)
        Tnu3 = -calTmode(Snu3)
        T0 = -calTmode(S0)

    U1sh = sparse_mode(Unu, lmax=lJmax); U0sh = sparse_mode(U0, lmax=lJmax);
    S1sh = sparse_mode(Snu1, lmax=lJmax); S2sh = sparse_mode(Snu2, lmax=lJmax);
    S3sh = sparse_mode(Snu3, lmax=lJmax); S0sh = sparse_mode(S0, lmax=lJmax);
    T1sh = sparse_mode(Tnu1, lmax=lJmax); T2sh = sparse_mode(Tnu2, lmax=lJmax);
    T3sh = sparse_mode(Tnu3, lmax=lJmax); T0sh = sparse_mode(T0, lmax=lJmax);

    # compare the saved mode and generated mode
    print('Test displacement:')
    print('diff (U0, refU0):', U0sh.shape, U0_exam.shape)
    diff_idx = (np.abs(U0sh-U0_exam)>1e-10).nonzero()
    if diff_idx[0].size == 0:
        print('no difference')
    else:
        print((U0sh-U0_exam)[diff_idx])

    print('Test stress:')
    print('diff (S0, refS0):', S0sh.shape, S0_exam.shape)
    diff_idx = (np.abs(S0sh-S0_exam)>1e-10).nonzero()
    if diff_idx[0].size == 0:
        print('no difference')
    else:
        print((S0sh-S0_exam)[diff_idx])

    print('Test traction:')
    print('diff (T0, refT0):', T0sh.shape, T0_exam.shape)
    diff_idx = (np.abs(T0sh-T0_exam)>1e-10).nonzero()
    if diff_idx[0].size == 0:
        print('no difference')
    else:
        print((T0sh-T0_exam)[diff_idx])
