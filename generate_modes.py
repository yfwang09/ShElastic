from ShElastic import generate_modes

# generate modes for future use
lKfull = 40; lJfull = 43;
generate_modes(lKfull, save_lmax=lJfull)

# test the generated modes

from ShElastic import genUmode, calUmode, genSmode, calTmode
from SHUtil import lmk2K, K2lmk
from SHUtil import sparse_mode, dense_mode
from SHBV import generate_submat
import numpy as np
from scipy.io import loadmat

Umodes = loadmat('Umodes.mat')
Smodes = loadmat('Smodes.mat')
Tmodes = loadmat('Tmodes.mat')

lKmax = 10; lJmax = 13;
l = 10; m = 5; k = 0; shtype = 'irr';
K = lmk2K(l, m, k, lmax=lKmax)

# load U,S,T mode from file
U0modes = generate_submat(1, 1/3, Umodes['U0'+shtype], lKmax, lJmax, lKfull=lKfull, lJfull=lJfull).tocsc()
S0modes = generate_submat(1, 1/3, Smodes['S0'+shtype], lKmax, lJmax, lKfull=lKfull, lJfull=lJfull, kJ=9).tocsc()
T0modes = generate_submat(1, 1/3, Tmodes['T0'+shtype], lKmax, lJmax, lKfull=lKfull, lJfull=lJfull).tocsc()
U0_exam = U0modes[:, K]
S0_exam = S0modes[:, K]
T0_exam = T0modes[:, K]

# generate U,S,T from scratch
Unu, U0, Snu1, Snu2, Snu3, S0 = genSmode(l, m, k, shtype=shtype, returnU=True)
Tnu1 = calTmode(Snu1); Tnu2 = calTmode(Snu2); Tnu3 = calTmode(Snu3); T0 = calTmode(S0)

U1sh = sparse_mode(Unu, lmax=lJmax); U0sh = sparse_mode(U0, lmax=lJmax);
S1sh = sparse_mode(Snu1, lmax=lJmax); S2sh = sparse_mode(Snu2, lmax=lJmax);
S3sh = sparse_mode(Snu3, lmax=lJmax); S0sh = sparse_mode(S0, lmax=lJmax);
T1sh = sparse_mode(Tnu1, lmax=lJmax); T2sh = sparse_mode(Tnu2, lmax=lJmax);
T3sh = sparse_mode(Tnu3, lmax=lJmax); T0sh = sparse_mode(T0, lmax=lJmax);

# compare the saved mode and generated mode
print(K)
print('sparseU0', U0sh.shape)
#print(U0sh)
print('U0_exam', U0_exam.shape)
#print(U0_exam)
print('diff')
diff_idx = (np.abs(U0sh-U0_exam)>1e-10).nonzero()
if diff_idx[0].size == 0:
    print('no difference')
else:
    print((U0sh-U0_exam)[diff_idx])
#print(U0sh-U0_exam)
print('sparseS', S0sh.shape)
#print(S0sh)
print('Smodes', S0_exam.shape)
#print(S0_exam)
print('diff')
diff_idx = (np.abs(S0sh-S0_exam)>1e-10).nonzero()
if diff_idx[0].size == 0:
    print('no difference')
else:
    print((S0sh-S0_exam)[diff_idx])
print('sparseT', T0sh.shape)
#print(T0sh)
print('Tmodes', T0_exam.shape)
#print(T0_exam)
print('diff')
diff_idx = (np.abs(T0sh-T0_exam)>1e-10).nonzero()
if diff_idx[0].size == 0:
    print('no difference')
else:
    print((T0sh-T0_exam)[diff_idx])

