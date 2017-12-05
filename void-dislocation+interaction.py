
# coding: utf-8

# # Void-Dislocation Interaction Test Case
# 
# Based on the analytical solution provided by Gavazza & Barnett (1974), and the implementation in Takahashi and Ghoniem (2008), we have the testcase as following:

# * The solution of the stress:
# $$
# \tau_{yz}(t,0,z)=
# \sum_{n=1}^\infty\left\{-a^{n+1}(2\mu_1\alpha_n+\mu_1\beta_n)+6\mu_1K_n\Omega_n\frac{2n+1}{2n+5}\right\}
#     {1\over r^{n+2}}P^{n-1}_{n+1}
# $$
# 
# $$ +
# \sum_{n=1}^\infty\left\{-{1\over2}\mu_1a^{n+1}\beta_n+\mu_1K_n\Omega_n\frac{2n+3}{2n+5}\right\}
#     {1\over r^{n+2}}P^{n+1}_{n+1}
# $$
# 
# $$ +
# \sum_{n=1}^\infty\left\{2\mu_1\left({a^2\over r^2}-1\right)K_n\Omega_n+\mu_1K_n\Omega_n\frac{2}{2n+5}\right\}
#     {1\over r^{n+2}}P^{n+1}_{n+3}
# $$
# 
# $$ +
# \sum_{n=1}^\infty\left\{24\mu_1\left({a^2\over r^2}-1\right)K_n\Omega_n+\mu_1K_n\Omega_n\frac{24}{2n+5}\right\}
#     {1\over r^{n+2}}P^{n-1}_{n+3}
# $$

# where,
# $$
# K_n=-\frac{\lambda_1 + \mu_1}{2[(n+2)\lambda_1+(3n+5)\mu_1]}
# $$
# 
# $$
# \Omega_n=\frac{2\eta_n(\mu_1-\mu_2)a^{2n+1}/t^n}{\mu_1[(n+2)+E_n^I(2n+1)(n+1)]+\mu_2(2n)}
# $$
# 
# $$
# \alpha_n=\frac{(\mu_1-\mu_2)\eta_n}{\mu_1(n+2)+\mu_2(n-1)}
# \frac{\mu_1\{(n+2)-E_n^I(2n+1)(n-1)\}}{\mu_1{(n+2)+E_n^I(2n+1)(n+1)}+\mu_2(2n)}
# \left({a\over t}\right)^n
# $$
# 
# $$
# \beta_n=\frac{(\mu_2-\mu_1)\eta_n}{\mu_1(n+2)+\mu_2(n-1)}
# \frac{\mu_1\{(n+2)+E_n^I(2n+1)(n-1)\}+2\mu_2(n-1)}{\mu_1{(n+2)+E_n^I(2n+1)(n+1)}+\mu_2(2n)}
# \left({a\over t}\right)^n
# $$
# 
# $$
# \eta_n=\left(b\over4\pi\right)\frac{(-1)^n2^n(n-1)!}{(2n-1)!}
# $$
# 
# $$
# E_n^I=\frac{1}{2n+1}\frac{(n-1)\lambda_1-(n+4)\mu_1}{(n+2)\lambda_1+(3n+5)\mu_1}
# $$
# 
# $P_l^m$ is the *unnormalized Legendre Polynomial* with *Condon-Shortley Phase* coefficients.

# In[1]:

import numpy as np
from scipy.misc import factorial
from scipy.io import loadmat
import pyshtools
import matplotlib.pyplot as plt

def Legendre_poly(N, Z, LegendreP,dl,dm, csphase=-1):
    nmax = N.max()
    dl = np.array(dl)
    dm = np.array(dm)
    p = np.empty(Z.shape+dl.shape)
    #print(p.shape)
    for idx, _ in np.ndenumerate(N):
        #print(idx)
        n, z = (N[idx], Z[idx])
        Pvalue = LegendreP(nmax+3, z, csphase=csphase)
        l = n + dl; m = n + dm;
        Pidx = np.array((l*(l+1)/2+m), dtype=np.int)
        #print(p[idx].shape)
        #print(Pidx)
        p[idx] = Pvalue[Pidx]

    return p

def void_screw_disl(nmax, zs, ts, mu1, mu2, nu1=0.25, nu2=0.25, a=1, b=1):
    ns = np.arange(nmax) + 1
    n, z, t = np.meshgrid(ns, zs, ts)
    r = np.sqrt(z**2 + t**2)
    z = z/r
    lambda1 = 2*mu1*nu1/(1-2*nu1)
    lambda2 = 2*mu2*nu2/(1-2*nu2)

    Kn = -(lambda1+mu1)/2/((n+2)*lambda1+(3*n+5)*mu1)
    #EnI = ((n-1)*lambda1 - (n+4)*mu1)/((n+2)*(2*n+1)*lambda1+(3*n+5)*mu1)
    EnI = 1/(2*n+1)*((n-1)*lambda1 - (n+4)*mu1)/((n+2)*lambda1+(3*n+5)*mu1)
    #EnI = 1/(2*n+1)*((n+2)*lambda1 - (n-3)*mu1)/((n-1)*lambda1+(3*n-2)*mu1)
    #EnI = ((n+2)*lambda1 - (n-3)*mu1)/((2*n+1)*(n-1)*lambda1+(3*n-2)*mu1)
    eta_n = b/4/np.pi * ((-2)**n) *factorial(n-1)/factorial(2*n-1)
    #print('Kn', Kn.shape)
    #print('EnI', EnI.shape)
    #print('eta_n', eta_n.shape)
    #print('Kn', Kn)
    #print('EnI', EnI)
    #print('eta_n', eta_n)

    Cab = (mu1-mu2)*eta_n/(mu1*(n+2)+mu2*(n-1))
    denom_ab = mu1*((n+2) + EnI*(2*n+1)*(n+1)) + mu2*2*n
    a_t_n = (a/t)**n
    alpha_n = Cab/denom_ab * (mu1*((n+2) - EnI*(2*n+1)*(n-1))) * a_t_n
    beta_n = -Cab/denom_ab * (mu1*((n+2) + EnI*(2*n+1)*(n-1)) + 2*mu2*(n-1)) * a_t_n
    #print('alpha_n', alpha_n.shape)
    #print('beta_n', beta_n.shape)
    #print('alpha_n', alpha_n)
    #print('beta_n', beta_n)

    Omega_n = 2*eta_n*(mu1-mu2)* a_t_n * (a**(n+1)) / denom_ab
    #print('Omega_n', Omega_n.shape)
    #print('Omega_n', Omega_n)

    C_term1 = (-a**(n+1) * (2*mu1*alpha_n + mu1*beta_n) + 6*mu1*Kn*Omega_n*(2*n+1)/(2*n+5)) /r**(n+2)
    C_term2 = (-a**(n+1)/2 *mu1*beta_n + mu1*Kn*Omega_n*(2*n+3)/(2*n+5)) /r**(n+2)
    C_term3 = (2*mu1*(a**2/r**2 - 1)*Kn*Omega_n+mu1*Kn*Omega_n*2/(2*n+5)) /r**(n+2)
    C_term4 = (24*mu1*(a**2/r**2 - 1)*Kn*Omega_n+mu1*Kn*Omega_n*24/(2*n+5)) /r**(n+2)
    #C_term3 = (2*mu1*(0 - 1)*Kn*Omega_n+mu1*Kn*Omega_n*2/(2*n+5)) /r**(n+2)
    #C_term4 = (24*mu1*(0 - 1)*Kn*Omega_n+mu1*Kn*Omega_n*24/(2*n+5)) /r**(n+2)
    #print(C_term1.shape, C_term2.shape, C_term3.shape, C_term4.shape)
    #print(C_term1, C_term2, C_term3, C_term4)

    # After experiment, the implementation in Takahashi&Ghoniem(2008) is unnormalized Legendre Polynomial
    p = Legendre_poly(n, z, pyshtools.legendre.PLegendreA,dl = np.array([1, 1, 3, 3]),dm = np.array([-1,1, 1,-1]), csphase=-1)
    #C3 = 2*mu1*a**2*Kn*Omega_n
    #C4 = 24*mu1*a**2*Kn*Omega_n
    #print(Kn, Omega_n, (C3+C4))
    return np.sum(C_term1*p[:,:,:,0] + C_term2*p[:,:,:,1] + C_term3*p[:,:,:,2] + C_term4*p[:,:,:,3], axis=1)

def gavazza1974(nmax, zs, ts, mu1, mu2, nu1=0.25, nu2=0.25, a=1, b=1):
    # this version gets rid of the round-off errors
    ns = np.arange(nmax)+1
    n, z, t = np.meshgrid(ns, zs, ts)
    r = np.sqrt(z**2 + t**2)
    ct = z/r
    lambda1 = 2*mu1*nu1/(1-2*nu1)
    lambda2 = 2*mu2*nu2/(1-2*nu2)

    Kn = -(lambda1+mu1)/2/((n+2)*lambda1+(3*n+5)*mu1)
    EnI = 1/(2*n+1)*((n-1)*lambda1 - (n+4)*mu1)/((n+2)*lambda1+(3*n+5)*mu1)
    eta_n = b/2/np.pi * ((-2)**n) *factorial(n)/factorial(2*n)
    #eta_n = b/2/np.pi * (-1)**(n%2) * np.exp(np.sum(np.log(2/np.arange(n+1, 2*n+1))))

    Cab = (mu1-mu2)/(mu1*(n+2)+mu2*(n-1))
    denom_ab = mu1*((n+2) + EnI*(2*n+1)*(n+1)) + mu2*2*n
    a_t_n = (a/t)**n
    alpha_n = Cab/denom_ab * (mu1*((n+2) - EnI*(2*n+1)*(n-1))) * a_t_n
    beta_n = -Cab/denom_ab * (mu1*((n+2) + EnI*(2*n+1)*(n-1)) + 2*mu2*(n-1)) * a_t_n

    Omega_n = 2*(mu1-mu2)* a_t_n * ((a/r)**(n+1)) / denom_ab

    # After experiment, the implementation in Takahashi&Ghoniem(2008) is unnormalized Legendre Polynomial
    p = Legendre_poly(n, ct, pyshtools.legendre.PLegendreA,dl=[1, 1, 2, 2, 3, 3], dm=[-1, 1, 1, -1, 1, -1], csphase=-1)
    term1 = -(a/r)**(n+1) * (2*alpha_n/r * p[:,:,:,0] + beta_n/2/r * (p[:,:,:,1]+2*p[:,:,:,0]))
    term2 = 2*((a/r)**2-1)*Kn*Omega_n/r * (p[:,:,:,4]+12*p[:,:,:,5])
    term3 = Kn*Omega_n/r * ct * (p[:,:,:,2] + 6*p[:,:,:,3])
    sol = mu1*(term1 + term2 + term3)*eta_n
    #sol = mu*b*(term2)
    return np.sum(sol, axis=1)


# In Willis et al. (1972), The force term is:
# 
# $$
# F_1 = {\mu b^2\over2\pi a}\sum_{n=1}^\infty\frac{(-1)^n2^nn!}{(2n)!}\left(a^2\over rd\right)^nz
#     \left\{P_{n+1}^{n-1}(\cos\theta)\left(\frac{1+(2n-1)m}{n+2}\right)
#           -\frac{1+m}{2(n+2)}P_{n+1}^{n+1}(\cos\theta)
#           +{3\over l}cos\theta(P_{n+2}^{n+1}(\cos\theta)+P_{n+2}^{n-1}(\cos\theta))
#           -{6\over l}(1-z)(P_{n+3}^{n+1}(\cos\theta)+12P_{n+3}^{n-1}(\cos\theta))
#     \right\}
# $$
# 
# where $z=(a/r)^2$ and $l=2(3n^2+7n+6)$

# In[5]:

def willis1972(a, d, nmax, x3):
    n = np.arange(1, nmax)
    r = np.sqrt(x3**2+d**2)
    ct= x3/r
    st= d/r
    N, X3 = np.meshgrid(n, x3)

    M = (6-N)/(3*N**2+7*N+6)
    R = np.sqrt(X3**2+d**2)
    CT= X3/R
    L = 2*(3*N**2+7*N+6)
    Z = (a/R)**2

    p = Legendre_poly(N, CT, pyshtools.legendre.PLegendreA, dl=[1, 1, 2, 2, 3, 3], dm=[-1, 1, 1, -1, 1, -1], csphase=-1)
    #print(p.shape)

    prefactor = 1/2/np.pi*(-2*a**2/R/d)**N*factorial(N)/factorial(2*N) * Z
    term1 = (1+(2*N-1)*M)/(N+2)*p[:,:,0]
    term2 = -(1+M)/2/(N+2)*p[:,:,1]
    term3 = 3/L*CT*(p[:,:,2]+6*p[:,:,3])
    term4 = -6/L*(1-Z)*(p[:,:,4]+12*p[:,:,5])

    F1 = prefactor*(term1+term2+term3+term4)
    #print(F1.shape)
    Fa = 5.0/32/np.pi*(a/d)**4*(1+3*ct**2)*st**3
    #tau_yz = -void_screw_disl(49, x3, [d,], 1, 0, 1.0/3, 1.0/3, a, 1)
    F = -2*np.trapz(np.sum(F1[:, :], axis=1), x=x3)
    #print(F)
    
    return (F1, Fa, F)

# ## Implementation of ShElastic Solution
# 
# First we import the libraries we developed:

# In[6]:

import numpy as np
import pyshtools
from SHUtil import SHCilmToVector
from SHBV import spec_J2lmk, spec_lmk2J, subCmat, print_SH_mode, create_Cmat, visualize_Cmat, stress_solution
from ShElastic import T_mode, S_mode


# Then we generate the meshgrid on the void surface for boundary conditions. 

# In[7]:

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

#### Willis, Hayns & Bullough (1972) ####
mu = 1
nu = 1.0/3
a = 1
b = 1
t = 2

x3 = np.linspace(0, 3, 30)

# zs = np.linspace(0,6,50)
# ts = [t, ]

lmax_full = 15
lmax_mode = 12
Cmat_file = 'Cmat_lmax%d_mode%d_mu%f_nu%f.mat' % (lmax_full, lmax_mode, mu, nu)

Ngrid = 20
theta = np.arange(0, np.pi, np.pi/Ngrid)
phi = np.arange(0, 2*np.pi, np.pi/Ngrid)
THETA, PHI = np.meshgrid(theta, phi)
print(THETA.shape, PHI.shape)

Z = a*np.cos(THETA)
Y = a*np.sin(THETA)*np.sin(PHI)
X = a*np.sin(THETA)*np.cos(PHI)
#print(Z)

N = -np.stack([X/a, Y/a, Z/a], axis=-1)
print(N.shape)


# Then we evaluate traction boundary conditions on the spherical surface. We have a spherical void center at the origin with radius $a$, and a RH screw dislocation at $(t,0,0)$, with $\hat{\xi}=\hat{\mathbf{e}}_z$, with a burger's vector magnitude of $b$. The stress field induced by the dislocation in an infinite medium can be written as $\mathbf{\sigma}^\infty$, with the only two non-zero terms:
# 
# $$
# \sigma_{xz}^\infty = -\frac{\mu b}{2\pi}\frac{y}{(x-t)^2+y^2} \\
# \sigma_{yz}^\infty = \frac{\mu b}{2\pi}\frac{x-t}{(x-t)^2+y^2}
# $$

# In[8]:

sigma_inf = np.zeros(THETA.shape+(3, 3))
sigma_inf[:, :, 0, 2] = -mu*b/2/np.pi * Y/((X-t)**2+Y**2)
sigma_inf[:, :, 2, 0] = -mu*b/2/np.pi * Y/((X-t)**2+Y**2)
sigma_inf[:, :, 1, 2] =  mu*b/2/np.pi * (X-t)/((X-t)**2+Y**2)
sigma_inf[:, :, 2, 1] =  mu*b/2/np.pi * (X-t)/((X-t)**2+Y**2)

print(sigma_inf.shape)

T_inf = N*0
for i,x in np.ndenumerate(THETA):
    T_inf[i] = np.dot(sigma_inf[i], N[i])

T_usr_mesh = T_inf
print(T_usr_mesh.shape)
'''
ttl = ['$T_x$','$T_y$','$T_z$']
for i in range(3):
    fv = T_usr_mesh[:,:,i]
    fmax, fmin = fv.max(), fv.min()
    fcolors = (fv - fmin)/(fmax - fmin)    # normalize the values into range [0, 1]
    fig0 = plt.figure(i*2, figsize=(10, 5))
    ax0 = fig0.add_subplot(111)
    cax0 = ax0.imshow(fv.T, extent=(0, 360, -90, 90), cmap='viridis')
    ax0.set(xlabel='longitude', ylabel='latitude', title=ttl[i])
    fig0.colorbar(cax0)
    fig1 = plt.figure(i*2+1, figsize=plt.figaspect(1.))  # make the axis with equal aspects
    ax1= fig1.add_subplot(111, projection='3d')   # add an subplot object with 3D plot
    ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.viridis(fcolors))
    ax1.set(xlabel='x', ylabel='y', zlabel='z')
plt.show()
'''

# Then we expand the traction boundary conditions to spherical harmonic modes:

# In[9]:

T_usr_vec = np.array([])
lmax_sub = np.int(Ngrid/2) - 1
mode_sub = lmax_sub - 3
print('lmax =', lmax_sub, 'mode_max =', mode_sub)
np.set_printoptions(suppress=True)

mk = '*x.'
for k in range(3):
    T_usr_cilm = pyshtools.expand.SHExpandDHC(T_usr_mesh[:,:,k].T, sampling=2, lmax_calc=lmax_sub)
    T_usr_mode = pyshtools.SHCoeffs.from_array(T_usr_cilm)
    T_usr_grid = T_usr_mode.expand()
    #fig1, ax1 = T_usr_grid.plot()
    fig2, ax2 = T_usr_grid.plot3d(elevation=20, azimuth=30)
    #fig1.suptitle('k=%d'%k)
    fig2.suptitle('k=%d'%k)
    #power_per_l = pyshtools.spectralanalysis.spectrum(T_usr_cilm)
    #plt.loglog(np.arange(2, T_usr_cilm.shape[1]), power_per_l[2:], mk[k])
    # print(T_usr_cilm.shape)
    # T_usr_vec_p = SHCilmToVector(T_usr_cilm, lmax = lmax_sub)
    # print(T_usr_vec_p.shape)
    T_usr_vec = np.hstack((T_usr_vec, SHCilmToVector(T_usr_cilm, lmax = lmax_sub)))
#T_usr_idx = print_SH_mode(T_usr_vec,m_dir=3)
#plt.grid(True)
#plt.xlabel('degree l')
#plt.ylabel('the power of degree l')
#plt.legend(['$T_x$','$T_y$','$T_z$'])
#plt.show()


# In[10]:

import scipy.sparse as spm
from scipy.io import loadmat, savemat

m_max = 3
Cmat = create_Cmat(lmax_full, lmax_mode, mu, nu, m_max=m_max, Cmat_file=Cmat_file, recalc=False, etol=1e-10)
#Cmat = loadmat(Cmat_file)['Cmat']
#print(Cmat_file, Cmat.shape, lmax_full, lmax_mode, lmax_sub, mode_sub)
#savemat(Cmat_file, {'Cmat': Cmat})


# In[11]:

Csub = subCmat(Cmat, lmax_full, lmax_mode, lmax_sub, mode_sub, m_max=m_max)
#savemat(Cmat_file, {'Cmat': Cmat})


# In[12]:

#print('Rank of the submatrix', np.linalg.matrix_rank(Csub.todense()))
plt.figure(figsize=(24,24))
visualize_Cmat(Csub, precision=1e-8, m_max=m_max)
plt.show()


# In[13]:

Csub_copy = spm.lil_matrix(Csub.shape, dtype=np.complex)
etol = 1e-8
real_idx = np.abs(np.real(Csub)) > etol
imag_idx = np.abs(np.imag(Csub)) > etol
Csub_copy[real_idx] = Csub[real_idx]
Csub_copy[imag_idx] = Csub[imag_idx]
#print(real_idx.shape, imag_idx.shape)
Csub = Csub_copy
print(Csub.shape)


# In[14]:

import time
tic = time.time()
A = spm.linalg.lsqr(Csub, T_usr_vec.transpose())
toc = time.time()
print('Residual:', A[3], 'Time:', toc-tic)
A_sol = A[0]
print('Solution:', A_sol.size)
index_sol = print_SH_mode(A_sol, m_dir=3)


# Then we integrate the stress solution:

# In[15]:

X, Y, Z = np.meshgrid([t, ], [0,], x3)
#print(ts, zs)
sigma_tot = stress_solution(index_sol, X, Y, Z, MU=mu, NU=nu, lmax=lmax_sub, recalc=False)


# In[16]:

# plot for Willis (1972)
print(sigma_tot[:,:,:,1,2].shape)
plt.figure(figsize=(12, 9))
tau_ShE = np.real(sigma_tot[:,:,:,1,2]).flatten()
#plt.plot(x3, -F_mode36, 'x')
#plt.plot(x3, -F_mode26, 'x')
#plt.plot(x3, -F_mode16, '*')
x3zT= np.linspace(0, 3, 50)
x3z = np.linspace(0, 3, 500)
tau_yz = gavazza1974(50, x3zT, [t, ], mu, 0, nu, nu, a, b)
#tau_ShE = void_screw_disl(50, x3, [1/0.9, ], mu, 0, nu, nu, a, b)
F1, Fa, F = willis1972(a, t, 50, x3z)
plt.plot(x3zT, -tau_yz, 'o')
plt.plot(x3z, np.sum(F1[:, :], axis=1))
plt.plot(x3, -tau_ShE.flatten(), '^')
plt.legend(['Takahashi&Ghoniem(2008): $t/a='+str(t/a)+'$',\
             'Willis et al.(1972): $t/a='+str(t/a)+'$',\
             'ShElastic solution ($t/a='+str(t/a)+',l_{max}=20$)'])

plt.xlabel('z/t')
plt.ylabel(r'$\tau_{yz}a/\mu$')
plt.xlim(0, 3)
plt.ylim(0, 0.01)
#plt.legend(['ShElastic solution ($a/d=0.9,l_{max}=26$)', 'ShElastic solution ($l_{max}=16$)', 'ShElastic solution ($l_{max}=6$)', 'Willis(1972)'])
plt.show()



