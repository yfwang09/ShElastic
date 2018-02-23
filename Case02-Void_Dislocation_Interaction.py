
# coding: utf-8

# # Void-Dislocation Interaction Test Case
# 
# First we import the libraries we developed:

# In[4]:

import numpy as np
import matplotlib.pyplot as plt
import pyshtools
from SHTest import gavazza1974
from shimg import tauyz_ShElastic

#### Gavazza & Barnett (1974) ####
mu = 2.65
nu = 0.347
r0 = 1.25
b = 0.248
t = 1.5*r0
z = np.linspace(0, 3, 30)*r0

# ShElastic Solution

sigma, act_mode, tsolve, tconstruct = tauyz_ShElastic(20, z, t, r0, b, mu0=mu, nu=nu)
print(sigma[...,1,2].shape)
tau_ShE = sigma[...,1,2].flatten()

# Analytical Solution

x3zT= np.linspace(0, 3, 50)*r0

tau_yz = gavazza1974(20, x3zT, [t, ], mu, 0, nu, nu, r0, b)

# Plot the result
plt.figure()
plt.plot(x3zT/r0, -tau_yz/mu*r0, label='Gavazza&Barnett(1974)')
plt.plot(z/r0, -tau_ShE.flatten()/mu*r0, '^', label='ShElastic')
plt.xlabel(r'$z/r_0$', fontsize=20)
plt.xlim(0, 3)
plt.ylabel(r'$\tau_{yz}r_0/\mu$', fontsize=20)
plt.tick_params(labelsize=14)
plt.tight_layout()
plt.legend(fontsize=14)
plt.show()

