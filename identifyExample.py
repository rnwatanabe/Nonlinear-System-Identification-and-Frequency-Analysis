#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 18:37:27 2019

@author: Renato Naville Watanabe
"""

import identFunctions as ident
import GFRFfunctions as GFRF
import numpy as np
import matplotlib.pyplot as plt
from sympy.plotting import plot3d, plot
import sympy as sym
from scipy.signal import filtfilt, butter
plt.close('all')
#%%
def system(u, Fs, noiseSTD=0.0, eu=0, ey=0):
    N = len(u)
    t = np.arange(0, N/Fs, 1/Fs)
#    y = np.sin(2*np.pi*4.52*t).reshape(-1,1)
    y = 0*np.random.randn(len(u), 1)
    
    for i in range(2, len(u)):
        u[i] = 0.5*u[i-1] - 0.3*u[i-2] + 0.1*y[i-2] + 0.4*y[i-1]*y[i-2] + eu[i-1]
        y[i] = 0.3*y[i-1] - y[i-2] - 0.1*u[i-2] + ey[i-1]

    u = u + noiseSTD*np.random.randn(N, 1)
    y = y + noiseSTD*np.random.randn(N, 1)
    
    return t, u, y
#%%
maxLagu = 2
maxLagy = 2
order = 2
Fs = 20.0
fres = 0.01
N = 100000
#%%
b, a = butter(2, 0.95)
#%%
u = 0*np.random.randn(N, 1)
eu = 0.2*np.random.randn(N, 1)
ey = 0.2*np.random.randn(N, 1)

#ey = filtfilt(b, a, ey, axis = 0)
#eu = filtfilt(b, a, eu, axis = 0)

t, u, y = system(u, Fs, noiseSTD=0, eu=eu, ey=ey)

#t = t[1000:]
#u = u[1000:]
#y = y[1000:]
#eu = eu[1000:]
#ey = ey[1000:]

plt.figure()
plt.plot(t, u)
plt.show()
plt.figure()
plt.plot(t, y)
plt.show()
#%%
print('system identification')
beta_uy, n_uy, Duy = ident.identifyModel(u, y, maxLagu, maxLagy, ustring='u',
                                         ystring='y', nstring='n', delay=1,
                                         degree=order, L=20, constantTerm=True,
                                         pho = 0.005, supress=False)
#%%
beta_yu, n_yu, Dyu = ident.identifyModel(y, u, maxLagy, maxLagu, ustring='y',
                                         ystring='u', nstring='m', delay=1,
                                         degree=2, L=20, constantTerm=True,
                                         pho=0.01, supress=False)
#%%
print('system GFRFs')
Hnuy = GFRF.computeSystemGFRF(Duy, Fs, beta_uy, 2, ustring='u', ystring='y')
f1, f2 = sym.symbols('f1 f2')
Hy = sym.lambdify(f1, Hnuy[1], 'numpy')
#plot3d(sym.Abs(Hnuy[2]), (f1,-Fs/2, Fs/2), (f2,-Fs/2, Fs/2))
#%%
Hnyu = GFRF.computeSystemGFRF(Dyu, Fs, beta_yu, 2, ustring='y', ystring='u')
Hu = sym.lambdify(f1, Hnyu[1], 'numpy')

fV = np.linspace(-Fs/2, Fs/2, 100000)
plt.figure()
plt.plot(fV, np.abs(Hy(fV)))
plt.title('Hy1')
plt.show()

plt.figure()
plt.plot(fV, np.abs(Hu(fV)))
plt.title('Hu1')
plt.show()
###############################################################################
#%%
fmin = 0
fmax = Fs/2
#%%
print('signals FFTs')
Y, f = GFRF.computeSignalFFT(ident.reshapeyvector(y, L=1), Fs, fres)
U, f = GFRF.computeSignalFFT(ident.reshapeyvector(u, L=1), Fs, fres)
EU, f = GFRF.computeSignalFFT(ident.reshapeyvector(eu, L=1), Fs, fres)

plt.figure()
plt.plot(f, 20*np.log10(np.abs(U)))
plt.title('U')
plt.show()

plt.figure()
plt.plot(f, 20*np.log10(np.abs(Y)))
plt.title('Y')
plt.show()
#print(f)
#%%

#print('NOFRF y->u')
#NOFRFyu, _ = GFRF.computeNOFRF(Hnyu, Y, 1, order, Fs, fres, 
#                               fmin, fmax, 0, fmax)
#NOFRFyu = np.reshape(NOFRFyu, (-1,1))
#
#plt.figure()
#plt.plot(f, 20*np.log10(np.abs(2*U)), 'b')
#plt.plot(f[f>=-fres/2], 20*np.log10(np.abs(NOFRFyu)), 'g')
##plt.plot(f[f>=-fres/2], 20*np.log10(np.abs(NOFRFyu - 2*U[f>=-fres/2])))
#plt.xlim(fmin, fmax)
#plt.show()
##%%
#print('NOFRF u->y')
#NOFRFuy, _ = GFRF.computeNOFRF(Hnuy, U, 1, order, Fs, fres, 
#                               fmin, fmax, 0, fmax)
#
#NOFRFuy = np.reshape(NOFRFuy, (-1,1))
#plt.figure()
#plt.plot(f, 20*np.log10(np.abs(2*Y)), 'b')
#plt.plot(f[f>=-fres/2], 20*np.log10(np.abs(NOFRFuy)), 'g')
#plt.xlim(fmin, fmax)
#plt.show()
#%%

f_inputMin = 0
f_inputMax = Fs/2
maxDegree = 2
NPDCuy, NPDCyu, f, NPDCuyn, NPDCyun, NPDCuyLinear, NPDCyuLinear, NPDCuynLinear, NPDCyunLinear = GFRF.NPDC(u, y, Fs, fres, beta_uy, beta_yu, Duy, Dyu,
                                                                                                          Hnuy, Hnyu, f_inputMin, f_inputMax, maxOrder=maxDegree, 
                                                                                                          N=100, ustring='u', ystring='y')
                                                                                         
plt.figure()
plt.plot(f, np.abs(NPDCuyn))
plt.plot(f, np.abs(NPDCuy), linewidth=6)
plt.ylim(0, 1)
plt.title('NPDC u->y')
plt.show()


plt.figure()
plt.plot(f, np.abs(NPDCyun))
plt.plot(f, np.abs(NPDCyu), linewidth=6)
plt.ylim(0, 1)
plt.title('NPDC y->u')
plt.show()



                                                                                         
plt.figure()
plt.plot(f, np.abs(NPDCuynLinear))
plt.plot(f, np.abs(NPDCuyLinear), linewidth=6)
plt.ylim(0, 1)
plt.title('NPDC u->y Linear')
plt.show()


plt.figure()
plt.plot(f, np.abs(NPDCyunLinear))
plt.plot(f, np.abs(NPDCyuLinear), linewidth=6)
plt.ylim(0, 1)
plt.title('NPDC y->u Linear')
plt.show()

