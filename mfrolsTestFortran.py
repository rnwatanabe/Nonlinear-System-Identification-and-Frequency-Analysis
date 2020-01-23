#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:16:35 2020

@author: Renato Naville Watanabe
"""

import identFunctions as ident
import GFRFfunctions as GFRF
import numpy as np
import matplotlib.pyplot as plt
from sympy.plotting import plot3d, plot
import sympy as sym
from scipy.signal import filtfilt, butter, correlate
import scipy


def system2(u, Fs, noiseSTD=0.0, eu=0, ey=0):
    N = len(u)
    t = np.arange(0, N/Fs, 1/Fs)
#     y = np.sin(2*np.pi*4.52*t).reshape(-1,1)
    y = 0*np.random.randn(len(u), 1)
    
    for i in range(2, len(u)):
        u[i] = 0.5*u[i-1] - 0.3*u[i-2] + 0.1*y[i-2] + 0.4*y[i-1]*y[i-2] + eu[i-1]
        y[i] = 5*np.sin(2*np.pi*4.5*t[i]) + ey[i-1]

    u = u + noiseSTD*np.random.randn(N, 1)
    y = y + noiseSTD*np.random.randn(N, 1)
    
    return t, u, y

Fs = 20.0
N = 100000

u = 0*np.random.randn(N, 1)
eu = 0.2*np.random.randn(N, 1)
ey = 0.1*np.random.randn(N, 1)

#ey = filtfilt(b, a, ey, axis = 0)
#eu = filtfilt(b, a, eu, axis = 0)

t, u, y = system2(u, Fs, noiseSTD=0, eu=eu, ey=ey)
maxLagu = 5
maxLagy = 5
order = 2

beta_uy, n_uy, Duy = ident.identifyModel(u, y, maxLagu, maxLagy, ustring='u',
                                         ystring='y', nstring='n', delay=1,
                                         degree=order, L=20, constantTerm=True,
                                         pho = 0.0001, supress=False, 
                                         mfrolsEngine='fortran')
