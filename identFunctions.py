#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:59:36 2019

@author: rnwatanabe
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

def ls(p, y):
    py = (p.T).dot(y)
    covP = np.linalg.inv((p.T).dot(p))
    beta = covP.dot(py)
    return beta

def mls(p, y, L=1):
    p, y = reshapepymatrices(p, y, L)
    L = p.shape[2]
    betaL = np.zeros((p.shape[1], L))
    for j in range(L):
        py = (p[:, :, j].T).dot(y[:, [j]])
        covP = np.linalg.inv((p[:, :, j].T).dot(p[:, :, j]))
        betaL[:, [j]] = covP.dot(py)
    beta = np.mean(betaL, axis=1, keepdims=True)
    return beta

def mols(p, y, L=1):
    import numpy as np
    p, y = reshapepymatrices(p, y, L)
    M = p.shape[1]
    L = p.shape[2]
    qs = np.zeros_like(p)
    q = np.zeros_like(p)
    A = np.zeros((M, M, L))
    g = np.zeros((L, M))
    gs = np.zeros((L, M))
    beta = np.zeros((M, L))


    qs = np.copy(p)
    
    for m in range(M):
    ## The Gram-Schmidt method was implemented in a modified way, as shown in Rice, JR(1966)
        for r in range(m):
            qs[:, [m], :] = qs[:, [m], :] - np.sum(q[:, [r], :]*qs[:, [m], :], axis=0)/(np.sum(q[:, [r], :]*q[:, [r], :], axis=0)+1e-6)*q[:, [r], :]
            A[r, m, :] = np.sum(q[:, [r], :]*p[:, [m], :], axis=0)/(np.sum(q[:, [r], :]*q[:, [r], :], axis=0)+1e-6) 
        gs[:, [m]] = (np.sum(y*qs[:, m, :], axis=0, keepdims=True)/(np.sum(qs[:, m, :]*qs[:, m, :], axis=0, keepdims=True)+1e-6)).T
        A[m, m, :] = 1.0
        q[:, m, :] = qs[:, m, :]
        g[:, m] = gs[:, m]

    for j in range(L):
        if M > 1:
            beta[:, [j]] = np.linalg.inv(A[:, :, j])@g[[j], :].T
        else:
            beta[:, j] = (np.squeeze(A)**-1)*g[j, :]

    beta_m = np.mean(beta, axis=1, keepdims=True)
    return beta_m, beta

def whitenMatrix(X):
    X_c = X - np.mean(X, 0, keepdims=True)
    covMatrixEst = X_c.T.dot(X_c)/X_c.shape[0]
    D, E = np.linalg.eig(covMatrixEst)
    W = E.T
    scale = np.diag(D**(-1/2.0))
    W = scale.dot(W)
    W = E.dot(W)

    return W

def wls(p, y, W):

    py = p.T.dot(W).dot(y)
    covP = np.linalg.inv((p.T).dot(W).dot(p))
    beta = covP.dot(py)

def covMatrix(x):

    cov = x.dot(x.T)#/x.shape[0]
    return cov

def mfrols(p, y, pho, s, ESR, l, err, A, q, g, verbose=False):
    '''
    Implements the MFROLS algorithm (see page 97 from Billings, SA (2013)).
        written by: Renato Naville Watanabe
        beta = mfrols(p, y, pho, s)
        Inputs:
          p: matrix of floats, is the matrix of candidate terms.
          y: vector of floats, output signal.
          pho: float, stop criteria.
          s: integer, iteration step of the mfrols algorithm.
          l: vector of integers, indices of the chosen terms.M = np.shape(p)[1]; l = -1*np.ones((M))
          err: vector of floats, the error reduction ratio of each chosen term. err = np.zeros((M))
          ESR: float, the sum of the individual error reduction ratios. Initial value eual 1.
          A: matrix of floats, auxiliary matrix in the orthogonalization process.
                  A = np.zeros((M,M,1))
          q: matrix of floats, matrix with each column being the terms orthogonalized
                  by the Gram-Schmidt process. q = np.zeros_like(p)
          g: vector of floats, auxiliary vector in the orthogonalization process.
                  g = np.zeros((1,M))
        Output:
          beta: vector of floats, coefficients of the chosen terms.
          l: vector of integers, indices of the chosen terms
          M0: number of chosen terms
    '''
    

    if np.ndim(p) == 2:
        pTemp = np.zeros((np.shape(p)[0], np.shape(p)[1], 1))
        pTemp[:, :, 0] = p
        p = pTemp
        M = p.shape[1]
        l = -1*np.ones((M))
        err = np.zeros((M))
        A = np.zeros((M, M, 1))
        q = np.zeros_like(p)
        g = np.zeros((1, M))

    if np.ndim(y) == 1:
        yTemp = np.zeros((np.shape(y)[0], 1))
        yTemp[:, 0] = y
        y = yTemp

    M = p.shape[1]
    L = p.shape[2]
    gs = np.zeros((L, M))
    ERR = np.zeros((L, M))
    qs = np.copy(p)
    
    sigma = np.sum(y**2, axis=0)
            
    for m in range(M):
        if np.all(m!=l):
            ## The Gram-Schmidt method was implemented in a modified way,
            ## as shown in Rice, JR(1966)                
            for r in range(s):
                qs[:, [m], :] = qs[:, [m], :] - (np.sum(q[:, [r], :]*qs[:, [m], :], axis=0, keepdims=True)
                                                 /np.sum(q[:, [r], :]*q[:, [r], :], axis=0, keepdims=True)*q[:, [r], :])
            
            gs[:, [m]] = (np.sum(y*qs[:, m, :], axis=0, keepdims=True)/(np.sum(qs[:, m, :]*qs[:, m, :], axis=0, keepdims=True) + 1e-6)).T
            
            ERR[:, m] = (gs[:, m]**2*np.sum(qs[:, [m], :]*qs[:, [m], :], axis=0)/sigma)
            


    ERR_m = np.mean(ERR, axis=0)
        
    l[s] = np.where(ERR_m==np.nanmax(ERR_m))[0][0]
    err[s] = ERR_m[int(l[s])]
    
    r = np.arange(s)
    A[r, s, :] = np.sum(q[:, r, :]*p[:, [int(l[s])], :], axis=0)/(np.sum(q[:, r, :]*q[:, r, :], axis=0))
    A[s, s, :] = 1.0
    q[:, s, :] = qs[:, int(l[s]), :]
    g[:, s] = gs[:, int(l[s])]

    ESR = ESR - err[s]   

    ## recursive call

    if (err[s]>=pho and s<M-1):
        if verbose:
            print('term number', s)
            print('ERR', err[s])
        s += 1
        del qs
        del gs
        beta, l, M0 = mfrols(p, y, pho, s, ESR, l, err, A, q, g, verbose=verbose)
    else:
        if verbose:
            print('term number', s)
            print('ERR', err[s])
        s += 1  
        M0 = s              
        beta = np.empty((M0, L))
        for j in range(L):
            if s > 1:
                beta[:, j] = np.linalg.inv(np.squeeze(A[0:M0, 0:M0, j]))@np.transpose(g[j, 0:M0])
            else:
                beta[:, j] = (np.squeeze(A[0:M0, 0:M0,j])**-1)*g[j, 0:M0]
    return beta, l, M0

def crosscorr(x,y, alpha = 0.05):
    '''
     Computes the normalized cross-correlation (formula 5.3 from Billings, SA (2013)) between the signals x and y.
     
     written by: Renato Naville Watanabe 
     
     [phi, lags, CB] = crosscorr(x,y, alpha)
     	
     
     Inputs:
       
       x and y: vector of floats, column-vectors with the signals to compute the cross-correlation.
     
       alpha: float, significance value of the confidence boundaries. Usually is used alpha = 0.05.
     
     
     Outputs:
     
       phi: vector of floats, the normalized crosscorrelation.
     
       lags: vector of integers, vector with the corresponding lags of the phi vector.
     
       CB: vector of 2 float elements, confidence boundaries to consider that the cross-correlation at a given value is zero.
    '''
    import numpy as np
    from scipy.stats import norm
    
    x = np.reshape(x,(x.shape[0]))
    y = np.reshape(y,(y.shape[0]))
    c = np.correlate(x - np.mean(x), y - np.mean(y), mode='same')/len(x)
    phi = c/(np.std(x-np.mean(x))*np.std(y-np.mean(y)))
    N = len(x)
    CB = np.array([-norm.ppf(alpha/2)/np.sqrt(N), norm.ppf(alpha/2,0,1)/np.sqrt(N)])
    lags = np.arange(N)
    lags = lags - N//2
    return phi, lags, CB

def validation(u, xi, D, ustring='u', ystring='y'):
    '''
     Runs the tests form Eq. 5.13 in Billings (2013).
     
       written by: Renato Naville Watanabe 
     
     validation(u, xi,maxLag)
     
       Inputs:
     
        u: matrix, each column is an input signal for each trial used in the ifyMfication.
        
        xi: matrix, each column is the residual signal from the identification for each trial used in the identfication.
     
        D: vector of strings, structure of the model
     
        ustring: string, indicates the character that represents the input signal
        
        ystring: string, indicates the character that represents the output signal
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    trials = u.shape[1]  
    
    maxLag = findMaxLagFromStruct(D)
    
    u = u[maxLag:,:]
    u = u - np.mean(u, axis=0, keepdims=True)
    xi = xi - np.mean(xi, axis=0, keepdims=True)
    phi_xixi = np.zeros((u.shape[0], u.shape[1]))
    phi_uxi = np.zeros((u.shape[0], u.shape[1]))
    phi_xixiu = np.zeros((u.shape[0]-1, u.shape[1]))
    phi_u2xi = np.zeros((u.shape[0], u.shape[1]))
    phi_u2xi2 = np.zeros((u.shape[0], u.shape[1]))
    for i in range(trials):
        xiu = xi[0:-1, i]*u[0:-1, i]
        u2 = u[:,i]**2 - np.mean(u[:,i]**2)
        phi_xixi[:, i], lags, CB = crosscorr(xi[:,i], xi[:,i], 0.1)
        phi_uxi[:, i], lags, CB = crosscorr(u[:,i], xi[:,i], 0.1) 
        phi_xixiu[:, i], lags1, CB1 = crosscorr(xi[1:,i],xiu, 0.1)
        phi_u2xi[:, i], lags, CB1 = crosscorr(u2,xi[:,i], 0.1)
        phi_u2xi2[:, i], lags, CB1 = crosscorr(u2,xi[:,i]**2, 0.1)
    plt.figure()
    plt.title(['Validation Tests - trial ', i])
    plt.subplot(5, 1, 1)
    plt.plot(lags, phi_xixi)
    plt.plot(lags, np.mean(phi_xixi, axis=1), '-k', lags, CB[0]*np.ones_like(lags), '--b', lags, CB[1]*np.ones_like(lags), '--b')
    plt.ylabel(r'$\Phi_{\xi_'+ ystring +'\\xi_'+ ystring +'}$')
    plt.xlim(-3*maxLag, 3*maxLag)
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)        
    plt.plot(lags, phi_uxi)
    plt.plot(lags, np.mean(phi_uxi, axis=1), '-k', lags, CB[0]*np.ones_like(lags), '--b', lags, CB[1]*np.ones_like(lags), '--b')
    plt.ylabel(r'$\Phi_{' + ustring + '\\xi_'+ ystring+'}$')
    plt.xlim(-3*maxLag, 3*maxLag)
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 3)
    plt.plot(lags1, phi_xixiu)
    plt.plot(lags1, np.mean(phi_xixiu, axis=1), '-k', lags1, CB1[0]*np.ones_like(lags1), '--b', lags1, CB1[1]*np.ones_like(lags1), '--b')
    plt.ylabel(r'$\Phi_{\xi_' + ystring + '(\\xi_' + ystring + ustring + ')}$')
    plt.xlim(0, 3*maxLag)
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 4)
    plt.plot(lags, phi_u2xi)
    plt.plot(lags, np.mean(phi_u2xi, axis=1), '-k', lags, CB[0]*np.ones_like(lags), '--b', lags, CB[1]*np.ones_like(lags), '--b')
    plt.ylabel(r'$\Phi_{(' + ustring + '^2)\\xi_' + ystring +'}$')
    plt.ylim(-1, 1)
    plt.xlim(-3*maxLag, 3*maxLag)
    plt.subplot(5, 1, 5)
    plt.plot(lags, phi_u2xi2)
    plt.plot(lags, np.mean(phi_u2xi2, axis=1), '-k', lags, CB[0]*np.ones_like(lags), '--b', lags, CB[1]*np.ones_like(lags), '--b')
    plt.ylabel(r'$\Phi_{(' + ustring + '^2)\\xi_' + ystring + '^2}$')
    plt.ylim(-1, 1)
    plt.xlim(-3*maxLag, 3*maxLag)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
def validationTimeSeries(xi, maxLag):
    '''
     Runs the tests form Eq. 5.13 in Billings (2013).
     
       written by: Renato Naville Watanabe 
     
     validation(u, xi,maxLag)
     
       Inputs:
     
        u: matrix, each column is an input signal for each trial used in the identfication.
     
        xi: vector, is the residue obtained after the identification process.
     
        maxLag: integer, maximal lag to display in the plots. Usually it is good to plot until the maxLag of the model.
    '''
    import numpy as np
    import matplotlib.pyplot as plt    
    
    phi_xixi = np.zeros((xi.shape[0], 1))
    phi_xixi2 = np.zeros((xi.shape[0], 1))
    phi_xi2xi2 = np.zeros((xi.shape[0], 1))
    
    xi = xi - np.mean(xi)
    xi2 = xi**2 - np.mean(xi**2)
    phi_xixi, lags, CB = crosscorr(xi, xi, 0.1)
    phi_xixi2, lags, CB = crosscorr(xi, xi2, 0.1) 
    phi_xi2xi2, lags, CB = crosscorr(xi2,xi2, 0.1)
    
    plt.figure()
    plt.title(['Validation Tests - trial '])
    
    plt.subplot(3,1,1)
    plt.plot(lags, phi_xixi, '-k', lags, CB[0]*np.ones_like(lags), '--b', lags, CB[1]*np.ones_like(lags), '--b')
    plt.ylabel('Phi_{xixi}')
    plt.xlim(-maxLag, maxLag)
    
    plt.subplot(3,1,2)        
    plt.plot(lags, phi_xixi2, '-k', lags, CB[0]*np.ones_like(lags), '--b', lags, CB[1]*np.ones_like(lags), '--b')
    plt.ylabel('Phi_{xixi^2}')
    plt.xlim(-maxLag, maxLag)
    
    plt.subplot(3,1,3)
    plt.plot(lags, phi_xi2xi2, '-k', lags, CB[0]*np.ones_like(lags), '--b', lags, CB[1]*np.ones_like(lags), '--b')
    plt.ylabel('Phi_{xi^2xi^2}')
    plt.xlim(-maxLag, maxLag)
    
    plt.show()


def noiseMatrix(n, maxLag, noiseString = 'n'):
    pn = n[maxLag-1:-1,[0]]
    Dn = np.array([noiseString + '(i-1)'])
    for i in range(2, maxLag+1):
        pn = np.hstack((pn, n[maxLag-i:-i,[0]]))
        Dn = np.hstack((Dn, np.array([noiseString + '(i-' + str(i) + ')'])))
    
    return pn, Dn

def whitenSignal(x, beta, maxLag, delay):
    xw = x[maxLag:]
    for i in range(maxLag, delay+1):
        xw = xw - x[i:-(maxLag-i)]*beta[i]
    return xw

def buildMultipliedTerms(terms):
    multTerm = np.zeros((len(terms)), dtype=object)
    for i in range(len(terms)):
        multTerm[i] = terms[i][0]
        for j in range(1, len(terms[i])):
            multTerm[i] = multTerm[i] + terms[i][j]
    return multTerm
            

def pMatrix(u, y, maxLagu, maxLagy, ustring = 'u', 
            ystring = 'y', delay = 0, degree = 1, 
            constantTerm=False):
    from itertools import combinations_with_replacement
    
    maxLag = np.max(np.array([maxLagu, maxLagy]))
    
    if delay == 0:
        indD = np.array([ustring + '(i-0)'])
    else:
        indD = np.array([ustring + '(i-' + str(delay) + ')'])
        
    for i in range(delay+1, maxLagu+1):
        indD = np.hstack((indD, np.array([ustring + '(i-' + str(i) + ')'])))
    
    for i in range(1,maxLagy+1):
        indD = np.hstack((indD, np.array([ystring + '(i-' + str(i) + ')'])))    
    
    for i in range(1, degree+1):
        terms = list(combinations_with_replacement(indD.tolist(), i))
        multindD = buildMultipliedTerms(terms)
        if i == 1:
            p, _, _ = pMatrixFromStruct(u, y, 0, multindD, ustring = ustring, 
                                        ystring = ystring)
            D = indD
        else:
            pNew, _, _ = pMatrixFromStruct(u, y, 0, multindD, ustring = ustring, 
                                           ystring = ystring)
            p = np.hstack((p, pNew))
            D = np.hstack((D, multindD))
    if constantTerm:
        pNew = np.ones((p.shape[0], 1))
        p = np.hstack((pNew, p))
        indD = list(np.array(['1']))
        D = np.hstack((indD, D))
    D = D.astype(str)        
    return p, D

def findMaxLag(D, beta, signal):
    maxLag = 0
    maxOrder = 1
    
    for i in range(len(D)):
        if D[i] != '1' and beta[i] != 0:
            lagString = D[i][-2]
            if lagString == 'i':
                lag = 0
            else:
                lag = int(lagString)
            
            if lag > maxLag:
                maxLag = lag  
            order = D[i].count(')')
            if order > maxOrder and D[i].count(signal) == order:
                maxOrder = order
                
    return maxLag, maxOrder

def findMaxLagFromStruct(D):
    
    maxLag = 0
    for i in range(len(D)):
        if D[i] != '1':
            degree = D[i].count(')')            
            index = -2
            indexS = 0
            for j in range(degree):
                minusString = D[i].rfind('-',0, index)
                if minusString >= 0: 
                    lagString = D[i][minusString+1:index+1]
                else:
                    lagString = 'i'
                if lagString == 'i':
                    lag = 0
                    step = 4
                    index = index - step
                else:
                    lag = int(lagString)
                    step = 5
                    index = minusString - step
                if lag > maxLag: maxLag = lag
                
    return maxLag

def findDegreeFromStruct(D):
    
    maxDegree = 0
    for i in range(len(D)):
        if D[i] != '1':
            degree = D[i].count(')') 
            if degree > maxDegree: maxDegree = degree
            
                
    return maxDegree

def pMatrixFromStruct(u, y, n, D, ustring = 'u', 
                      ystring = 'y', nstring = 'n'):
    
    maxLag = findMaxLagFromStruct(D)
    
    p = np.ones((y.shape[0] - maxLag, len(D)))
    step = 6
    for i in range(len(D)):
        if D[i] != '1':
            degree = D[i].count(')')            
            index = -2
            indexS = 0
            for j in range(degree):
                minusString = D[i].rfind('-',0, index)
                if minusString >= 0: 
                    lagString = D[i][minusString+1:index+1]
                else:
                    lagString = 'i'
                if lagString == 'i':
                    lag = 0
                    step = 4
                    stepS = 4
                    index = index - step
                    indexS = indexS - stepS
                else:
                    lag = int(lagString)
                    step = 5
                    stepS = 3
                    index = minusString - step
                    indexS = minusString - stepS
                            
                if D[i][indexS] == ustring:
                    if (lag == 0):
                        p[:,[i]] = p[:,[i]]*u[maxLag:, [0]]
                    else:
                        p[:,[i]] = p[:,[i]]*u[maxLag-lag:-lag, [0]]
                elif D[i][indexS] == ystring:
                    p[:,[i]] = p[:,[i]]*y[maxLag-lag:-lag, [0]]
                else:
                    p[:,[i]] = p[:,[i]]*n[maxLag-lag:-lag, [0]]
    
        
    return p, D, maxLag

def pNoiseMatrix(u,y,n, maxLagu, maxLagy, maxLagNoise, 
                 ustring = 'u', ystring = 'y', 
                 nstring = 'n', delay = 0, degree = 1,
                 constantTerm=False):
    
    from itertools import combinations_with_replacement
    
    maxLag = np.max(np.array([maxLagu, maxLagy, maxLagNoise]))
    
    if delay == 0:
        indD = np.array([ustring + '(i-0)'])
    else:
        indD = np.array([ustring + '(i-' + str(delay) + ')'])
        
    for i in range(delay+1, maxLagu+1):
        indD = np.hstack((indD, np.array([ustring + '(i-' + str(i) + ')'])))
    
    for i in range(1, maxLagy+1):
        indD = np.hstack((indD, np.array([ystring + '(i-' + str(i) + ')'])))    
        
    for i in range(1, maxLagNoise+1):
        indD = np.hstack((indD, np.array([nstring + '(i-' + str(i) + ')'])))    
    
    for i in range(1, degree+1):
        terms = list(combinations_with_replacement(indD.tolist(), i))
        multindD = buildMultipliedTerms(terms)
        indToRemove = np.reshape(np.array([]), (-1, 1))
        for j in range(len(multindD)):
            if multindD[j].find('n') == -1:
                indToRemove = np.vstack((indToRemove, j))
        multindD = np.delete(multindD, indToRemove)
        if i == 1:
            p, _, _ = pMatrixFromStruct(u, y, n, multindD, ustring=ustring, 
                                        ystring = ystring)
            D = multindD
        else:
            pNew, _, _ = pMatrixFromStruct(u, y, n, multindD, ustring=ustring, 
                                           ystring = ystring)
            p = np.hstack((p, pNew))
            D = np.hstack((D, multindD))
    if constantTerm:
        pNew = np.ones((p.shape[0], 1))
        p = np.hstack((pNew, p))
        indD = list(np.array(['1']))
        D = np.hstack((indD, D))
    D = D.astype(str)
    
    return p, D

def reshapepymatrices(p, y, L):
    
    if len(p.shape) == 2:
        p = p.reshape(p.shape[0], p.shape[1], 1)
    m = int(np.floor(p.shape[0]/L))
    ptemp = np.zeros((m,p.shape[1],L*p.shape[2]))
    ytemp = np.zeros((m,L*p.shape[2]))
    for j in range(p.shape[2]):
        for i in range(L):
            ptemp[:,:,j*L+i] = p[m*i:m*(i+1),:, j]
            ytemp[:, j*L+i] = y[m*i:m*(i+1), j]
    pt = ptemp
    yt = ytemp
    return pt, yt

def reshapeyvector(y, L):
    

    m = int(np.floor(y.shape[0]/L))
    ytemp = np.zeros((m,L))
    for i in range(L):
        ytemp[:,i] = y[m*i:m*(i+1),0]
    yt = ytemp
    return yt

def executeMFrols(p,y, pho, D, L=1, supress = False):
    
    p, y = reshapepymatrices(p, y, L)    
    s = 0
    ESR = 1
    l = -1*np.ones((p.shape[1]))
    err = np.zeros((p.shape[1]))
    A = np.zeros((p.shape[1], p.shape[1], p.shape[2]))
    q = np.zeros_like(p)
    g = np.zeros((p.shape[2], p.shape[1]))
    
    beta, l, M0 = mfrols(p, y, pho, s, ESR, l, err, A, q, g, verbose= not supress)

    betatemp = np.zeros((p.shape[1],1))
    
    for i in range(M0):
        if (not supress): print(D[int(l[i])], np.mean(beta[i,:]))
        betatemp[int(l[i])] = np.mean(beta[i,:])

    beta = betatemp
    return beta

def matmulStacked(a, b):
    
    m = np.moveaxis(np.matmul(np.moveaxis(a, -1, 0), np.moveaxis(b, -1, 0)), 0, -1)
    return m

def RLS(p, y, lamb, Nmax=100000, supress=False):
    
    invLambda = 1.0/lamb
    
    beta = np.zeros((p.shape[1],p.shape[2]))
    
    P = np.repeat(1e6*np.eye(p.shape[1]).reshape(p.shape[1], -1, 1), p.shape[2], axis=2)
    
    e_beta = np.zeros((Nmax, p.shape[2]))
    betahist = np.zeros((beta.shape[0], Nmax, p.shape[2]))
    betaant = np.copy(beta)
    
    i = 0
    for N in range(Nmax):
        
        P = invLambda*(P - (invLambda*matmulStacked(matmulStacked(matmulStacked(P, np.moveaxis(p[[i], :, :], 0, 1)), p[[i],:,:]),P))/
                       (1+invLambda*matmulStacked(matmulStacked(p[[i], :, :],P), np.moveaxis(p[[i], :, :], 0, 1))))
        
        
        beta = beta + np.squeeze(matmulStacked(P, np.moveaxis(p[[i],:,:], 0, 1))*(y[i, :] - matmulStacked(p[[i],:,:],  np.moveaxis(beta.reshape(beta.shape[0], beta.shape[1], 1), 1, 2))), axis=2)
        e_beta[N,:] = np.sum((beta - betaant)**2, axis=0)
        betahist[:,N,:] = beta
        betaant = np.copy(beta)
        i = i + 1
        if (i > p.shape[0]-1):
            i = 0
            if (not supress): print(N, np.mean(e_beta[N-2,:]))
    beta = np.mean(beta, axis=1, keepdims=True)    
    return beta, e_beta, betahist
    

    

def whitenSignalIIR(b, a, x):
    xw = x
    maxLag = len(b)
    for i in range(maxLag, len(x)):
        xw[i] = x[i]
        for j in range(len(b)):
            xw[i] = xw[i] - b[j]*x[i-j]
        for j in range(len(a)):
            xw[i] = xw[i] - a[j]*xw[i-j]
    return xw

def els(u, y, n, maxLagu, maxLagy, maxLagn, structure = 'ARMA', maxIter = 6, ustring = 'u', ystring = 'y', Nmax = 20000, delay = 1, lamb = 1, supress = False):
    
   
    for k in range(maxIter):
        if (structure == 'ARMA'):
            pn, Dn = pNoiseMatrix(u, y, n, maxLagu, maxLagy, maxLagn, ystring=ystring, ustring=ustring, delay = delay)
            maxLag = np.max([maxLagu, maxLagy, maxLagn])
        if (structure == 'AR'):
            pn, Dn = pMatrix(n, y, maxLagn, maxLagy, delay = delay, ustring='n', ystring=ystring)
            maxLag = np.max([maxLagy, maxLagn])
        
#        betan, e_betan, _ = RLS(pn, y[maxLag:,[0]], lamb = lamb, Nmax=Nmax, supress = supress)
        betan = ls(pn, y[maxLag:])
#        betan = mols(pn, y[maxLag:], L = 40)
#        betan = executeMFrols(pn,y[maxLag:], pho = 1e-4, D = Dn, L = 10, supress = supress)
        
        if (structure == 'ARMA'):
            n[maxLag:,[0]] = y[maxLag:,[0]] - pn@betan
        if (structure == 'AR'):
            n[maxLag:,[0]] = y[maxLag:,[0]] - pn@betan
            
            
            
        
    return betan, pn, n, Dn

def elsWithStruct(u, y, n, D, maxIter=10, ustring='u',
                  ystring='y', nstring='n', Nmax=20000, lamb=1,
                  pho=1e-2, supress=False, L=5, method='mfrols'):
          
    
    for k in range(maxIter):
        if not supress: 
            print('ELS iteration -', k)
            print('Mounting matrix of features with noise')
        for i in range(y.shape[1]):
            pNew, Dn, maxLag = pMatrixFromStruct(u[:,[i]], y[:,[i]], 
                                                 n[:,[i]], D, 
                                                 ustring=ustring,
                                                 ystring=ystring,
                                                 nstring=nstring)
            if i == 0:
                pn = pNew.reshape(pNew.shape[0], pNew.shape[1], 1)
            else:
                pNew = pNew.reshape(pNew.shape[0], pNew.shape[1], 1)
                pn = np.concatenate((pn, pNew), axis=2)       
        
#        betan = ls(pn, y[maxLag:])
#        betan = mols(pn, y[maxLag:], L = 40)
        if k==maxIter-1: 
            supressMessage = supress
        else:
            supressMessage = True
        if method=='RLS':
                if not supress: 
                    print('Executing RLS method')
                betan, e_betan, _, betani = RLS(pn, y[maxLag:,:], lamb=lamb, Nmax=Nmax, 
                                                supress=supressMessage)
        if method=='mfrols':
                if not supress: 
                    print('Executing MFROLS method')
                betan = executeMFrols(pn, y[maxLag:,:], pho=pho, D=Dn, 
                                      L=L, supress=supressMessage)
        if method=='mols':                
                if not supress: 
                    print('Executing MOLS method')
                betan, betani = mols(pn, y[maxLag:,:], L=L)
                
        for i in range(y.shape[1]):
            if not supress: 
                    print('Computing residue of the identification')
            n[maxLag:,[i]] = y[maxLag:,[i]] - pn[:,:,i]@betani[:,[i]]
        
        if not supress:
            for i in range(len(D)):
                if betan[i,0] != 0: print(D[i], betan[i,0])
    return betan, pn, n, D

def identifyModel(u, y, maxLagu, maxLagy, ustring='u',
                  ystring='y', nstring='n', delay=0, degree=1, L=5,
                  constantTerm=True, pho = 0.01, supress=False, 
                  method='mfrols', elsMethod='mols', elsMaxIter=10):
    
    if len(y.shape) == 1:
        y = y.reshape(-1,1)
        u = u.reshape(-1,1)
    
    for i in range(y.shape[1]):
        pNew, D = pMatrix(u[:,[i]], y[:,[i]], maxLagu, maxLagy, ustring=ustring,
                          ystring=ystring, delay=delay, degree=degree,
                          constantTerm=constantTerm)
        if i == 0:
            p = pNew.reshape(pNew.shape[0], pNew.shape[1], 1)
        else:
            pNew = pNew.reshape(pNew.shape[0], pNew.shape[1], 1)
            p = np.concatenate((p, pNew), axis=2)
    
   
    beta_uy = executeMFrols(p, y[max(maxLagu, maxLagy):,:], pho, D, L=L, supress=supress)
    
    if elsMethod != 'mfrols':
       indicesToRemove = []
       for i in range(len(beta_uy)):
           if beta_uy[i] == 0:
               indicesToRemove.append(i)
       D = np.delete(D, indicesToRemove)
    
    maxLagn = findMaxLagFromStruct(D)
    degree = findDegreeFromStruct(D)
    
    ny = np.zeros((u.shape[0]-maxLagn, u.shape[1]))
    
       
    _, Dn = pNoiseMatrix(u[maxLagn:,[0]], y[maxLagn:,[0]], ny[:,[0]], maxLagu, maxLagy,
                         maxLagn, ustring=ustring, ystring=ystring, 
                         nstring=nstring, delay=1, degree=degree,
                         constantTerm=False)
      
    
    
    Dels = np.hstack((D, Dn))
    beta_uy_ELS, _, nELS, _ = elsWithStruct(u[maxLagn:,:], y[maxLagn:,:], ny, Dels,
                                            maxIter=elsMaxIter, ustring=ustring, 
                                            ystring=ystring, nstring=nstring, 
                                            supress=supress, pho=pho, L=L,
                                            method=elsMethod, Nmax=p.shape[0])
    
    beta_uy = beta_uy_ELS[0:len(D)]
    
    print('\n')
    for i in range(len(D)):
        if beta_uy[i,0] != 0: print(D[i], beta_uy[i,0])
        
    
    ny = nELS
    print('\n')
    return beta_uy, ny, D

def findModelOrder(u, y, maxOrder, method='RLS', structure = 'ARMA'):
    import matplotlib.pyplot as plt
    import numpy as np    
    
    N = len(u)
    
    j = 0
    if (structure == 'ARMA'):
        options = 2
#        AIC = np.zeros((options*(maxOrder)-delay+1, 3))
        AIC = np.zeros((maxOrder + 1, 2))
        maxLagu = 0        
        maxLagy = maxOrder
    if (structure == 'AR'):
        options = 2
        maxLagu = 1    
        maxLagy = maxOrder
#        AIC = np.zeros((options*(maxOrder)-delay+1, 3))
        AIC = np.zeros((maxOrder, 2))   
    
    while (maxLagu <= maxLagy):
                   
        if (structure == 'ARMA'):
            p, D = pMatrix(u, y, maxLagu, maxLagy, 
                           delay = 0)
            maxLag = np.max([maxLagu, maxLagy])
            lamb = 1
            beta, e_beta, _ = RLS(p, y[maxLag:], lamb,
                                  supress = True)
        
        if (structure == 'AR'):
            
            maxLag = np.max([maxLagu, maxLagy])
            lamb = 1
            beta, _, n, D, p, _ = MRLSNoise(y, maxLagu, maxLagy, maxLag, 
                                            L = 700, delay = 1, 
                                            length = 100, Nmax = 10000, 
                                            supress = True)
                   
        n = y[maxLag:] - p@beta
                    
        if (structure == 'AR'):
            sumU, sumY = coefSum(beta, D)
            AIC[j, 0] = sumU
#            AIC[j, 0] = N*np.log(np.var(n))+np.log(N)*len(beta)
            print(maxLagu, maxLagy, sumU, sumY, sumU + sumY)
#            AIC[j, 0] = N*np.log(np.var(n)) + 2*len(beta)
#            AIC[j, 0] = N*np.log(np.var(n)) + N*np.log((N+len(beta))/(N-len(beta)))
        if (structure == 'ARMA'):
            AIC[j, 0] = N*np.log(np.var(n))+np.log(N)*len(beta)
#            AIC[j, 0] = sumU
#                AIC[j, 0] = N*np.log(np.var(n)) + 2*len(beta) + (2*len(beta)*(len(beta)+1))/(N-len(beta)-1)
#                AIC[j, 0] = N*np.log(np.var(n)) + N*np.log((N+len(beta))/(N-len(beta)))
        AIC[j, 1] = maxLagu
        
        
        j = j + 1
        
        maxLagu = maxLagu + 1
          
    if (structure == 'ARMA'):
        maxLagu = int(AIC[detectDifference(AIC[:,0]),1])
#        maxLagu = int(AIC[np.argmin(np.abs(np.diff(AIC[:,0]))),1])
#        maxLagu = AIC[np.argmin(AIC[:,0]),1]
#        bestOrder = AIC[np.arange(0, AIC.shape[0]-1,1)[np.abs(np.diff(AIC[:,0]))<5e-3][0]+1,1:]
    if (structure == 'AR'):
#        maxLagu = int(AIC[detectDifference(AIC[:,0]),1])
        maxLagu = int(AIC[np.argmin(np.abs(np.diff(AIC[:,0]))),1])
    
    print('maxLagu', maxLagu)
    
    plt.figure()
    plt.plot(AIC[:,1],AIC[:,0])
    plt.xlabel('maxLagu')
    plt.show()
    
    j = 0
    if (structure == 'ARMA'):
        options = 2
#        AIC = np.zeros((options*(maxOrder)-delay+1, 3))
        AIC = np.zeros((maxOrder+1-maxLagu, 2))
        maxLagy = maxLagu
    if (structure == 'AR'):
        options = 2
        maxLagy = maxLagu
#        AIC = np.zeros((options*(maxOrder)-delay+1, 3))
        AIC = np.zeros((maxOrder-maxLagu+1, 2))
    
    while (maxLagy <= maxOrder):                   
        if (structure == 'ARMA'):
            p, D = pMatrix(u, y, maxLagu, maxLagy, delay = 0)
            maxLag = np.max([maxLagu, maxLagy])
            lamb = 1
            beta, e_beta, _ = RLS(p, y[maxLag:], lamb,
                                  supress = True)
        
        if (structure == 'AR'):
            
            maxLag = np.max([maxLagu, maxLagy])
            lamb = 1
            beta, _, n, D, p, _ = MRLSNoise(y, maxLagu, maxLagy, maxLag,
                                            L = 1000, delay = 1,
                                            length = 100, Nmax = 10000,
                                            supress = True)
                   
        n = y[maxLag:] - p@beta
                    
        if (structure == 'AR'):
            sumU, sumY = coefSum(beta, D)
#            AIC[j, 0] = sumY
            AIC[j, 0] = N*np.log(np.var(n))+np.log(N)*len(beta)
            print(maxLagy, maxLagy, sumU,sumY, sumU+ sumY)
#            AIC[j, 0] = N*np.log(np.var(n)) + 2*len(beta) + (2*len(beta)*(len(beta)+1))/(N-len(beta)-1)
            
        if (structure == 'ARMA'):
            sumU, sumY = coefSum(beta, D)
            AIC[j, 0] = N*np.log(np.var(n))+np.log(N)*len(beta)
#            AIC[j, 0] = sumY
            print(maxLagy, maxLagy, sumU,sumY, sumU + sumY)
#            AIC[j, 0] = N*np.log(np.var(n)) + 2*len(beta) + (2*len(beta)*(len(beta)+1))/(N-len(beta)-1)
#                AIC[j, 0] = N*np.log(np.var(n)) + N*np.log((N+len(beta))/(N-len(beta)))
        AIC[j, 1] = maxLagy
                
        j = j + 1
        
        maxLagy = maxLagy + 1
    
    if (structure == 'ARMA'):
        maxLagy = int(AIC[detectDifference(AIC[:,0]),1])
#        maxLagy = int(AIC[np.argmin(np.abs(np.diff(AIC[:,0]))),1])
#        bestOrder = AIC[np.arange(0, AIC.shape[0]-1,1)[np.abs(np.diff(AIC[:,0]))<5e-3][0]+1,1:]
    if (structure == 'AR'):
        maxLagy = int(AIC[detectDifference(AIC[:,0]),1:])
#        maxLagy = int(AIC[np.argmin(np.abs(np.diff(AIC[:,0]))),1])
    
    print('maxLagy ', maxLagy)
    
    plt.figure()
    plt.plot(AIC[:,1],AIC[:,0])
    plt.xlabel('maxLagy')
    plt.show()   
    
    if structure == 'AR':
        delayMin = 1
    if structure == 'ARMA':
        delayMin = 0   
  
    delayMax = maxLagy
    AICdelay = 1000*np.ones((delayMax+1-delayMin, 2))
    
    j = 0
    if (AICdelay.shape[0] == 1):
        delay = delayMin
    else:
        for i in range(delayMax, delayMin-1,-1):
            if (structure == 'ARMA'):
                p, D = pMatrix(u, y, maxLagy, maxLagy,
                               delay = i)
                maxLag = maxLagy
                lamb = 1
                beta, e_beta, _ = RLS(p, y[maxLag:], lamb,
                                     supress = True)
                    
                
            if (structure == 'AR'):
                p, D = noiseMatrix(y, maxLagy)
                maxLag = np.max([maxLagy, maxLagy])
                lamb = 1
                beta, _, n, D, p, _ = MRLSNoise(y, maxLagy, 
                                                maxLagy, maxLag, 
                                                L = 200, delay = i, 
                                                length = 100, Nmax = 10000, 
                                                supress = True)
            
            
            n = y[maxLag:] - p@beta
            
            sumU, sumY = coefSum(beta, D)
            print(maxLagu, maxLagy, i, sumU, sumY, sumU+sumY)
            
    #        AICdelay[i, 0] = np.mean(n1.shape[0]*np.log(np.var(n1,axis = 0)))+0*np.log(n1.shape[0])*len(beta)
            AICdelay[i, 0] = N*np.log(np.var(n)) + 2*len(beta)
            
    #        AICdelay[j, 0] = sumU+sumY
    #        AICdelay[i, 0] = N*np.log(np.var(n))+N*np.log((N+len(beta))/(N-len(beta)))
            AICdelay[j, 1] = i
            j = j + 1
    
        plt.figure()
        plt.plot(AICdelay[:, 1], AICdelay[:,0])
        plt.xlabel('delay')
        plt.show()
            
    
    #    delay = int(AICdelay[np.argmin(np.abs(np.diff(AICdelay[:,0]))),1])
        delay = int(AICdelay[np.argmin(AICdelay[:,0]),1])
        print('delay', delay)
    
    if delay>maxLagu: maxLagu = delay
    
    
       
    return maxLagu, maxLagy, AIC, delay, AICdelay

def detectDifference(x):
    grad = np.diff(x)    
    if ((grad>0).all()):
        index = np.argmin(x)
    else:
        index = np.argmin(grad) + 1
    return index

def kmeans(x, k):
    '''
    clusters = kmeans(x, k)
    
    Inputs: 
    - x: numpy array with one element per line. It can have many columns.
    - k: integer with the number of clusters
    
    Outputs:
    - clusters: list with the length equal to the number of clusters. Each element 
of the list contains a numpy array with the indices of the elements of x that are
in the respective cluster.      
    '''
    import numpy as np
    clusters = list(np.arange(k))
          
    for j in range(len(x)):
        chooseCluster = np.random.rand(1)
        for i in range(k):
            if (chooseCluster > i/k and chooseCluster <= (i+1)/k):
                if (not np.iterable(clusters[i])):
                    clusters[i] = np.array([j])
                else:
                    clusters[i] = np.append(clusters[i], j)
        
    for w in range(10):
        means = np.zeros((k,x.shape[1]))
        for i in range(k):
            means[i,:] = np.mean(x[clusters[i],:], axis = 0)
           
        newClusters = list(np.arange(k))
        for j in range(len(x)):
            distance = np.zeros((k))
            for i in range(k):
                distance[i] = np.sum((x[j,:] - means[i,:])**2)
            chosenCluster = np.argmin(distance)
            if (not np.iterable(newClusters[chosenCluster])):
                newClusters[chosenCluster] = np.array([j])
            else:
                newClusters[chosenCluster]  = np.append(newClusters[chosenCluster] , j)
        clusters = newClusters
    
    
    return clusters

def simplifyTransferFunction(G, eps = 1e-3):
    import control
    import numpy as np
    
    num = control.tfdata(G)[0][0][0]
    den = control.tfdata(G)[1][0][0]
    
    
        
    
    g = num[0]
    z = np.roots(num)
    p = np.roots(den)
       
    zi = np.array([])
    pi = np.array([])    
    
    for i in range(len(z)):
        for j in range(len(p)):
            if (np.abs(np.real(z[i]) - np.real(p[j])) < eps and 
                np.abs(np.imag(z[i]) - np.imag(p[j])) < eps and 
                (not np.any(pi == j) and not np.any(zi == i))):
                zi = np.append(zi, i)
                pi = np.append(pi, j)
    
    zeros = np.zeros((len(z)-len(zi)), complex)
    poles = np.zeros((len(p)-len(pi)), complex)
       
    
    k = 0
    for i in range(len(z)):
        if (not np.any(zi == i)):
            zeros[k] = z[i]
            k = k + 1
            
    k = 0
    for j in range(len(p)):
        if (not np.any(pi == j)):
            poles[k] = p[j]
            k = k + 1
    
       
    num = g*np.poly(zeros)
    den = np.poly(poles)
    
    
    for i in range(len(num)):
        if np.abs(num[i]) < eps*np.max(np.abs(num)):
            num[i] = 0
            
    for i in range(len(den)):
        if np.abs(den[i]) < eps*np.max(np.abs(den)):
            den[i] = 0
    
    continueFlag = True
    
    while (continueFlag):
        if (num[0] == 0):
            num = num[1:]
        else:
            continueFlag = False
            
    continueFlag = True
    
    while (continueFlag):
        if (den[0] == 0):
            den = den[1:]
        else:
            continueFlag = False
    
    Gsimp = control.tf(num, den, True)
    
    return Gsimp

def buildTransferFunction(beta, D, ustring = 'u', ystring = 'y', eps = 1e-3, noise = True):
    import control
    maxLag = 0
    for i in range(len(D)):
        lagString = D[i][-2]
        if lagString == 'i':
            lag = 0
        else:
            lag = int(lagString)
        if lag > maxLag:
            maxLag = lag
    num = np.zeros((maxLag+1,1))
    den = np.zeros((maxLag+1,1))
    
    den[0] = 1
    if noise: num[0] = 1
    
    for i in range(len(D)):
        lagString = D[i][-2]
        if lagString == 'i':
            lag = 0
        else:
            lag = int(lagString)
        if D[i][0] == ustring:
            num[lag] = beta[i]
        else:
            den[lag] = -beta[i]
        
    G = simplifyTransferFunction(control.tf(np.squeeze(num), 
                                            np.squeeze(den), True), eps)
    
    return G

def coefSum(beta, D, ustring = 'u', ystring = 'y'):
    
    maxLag = 0
    for i in range(len(D)):
        lagString = D[i][-2]
        if lagString == 'i':
            lag = 0
        else:
            lag = int(lagString)
        if lag > maxLag:
            maxLag = lag
    num = np.zeros((maxLag+1,1))
    den = np.zeros((maxLag+1,1))
    
    den[0] = 1
   
    
    for i in range(len(D)):
        lagString = D[i][-2]
        if lagString == 'i':
            lag = 0
        else:
            lag = int(lagString)
        if D[i][0] == ustring:
            num[lag] = beta[i]
        else:
            den[lag] = -beta[i]
        
    sumU = np.sum(num)
    sumY = np.sum(den)
    
    return sumU, sumY

def RLSnoise(y, lamb, maxLagy, maxLagn, delay = 1, Nmax = 100000,
             supress = False, ustring = 'u', ystring = 'y', nstring = 'n'):
    
    invLambda = 1.0/lamb
    
    maxLag = np.max([maxLagy, maxLagn])
    
    beta = np.zeros((maxLagy+maxLagn+1-delay,1))
    P = 1e6*np.eye(beta.shape[0])
    
    e_beta = np.zeros((Nmax, 1))
    betaant = 1*beta
    i = maxLag+1
    betahist = np.zeros((beta.shape[0], Nmax))
    N = 0
        
    n = np.random.randn(y.shape[0],1)
   
    
    
    while (N < Nmax):
        pVec = np.reshape(np.vstack((n[i-delay:i-maxLagn-1:-1,[0]],y[i-1:i-maxLagy-1:-1,[0]])),(1,-1))
        
        K = (invLambda*P@pVec.T)/(1+invLambda*pVec@P@pVec.T)
        beta = beta + K*(y[i] - pVec@beta)
        P = invLambda*(P - K@pVec@P)
#        
#        P = invLambda*(P - (invLambda*P@pVec.T@pVec@P)/(1+invLambda*pVec@P@pVec.T))
#        beta = beta + P@pVec.T*(y[i] - pVec@beta)
        
#        beta = beta + 0.01*pVec.T*(y[i] - pVec@beta)
        
        e_beta[N] = (beta - betaant).T@(beta - betaant)
        betaant = 1*beta
        betahist[:,[N]] = beta
        n[i] = y[i] - pVec@beta
        i = i + 1
        N = N + 1
        if (i > len(y)-1):
            i = maxLag + 1
            if (not supress): print(N, e_beta[N-1])
            p, D = pMatrix(n, y, maxLagn, maxLagy,
                           delay = delay, ustring = nstring,
                           ystring = ystring)
            n[maxLag:,[0]] = y[maxLag:,[0]] - p@beta  
            
#        
                
    
    p, D = pMatrix(n, y, maxLagn, maxLagy,
                   delay = delay, ustring = nstring,
                   ystring = ystring)
    
#    beta = np.mean(betahist[:,int(Nmax/2):], axis = 1, keepdims = True)
    
#    beta = ls(p, y[maxLag:])
    
    return beta, e_beta, D, p, n, betahist

def MVInoise(y, maxLagy, maxLagn, delay = 1, Nmax = 100000,
             supress = False, ustring = 'u', ystring = 'y', nstring = 'n'):
    
   
    
    maxLag = np.max([maxLagy, maxLagn])
    
    beta = np.zeros((maxLagy+maxLagn+1-delay,1))
    M = 1e6*np.eye(beta.shape[0])
    
    e_beta = np.zeros((Nmax, 1))
    betaant = 1*beta
    i = 2*maxLag+1
    betahist = np.zeros((beta.shape[0], Nmax))
    N = 0
        
    n = 0*np.ones_like(y)
    yest = np.zeros_like(y)
    
    while (N < Nmax):
#        for j in range(maxLag+1,i):
#            pVec = np.reshape(np.vstack((n[j-delay:j-maxLagn-1:-1,[0]],
#                                         y[j-1:j-maxLagy-1:-1,[0]])),(1,-1))
#            yest[i] = pVec@beta
        pVec = np.reshape(np.vstack((n[i-delay:i-maxLagn-1:-1,[0]],
                                     y[i-1:i-maxLagy-1:-1,[0]])),(1,-1))
        yest[i] = pVec@beta
        zVec = np.reshape(np.vstack((n[i-delay:i-maxLagn-1:-1,[0]],
                                     n[i-maxLagn-1:i-maxLagn-maxLagy-1:-1,[0]])),(1,-1))
        
        M = (M - (M@zVec.T@pVec@M)/(1+pVec@M@zVec.T))
        beta = beta + M@zVec.T*(y[i] - pVec@beta)
        
        e_beta[N] = (beta - betaant).T@(beta - betaant)
        betaant = 1*beta
        betahist[:,[N]] = beta
        n[i] = y[i] - pVec@beta
        i = i + 1
        N = N + 1
        if (i > len(y)-1):
            i = 4*maxLag + 1
            if (not supress): print(N, e_beta[N-1])
            p, D = pMatrix(n, y, maxLagn, maxLagy,
                           delay = delay, ustring = nstring,
                           ystring = ystring)
            n[maxLag:,[0]] = y[maxLag:,[0]] - p@beta  
            while i<4*maxLag+1000:
                pVec = np.reshape(np.vstack((n[i-delay:i-maxLagn-1:-1,[0]],y[i-1:i-maxLagy-1:-1,[0]])),(1,-1))
                zVec = np.reshape(np.vstack((n[i-maxLagn-1:i-2*maxLagn + delay-2:-1,[0]],
                                             y[i-maxLagy-1:i-2*maxLagy-1:-1,[0]])),(1,-1))
                M = (M - (M@zVec.T@pVec@M)/(1+pVec@M@zVec.T))
                beta = beta + M@zVec.T*(y[i] - pVec@beta)
                i = i + 1
#        
                
    
    p, D = pMatrix(n, y, maxLagn, maxLagy,
                   delay = delay, ustring = nstring,
                   ystring = ystring)
    
#    beta = np.mean(betahist[:,int(Nmax/2):], axis = 1, keepdims = True)
    
#    beta = ls(p, y[maxLag:])
    
    return beta, e_beta, D, p, n, betahist

def LMSnoise(y, maxLagy, maxLagn, delay = 1, Nmax = 100000,
             supress = False, ustring = 'u', ystring = 'y', nstring = 'n'):
   
    
    maxLag = np.max([maxLagy, maxLagn])
    
    beta = np.zeros((maxLagy+maxLagn+1-delay,1))
    
    
    e_beta = np.zeros((Nmax, 1))
    betaant = 1*beta
    i = 2*maxLag+1
    betahist = np.zeros((beta.shape[0], Nmax))
    N = 0
        
    n = 1*y
   
    
    while (N < Nmax):
        pVec = np.reshape(np.vstack((n[i-delay:i-maxLagn-1:-1,[0]],y[i-1:i-maxLagy-1:-1,[0]])),(1,-1))
         
        beta = beta + 0.03*pVec.T*(y[i] - pVec@beta)
        
        e_beta[N] = (beta - betaant).T@(beta - betaant)
        betaant = 1*beta
        betahist[:,[N]] = beta
        n[i] = y[i] - pVec@beta
#        for j in range(i-maxLag,i+1):
#            pVec = np.reshape(np.vstack((n[j-delay:j-maxLagn-1:-1,[0]],y[j-1:j-maxLagy-1:-1,[0]])),(1,-1))
            
        i = i + 1
        N = N + 1
        if (i > len(y)-1):
            i = 2*maxLag + 1
            if (not supress): print(N, e_beta[N-1])
            p, D = pMatrix(n, y, maxLagn, maxLagy,
                           delay = delay, ustring = nstring,
                           ystring = ystring)
            n[maxLag:,[0]] = y[maxLag:,[0]] - p@beta  
            
#        
                
            
    
    p, D = pMatrix(n, y, maxLagn, maxLagy,
                   delay = delay, ustring = nstring,
                   ystring = ystring)
    
#    beta = np.mean(betahist[:,int(Nmax/2):], axis = 1, keepdims = True)
    
#    beta = ls(p, y[maxLag:])
    
    return beta, e_beta, D, p, n, betahist
    
def RLSWhiteningNoise(y, maxLagy, maxLagn, delay = 1, Nmax = 100000,
             supress = False, ustring = 'u', ystring = 'y', nstring = 'n'):
    
    maxLag = np.max([maxLagy, maxLagn])
    
    beta = np.zeros((maxLagy+maxLagn+1-delay,1))
    
    betaNoise = np.zeros((maxLagn+1-delay,1))
    betaOutput = np.zeros((maxLagy,1))
    
    Pn = 1e6*np.eye(betaNoise.shape[0])
    Py = 1e6*np.eye(betaOutput.shape[0])
    
    e_beta = np.zeros((Nmax, 1))
    betaant = 1*beta
    i = maxLag+1
    betahist = np.zeros((beta.shape[0], Nmax))
    N = 0
        
    n = 0*y
    yw = 1*y
    while (N < Nmax):
        p, D = pMatrix(n, y, maxLagn, maxLagy,
                   delay = delay, ustring = nstring,
                   ystring = ystring)
        n[maxLag:] = y[maxLag:] - p@beta
        i = maxLag+1
        while i < len(y) and N < Nmax:
            
            pnVec = np.reshape(n[i-delay:i-maxLagn-1:-1,[0]],(1,-1))
            
            
            Kn = (Pn@pnVec.T)/(1+pnVec@Pn@pnVec.T)
            betaNoise = betaNoise + Kn*(n[i] - pnVec@betaNoise)
            Pn = (Pn - Kn@pnVec@Pn)
            
            yw[i] = whitenSignal(y[i-maxLagn:i+1,[0]], betaNoise, maxLagn, delay)
            pyVec = np.reshape(yw[i-1:i-maxLagy-1:-1,[0]],(1,-1))
            
            Ky = (Py@pyVec.T)/(1+pyVec@Py@pyVec.T)
            betaOutput = betaOutput + Ky*(yw[i] - pyVec@betaOutput)
            Py = Py - Ky@pyVec@Py
            
            beta = np.vstack((betaNoise, betaOutput))
            
            e_beta[N] = (beta - betaant).T@(beta - betaant)
            betaant = 1*beta
            betahist[:,[N]] = beta
            n[i] = yw[i] - pyVec@betaOutput
            N = N + 1
            i = i + 1
        if (not supress): print(N, e_beta[N-1])               
    
    p, D = pMatrix(n, y, maxLagn, maxLagy,
                   delay = delay, ustring = nstring,
                   ystring = ystring)
    
#    n = y[maxLag:] - p@beta
#    p, D = pMatrix(n, y[maxLag:], maxLagn, maxLagy,
#                   delay = delay, ustring = nstring,
#                   ystring = ystring)
##    
#    beta = np.mean(betahist[:,int(Nmax/2):], axis = 1, keepdims = True)
    
#    beta = mls(p, y[2*maxLag:], L = 40)
    
    return beta, e_beta, D, p, n, betahist

def MRLSNoise(y, maxLagu, maxLagy, residueLag, L = 2, 
              delay = 1, length = 100, Nmax = 100000, supress = False, 
              ustring = 'u', ystring = 'y', nstring = 'n'):
    

    maxLag  = np.max([maxLagu, maxLagy])
    
    lamb = 1
    
    betaTrials = np.zeros((maxLagu+maxLagy + 1 - delay, L))
    
    uest = 1*y
    
    for i in range(L):
        begin = int(np.random.randint(maxLag+1,len(y)-length))
        end = begin+length
        betaTrials[:,[i]], e_beta, D, p, _, betahist = RLSnoise(y[begin:end,[0]], 
                                                                lamb, maxLagy, 
                                                                maxLagu, 
                                                                supress = supress,
                                                                ystring=ystring,
                                                                Nmax = Nmax,
                                                                delay = delay, 
                                                                nstring=ustring)
        
        
        uest[begin+maxLag:end] = y[begin+maxLag:end,[0]] - p@np.mean(betaTrials[:,:i], axis = 1, keepdims = True)
    
    
    if len(uest[np.isnan(uest)])>0: 
        continueFlag = True
    else:
        continueFlag = False
    while continueFlag:
        begin = np.arange(0,len(uest),1)[np.squeeze(np.isnan(uest))][0] - maxLag
        end = begin+length
        beta1, e_beta, D, p, _, betahist = RLSnoise(y[begin:end,[0]],
                                                     lamb, maxLagy,
                                                     maxLagu,
                                                     supress = supress,
                                                     ystring=ystring,
                                                     Nmax = Nmax,
                                                     delay = delay, 
                                                     nstring=ustring)
        
        betaTrials = np.hstack((betaTrials, beta1))
        uest[begin+maxLag:end] = y[begin+maxLag:end,[0]] - p@np.mean(betaTrials, axis = 1, keepdims = True)
        if len(uest[np.isnan(uest)])>0: 
            continueFlag = True
        else:
            continueFlag = False
    beta1 = np.mean(betaTrials, axis = 1, keepdims = True)
    
    p, D = pMatrix(uest, y, maxLagu, maxLagy, 
                   ustring=ustring,
                   ystring=ystring, 
                   delay = delay)
    
    uest[maxLag:] = y[maxLag:] - p@beta1
    
    D1 = np.hstack((np.array([ustring + '(i)']), D))
       
    p,_,_ = pMatrixFromStruct(uest,y, y, D1, 
                              ustring = ustring,
                              ystring = ystring,
                              nstring = nstring)
    
    beta = np.vstack((np.array([1]),beta1))
    
    n = y[maxLag:] - p@beta
    
    return beta,uest, n, D1, p, betahist


def  osa(u, y, beta, degree, maxLagu, maxLagy, delay, constantTerm = False):
    '''
    % Implements the one-step-ahead prediction algorithm  (page 124 from Billings, SA (2013)).
    %
    % written by: Renato Naville Watanabe 
    %
    % [yest xi] = osa(u, y, beta, l, degree, mu, my, delay)
    %
    % Inputs:
    %	
    %   u: vector of floats, input signal.
    %
    %   y: vector of floats, output signal.
    %
    %   beta: vector of floats, the coefficients of the model terms.
    %
    %   l: vector of integers, the indices of the model terms, sorted in the same order of the beta vector. 
    %   It works together with the buildPMatrix function.
    %
    %   degree: integer, the maximal polynomial degree that you want the FROLS method to look for (it has been tested until 
    %   the 9th degree).
    % 	
    %   maxLagu: integer, maximal lag of the input signal.
    %
    %   maxLagy: integer, maximal lag of the output signal.
    %
    %   delay: integer, how much lags you want to not consider in the input terms. It comes from a previous knowledge of your system.
    %
    % Outputs:
    %
    %   yest: vector of floats, the estimated output vector.
    %
    %   xi: vector of floats, the residue of the estimation.
    '''
    
    
    p, _ = pMatrix(u, y, maxLagu, maxLagy, delay=delay, 
                   degree = degree, 
                   constantTerm=constantTerm)
    
    yest = p@beta
    xi = y[max(maxLagu, maxLagy):] - yest
    
    return yest, xi

def findStructWithMaxDegree(D, beta, maxDegree):
    
    indices = list()
    for i in range(len(D)):
        if D[i] != '1':
            degree = D[i].count(')')
        else:
            degree = 0
        if degree <= maxDegree:
            indices.append(i)
     
    D = D[indices]
    beta = beta[indices]
    
    return D, beta

def  osaWithStruct(u, y, beta, D, degree=[], ustring='u', ystring='y'):
    '''
    % Implements the one-step-ahead prediction algorithm  (page 124 from Billings, SA (2013)).
    %
    % written by: Renato Naville Watanabe 
    %
    % [yest xi] = osa(u, y, beta, l, degree, mu, my, delay)
    %
    % Inputs:
    %	
    %   u: vector of floats, input signal.
    %
    %   y: vector of floats, output signal.
    %
    %   beta: vector of floats, the coefficients of the model terms.
    %
    %   l: vector of integers, the indices of the model terms, sorted in the same order of the beta vector. 
    %   It works together with the buildPMatrix function.
    %
    %   degree: integer, the maximal polynomial degree that you want the FROLS method to look for (it has been tested until 
    %   the 9th degree).
    % 	
    %   maxLagu: integer, maximal lag of the input signal.
    %
    %   maxLagy: integer, maximal lag of the output signal.
    %
    %   delay: integer, how much lags you want to not consider in the input terms. It comes from a previous knowledge of your system.
    %
    % Outputs:
    %
    %   yest: vector of floats, the estimated output vector.
    %
    %   xi: vector of floats, the residue of the estimation.
    '''
    
    if degree != []:
        D, beta = findStructWithMaxDegree(D, beta, degree)
    
    p, _, maxLag = pMatrixFromStruct(u, y, 0, D, ustring = ustring, 
                             ystring = ystring)
    
    yest = p@beta
    xi = y[maxLag:] - yest
    
    return yest, xi, maxLag