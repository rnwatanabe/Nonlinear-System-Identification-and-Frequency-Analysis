#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:01:24 2019

@author: Renato Naville Watanabe
"""

import numpy as np




def computeSystemGFRF(Da, Fs, a, degree, ustring='u', ystring='y', noiseFlag=False):
    '''        
    % Computes the GFRFs of a NARX model. It uses the Symbolic Toolbox of Matlab.
    %
    % written by: Renato Naville Watanabe 
    %
    % Hn = computeSystemGFRF(Da, Fs, a, degree)
    %	
    %
    % Inputs:
    %	
    %   Da: array, contains the strings with the NARX model terms.
    %
    %   Fs: float, the sampling frequency, in Hz.
    %
    %   a: array of floats, coefficients of the NARX model.
    %
    %   degree: integer, maximal order of GFRF you want to obtain.
    %
    %
    % Outputs:
    %   
    %   Hn: cell, contains the GFRFs of the NARX model.
    '''
    import copy
        
    D = copy.copy(Da)
    ar = copy.copy(a)
    D, ar, _ = removeConstant(D, ar, np.zeros_like(a))
    
    I = modelLags(D)
    
    C, maxLag = findCCoefficients(ar, I, ustring, ystring)
    Hnn, Hn = buildHn(degree, C, Fs, maxLag, noiseFlag)
    
    return Hn

def removeConstant(Da, a, la):
    '''
    % Function to remove a constant term from the identified model, in the case of the constant term ('1') was 
    % identified. This is necessary for the computation of GFRFs.
    %
    % written by: Renato Naville Watanabe 
    %
    % Da, a = removeConstant(Da, a)
    %	
    % Inputs:
    %
    %   Da: cell, contains strings with the NARX model terms.
    %
    %   a: vector,  coefficients of the NARX model.
    %
    %   la: vector,  indices of the chosen terms during the identification of
    %   the system. Obtained from buildPMatrix function.
    %   
    % Outputs:
    %
    %   The Da, a and la vectors are the same of the input, except for the deletion of the constant term. 

    
    '''
    termsNumber = len(Da)        
    
    removed = False
    i = 0
    while i < termsNumber and not removed:
            if Da[i] == '1':
                    if termsNumber > i:
                            Da[i:-1] = Da[i+1:]
                            a[i:-1,:] = a[i+1:,:]
                            la[i:-1,:] = la[i+1:,:]
                    a = a[:-1,:]
                    Da = Da[:-1]
                    la = la[0:-1]
                    removed = True
            i = i + 1
       
    return Da, a, la


def modelLags(D):
    '''
    % Obtain a vector formed by integers corresponding to the ASCII code of u or y of the identified terms 
    % and the correspondibg lags. It is used during the identification of the NARMAX model to read 
    % the vector of strings D.
    %
    % written by: Renato Naville Watanabe 
    %
    % ind = modelLags(D)
    %	
    % Inputs:
    %
    %   D: cell, contains the strings with the terms of a NARX model.
    %
    %   
    % Outputs:
    %
    %   ind is a vector with the following format:
    %   ___________________
    %   | 117 or 121 or 101|  % 117 corresponds to the input u, 121  to output y and 101 to the residue e.
    %   |      lag         |  % lag corresponds to the lag of the signal in the preceding cell.
    %   this format of vector continues to represent a term of multiplied inputs, outputs and residue.
    %   Example:
    %   u(k-1)u(k-2)y(k-5)e(k-2) corresponds to:
    %         ___________
    %         |   117   |
    %         |    1    |
    %   ind = |   117   |
    %         |    2    |
    %         |   121   |
    %         |    5    |
    %         |   101   |
    %         |    2    |
    %         |_________|
    %

    '''
    from parse import parse
    
    I = dict()
    for j in range(len(D)):
            degree = D[j].count(')')
            I[j] = list(parse('{}(i-{})'*degree, D[j]))           
    
    ind = I
    
    return ind

def findCCoefficients(a, I, ustring='u', ystring='y'):
        '''
        % Find the coefficients of the coefficients in the format of Equation 6.46 of Billings (2013).
        %
        % written by: Renato Naville Watanabe 
        %
        % [C, maxLag] = findCCoefficients(a, I)
        %	
        %
        % Inputs:
        %   
        %   a: vector of floats, coefficients of the NARX model.
        %
        %   I: dict, obtained from the modelLags functions.
        %
        %
        % Outputs:
        %   
        %   C: struct, contains the coefficients of the model, in the format of Equation 6.46 of Billings (2013):
        %   C.c_pq where:
        %   p is  the number that the output signal appears in a term.
        %   q is  the number that the input signal appears in a term.
        %   For example, a model with the following terms:
        %   y(k) = -2*y(k-1) + 4*y(k-2) - 1.5*u(k-5) + 10.5*u(k-6)y(k-2)
        %   has the following C struct:
        %   C.c_10 = [-2 4]
        %   C.c_01 = [0 0 0 0 -1.5]
        %   C.c11 = [0 0 0 0 0 0;...
        %            0 0 0 0 0 10.5]
        %
        %   maxLag: integer, maximal Lag of the model.
        '''
        Mp = len(a)
        maxLag = 0
        C = dict()
        for i in range(Mp):
                for j in range(0, len(I[i]), 2):       
                        if int(I[i][j+1]) > maxLag:
                                maxLag = int(I[i][j+1])
                                
        for i in range(Mp):
               lags_y = ''
               lags_u = ''
               p = 0
               q = 0
               for j in range(0, len(I[i]), 2):       
                   
                   
                   if I[i][j] == ystring:
                      if (len(lags_y) == 0): 
                          lags_y = str(I[i][j+1]) 
                      else:
                          lags_y = lags_y + ',' + str(I[i][j+1]) 
                      
                      p = p + 1
                   else:
                      if len(lags_u) == 0: 
                          lags_u = str(I[i][j+1])
                      else:
                          lags_u = lags_u + ',' + str(I[i][j+1]) 
                      
                      q = q + 1
                   
               
               if len(lags_y)>0 and len(lags_u)>0:
                  lags_y = lags_y + ','
               
               lags = lags_y + lags_u
               
               if "c_" + str(p) + str(q) in C.keys():
                       pass
               else:
                       degree = int(len(I[i])/2)
                       exec('C["c_' + str(p) + str(q) + '"] = np.zeros((' + (str(maxLag+1) + ',')*(degree-1) + str(maxLag+1)+'))')
               
               
               exec('C["c_' + str(p) + str(q) + '"][' + lags + '] = ' + str(a[i][0]))
               
        return C, maxLag     

  
def buildHn(order, C, fs, maxLag, noiseFlag=False):
        ''' 
        %% Implements the algorithm to find the GFRFs of a NARX model. It is the
        %	Equation (6.47) from Billings, 2013. It requires the Symbolic Matlab Toolbox.
        %	
        % written by: Renato Naville Watanabe 
        %
        % Hnn, Hn] = buildHn(order, C, fs, maxLag, noiseFlag)
        %	
        % Inputs:
        %
        %   order: integer, order of the GFRF you want to obtain. When you call it you should 
        %   put as entry the maximal order of GFRF you want to obtain. As it is a recursive algorithm, 
        %   it will give you as output all the GFRF with order lower and equal than the order number.
        %
        %   C: dict, obtained from the findCCoefficients function.
        %
        %   fs: float, is the sampling frequency of the data, in Hz.
        %
        %   maxLag: integer, is the maximal Lag of the model.
        %
        %   noiseFlag: boolean, is an indicator used for NonLinear Partial Directed Coherence. Normally set it to 0.
        %
        %
        % Outputs:
        %
        %   Hnn are the intermediate functions for the GFRF computation.
        %
        %   Hn is a cell with the GFRFs.
        '''
        
        import sympy as sym
        globals().update(C)
        locals().update(C)
        
        n = order
        
        globals()['Hn'] = dict()
        globals()['Hnn'] = dict()
        fvector = dict()
        lvector = dict()
        
        for i in range(1, order+1):
            exec('globals()["f' + str(i) + '"] = sym.symbols("f'+ str(i) +'")')
            exec('globals()["lag' + str(i) + '"] = sym.symbols("lag'+ str(i) +'")')
            if i == 1:
                fvector[i] = 'globals()["f' + str(i) + '"]'
                lvector[i] = 'globals()["l' + str(i) + '"]'
            else:
                fvector[i] = fvector[i-1] + ',globals()["f' + str(i) + '"]'
                lvector[i] = lvector[i-1] + ',globals()["l' + str(i) + '"]'
             
        # compute the GFRFs for n = 1
        if order == 1:
                   
            if not noiseFlag:
                globals()['numH'] = 0*globals()['f1']
            else:
                globals()['numH'] = 0*globals()['f1'] + 1
            
            globals()['denH'] = 0*globals()['f1'] + 1
            
            for globals()['l1'] in range(1, maxLag+1):
                if 'c_01' in locals():
                    if len(c_01) >= globals()['l1']:
                        exec('globals()["numH"] = globals()["numH"] + c_01[' + lvector[order] + '] * sym.exp(-sym.I*globals()["l1"]*2*sym.pi*globals()["f1"]/fs)')
                   
                if 'c_01' in locals():
                    if len(c_10) >= globals()['l1']:                        
                        exec('globals()["denH"] = globals()["denH"] - c_10['+lvector[order]+']*sym.exp(-sym.I*l1*2*sym.pi*globals()["f1"]/fs)')
            
            
            if noiseFlag:
                globals()['numH'] = 0*globals()['f1'] + 1
            
            H1 = globals()['numH']/globals()['denH']
            
            H1_1 = H1 * sym.exp(-sym.I*(2*sym.pi*globals()['f1']/fs)*globals()['lag1'])
            
            globals()['Hn'][1] = H1
            globals()['Hnn'][(1,1)] = H1_1
        else:
            # recursive call
            
            globals()['Hnn'], globals()['Hn'] = buildHn(order-1, C, fs, maxLag, noiseFlag) #% recursive call
            
            # Compute the Hn,p with n=order and p=2,..., n
            
            for p in range(2, n+1):
                print('n=', n, ', p = ', p)
                for j in range(1, p+1):
                    if j == 1:
                        lagsPvector = 'globals()["lag' + str(j) + '"]'
                    else:
                        lagsPvector = lagsPvector + ',globals()["lag' + str(j) + '"]'
                              
                globals()['Hnn'][(n,p)] = 0*globals()['f1']
                for i in range(1, n-p+2):
                    globals()['freqSum'] = 0*globals()['f1']
                    freqNivector = []
                    for j in range(1, i+1):
                        exec('globals()["freqSum"] = globals()["freqSum"] + 2*sym.pi*globals()["f' + str(j) + '"]/ fs')
                    
                    for j in range(1, p):
                        if j == 1:
                            lagsPvector = 'globals()["lag' + str(j) + '"]'
                        else:
                            lagsPvector = lagsPvector + ',globals()["lag' + str(j) + '"]'
                                       
                    for j in range(i+1, n+1):
                        if j == i+1:
                            freqNivector = '.subs([(globals()["f' + str(j-i) + '"], globals()["f' + str(j) + '"])'
                        else:
                            freqNivector = freqNivector + ',(globals()["f' + str(j-i) + '"], globals()["f' + str(j) + '"])'
                    freqNivector = freqNivector + '])'
                    
                    
                    exec('globals()["Hnn"][(n,p)] = globals()["Hnn"][(n,p)] + globals()["Hn"][i]*globals()["Hnn"][(n-i,p-1)]' +freqNivector +' *sym.exp(-sym.I*globals()["freqSum"]*globals()["lag' + str(p) + '"])')
                
            globals()['inputComponent'] = 0*globals()['f1'] # Component of the GFRF related to the input signal
            globals()['mixedComponent'] = 0*globals()['f1'] # Component of the GFRF related to the terms with input and output signals multiplied
            globals()['outputComponent'] = 0*globals()['f1'] # Component of the GFRF related to the output signal
            globals()['denH'] = 0*globals()['f1'] + 1 # Denominator of the GFRF
            globals()['numH'] = 0*globals()['f1'] # Numerator of the GFRF
                           
            # computation of the input component
            if 'c_0' + str(n) in locals():
                exec('globals()["lagsPosition"] = np.ravel_multi_index(np.where(c_0' + str(n) + ' != 0.0), c_0' + str(n) + '.shape)')
                exec('(' + lvector[n] + ',) = np.unravel_index(globals()["lagsPosition"], c_0' + str(n) + '.shape)')
                for k in range(len(globals()['lagsPosition'])):
                        globals()['freqVector'] = sym.Matrix([])
                        globals()['lagsVector'] = sym.Matrix([])
                        for j in range(1, n+1):
                                exec('globals()["freqVector"] = globals()["freqVector"].col_insert(len(globals()["freqVector"]), sym.Matrix([2*sym.pi*f' + str(j) + '/fs]))')
                                exec('globals()["lagsVector"] = globals()["lagsVector"].row_insert(len(globals()["lagsVector"]), sym.Matrix([globals()["l' + str(j) + '"][' + str(k) +']]))')
                        
                        for j in range(1, n+1):
                                if j == 1:
                                    lNvector = 'globals()["l'+ str(j) + '"][k]'
                                else:
                                    lNvector = lNvector + ',globals()["l' + str(j) + '"][k]'
                        globals()['expFreqLag'] = sym.exp(-sym.I*((globals()["freqVector"]*globals()["lagsVector"])[0]))
                        exec('globals()["inputComponent"] = globals()["inputComponent"] + c_0' + str(n) + '[' + lNvector + ']*globals()["expFreqLag"]')
                    
            # computation of the mixed component
            for q in range(1, n):
                    for p in range(1, n-q+1):
                            if 'c_' + str(p) + str(q) in locals():
                                    exec('globals()["lagsPosition"] = np.ravel_multi_index(np.where(c_' + str(p) + str(q) + ' != 0), c_' + str(p) + str(q) + '.shape)')
                                    exec(lvector[p+q] + ' = np.unravel_index(globals()["lagsPosition"], c_' + str(p) + str(q) + '.shape)')
                                                                     
                                    for k in range(len(globals()['lagsPosition'])):
                                        globals()['freqVector'] = sym.Matrix([])
                                        for j in range(n-q+1, n+1):
                                            exec('globals()["freqVector"] = globals()["freqVector"].col_insert(len(globals()["freqVector"]), sym.Matrix([2*sym.pi*globals()["f' + str(j) + '"]/fs]))')
                                        
                                        globals()['lagsVector'] = sym.Matrix([])
                                        for j in range(p+1, p+q+1):
                                            exec('globals()["lagsVector"] = globals()["lagsVector"].row_insert(len(globals()["lagsVector"]), sym.Matrix([globals()["l' + str(j) + '"][' + str(k) + ']]))')
                                        
                                        globals()['expFreqLag'] = sym.exp(-sym.I*((globals()['freqVector']*globals()['lagsVector'])[0]))
                                        for j in range(1, p+1):
                                            if j == 1:
                                                lPvector = '.subs([(globals()["lag' + str(j) + '"], globals()["l' + str(j) + '"][k])'
                                            else:
                                                lPvector = lPvector + ',(globals()["lag' + str(j) + '"], globals()["l' + str(j) + '"][k])'
                                        lPvector = lPvector + '])'        
                                        
                                        for j in range(1, p+q+1):
                                            if j == 1:
                                                lNvector = 'globals()["l'+ str(j) + '"][k]'
                                            else:
                                                lNvector = lNvector + ',globals()["l' + str(j) + '"][k]'
                                        
                                        
                                        exec('globals()["mixedComponent"] = globals()["mixedComponent"] + c_' + str(p) + str(q) + '[' + lNvector + ']*globals()["expFreqLag"]*globals()["Hnn"][(n-q,p)]'+lPvector)
                                    
                            
                    
            
            # computation of the output component
            for p in range(2, n+1):
                    if 'c_'+ str(p) + '0'  in locals():
                            exec('globals()["lagsPosition"] = np.ravel_multi_index(np.where(c_' + str(p) + '0 != 0), c_' + str(p) + '0.shape)')
                            exec(lvector[p] + ' = np.unravel_index(globals()["lagsPosition"], c_' + str(p) + '0.shape)')
                            
                            globals()['freqVector'] = sym.Matrix([])
                            globals()['lagsVector'] = sym.Matrix([])
                            
                            for k in range(len(globals()['lagsPosition'])):
                                for j in range(1, p+1):
                                    if j == 1:
                                        lNvector = 'globals()["l' + str(j) + '"][k]'
                                    else:
                                        lNvector = lNvector + ', globals()["l' + str(j) + '"][k]'
                                
                                for j in range(1, p+1):
                                    if j == 1:
                                        lPvector = '.subs([(globals()["lag' + str(j) + '"], globals()["l' + str(j) + '"][k])'
                                    else:
                                        lPvector = lPvector + ',(globals()["lag' + str(j) + '"], globals()["l' + str(j) + '"][k])'
                                lPvector = lPvector + '])'   
                                    
                                print(n)
                                exec('print(locals()["c_'+ str(p) + '0"])')
                                exec('print(globals()["Hnn"])')
                                exec('globals()["outputComponent"] = globals()["outputComponent"] + c_' + str(p) + '0[' + lNvector + ']*globals()["Hnn"][(n,p)]'+lPvector)
                          
            # computation of the denominator of the GFRF
            if 'c_10' in locals():
                globals()['lagsPosition'] = np.ravel_multi_index(np.where(c_10 != 0), c_10.shape)
                for k in range(len(globals()['lagsPosition'])):
                    globals()['freqSum'] = 0*globals()["f1"]
                    for j in range(1, n+1):
                        exec('globals()["freqSum"] = globals()["freqSum"] + 2*sym.pi*globals()["f' + str(j) + '"]/ fs')
                    
                    globals()['denH'] = globals()['denH'] - c_10[globals()['lagsPosition'][k]]*sym.exp(-sym.I*globals()['freqSum']*globals()['lagsPosition'][k])
           
            #  Computation of Hn for n = order
            
            globals()['numH'] = globals()['inputComponent'] + globals()['mixedComponent'] + globals()['outputComponent']
            
            globals()['Hn'][n] = globals()['numH']/globals()['denH']
    
            # Computation of Hn,1
                
            globals()['freqSum'] = 0
            for j in  range(1, n+1):
                exec('globals()["freqSum"] = globals()["freqSum"] + 2*sym.pi*globals()["f' + str(j) + '"]/ fs')
            
            exec('globals()["Hnn"][(' + str(n) + ',1)] = globals()["Hn"][n]*sym.exp(-sym.I*globals()["freqSum"]*lag1)')    
    
        return globals()['Hnn'], globals()['Hn']



def computeSignalFFT(signal, Fs, fres):
        '''
        % Compute the FFT from the signal with the appropriate frequency resolution, and normalize by the length of the signal.
        % 
        % written by: Renato Naville Watanabe 
        %
        % [S, f] = computeSignalFFT(signal, Fs, fres)
        %	
        % Inputs:
        %	
        %   signal: vector of floats, vector with signal to have the FFT computed.
        %
        %   Fs: float, sampling frequency, in Hz.
        %
        %   fres: float, the wanted frequency resolution, in Hz.
        %
        %
        % Outputs:
        %   
        %   S: vector of complex, the signal FFT.
        %
        %   f: vector of floats, vector of frequencies.
        '''
        from scipy.fftpack import fft, fftshift
        from scipy.signal import periodogram, welch
        
        f = np.arange(-Fs/2, Fs/2, fres)
#        f, F = periodogram(signal, fs=Fs, nfft=len(f), return_onesided=False, 
#                           axis=0, detrend=False)
#        f, F = welch(signal, fs=Fs, nfft=len(f), return_onesided=False, 
#                     axis=0, detrend=False, nperseg=150, 
#                     noverlap=75)
        F = fft(signal, len(f), axis=0)/len(f)
#        F = np.mean(F, axis=1, keepdims=True)
        S = fftshift(F, axes=0)
#        f = fftshift(f)
        
        return S, f
    


def determineFrequencies(f, fres, n, f_inputMin, f_inputMax):
        '''
        % Determine the combination of n frequencies that sum up f. Used in the NOFRF computation
        %
        % written by: Renato Naville Watanabe 
        %
        % fVector = determineFrequencies(f, fres, n, f_inputMin, f_inputMax)
        %	
        %
        % Inputs:
        %   
        %   f: float, the frequency, in Hz, that you wish to find the combinations of
        %   frequencies that sum to f. Example: for f = 2 and n = 2, you will have:
        %   f1=4 and f2 =-2, f1=0 and f2 = 2, and so on...
        %
        %   fres: float, the frequency resolution that the search of combinations will use.
        %
        %   n: integer, the number of frequencies to make the combinations.
        %
        %   f_inputMin: vector of floats, lower frequency limit of the input signal, in Hz.
        %   You can define one value for each degree or simply one value for all
        %   degrees. For example: f_inputMin = [19;19;0;0;19;0] if you will use
        %   GFRFs up to degree six.
        %
        %   f_inputMax: vector of floats, upper frequency limit of the input signal, in Hz.
        %   You can define one value for each degree or simply one value for all
        %   degrees. For example: f_inputMax = [21;21;2;2;21;2] if you will use
        %   GFRFs up to degree six.
        %
        %
        % Output:
        %
        %   fVector:  cell, contains n vectors with the found frequency
        %   combinations. It eliminates the frequency combinations that contains frequencies above the Nyquist frequency (Fs/2).
        '''
        
        fVectorTemp = dict()
        fVector = dict()
        
        if len(f_inputMin) == 1:
           f_inputMin = f_inputMin * np.ones(n)
           f_inputMax = f_inputMax * np.ones(n)
        
        for i in range(1, n):
            fVectorTemp[i] = np.hstack((np.arange(-f_inputMax[i], -f_inputMin[i], fres), 
                                        np.arange(f_inputMin[i], f_inputMax[i], fres)))
            if i == 1:
                fCoordVector = 'fVector[' + str(i) + ']'
                fVectorTempCall = 'fVectorTemp[' + str(i) + ']'
                fVectorDefinitionString = 'len(fVectorTemp[' + str(i) + '])' 
            else:
                fCoordVector = fCoordVector + ', fVector[' + str(i) + ']'
                fVectorTempCall = fVectorTempCall + ', fVectorTemp[' + str(i) + ']'
                fVectorDefinitionString = fVectorDefinitionString + ', len(fVectorTemp[' + str(i) + '])'
            
        
        fCoordVector = fCoordVector + ', fVector[' + str(n) + ']'
        fVectorTempCall = fVectorTempCall + ', f'
        
        for i in range(1, n):
            exec('fVector[i] = np.zeros((' + fVectorDefinitionString + '))')        
        
        exec(fCoordVector + ' = np.meshgrid(' + fVectorTempCall + ', indexing="ij")')
        
        del fVectorTemp        
        
        for j in range(1, n):             
             fVector[n] = fVector[n] - fVector[j]
        
        validFrequenciesIndex = np.logical_and(np.abs(fVector[n]) <= f_inputMax[n-1], 
                                               np.abs(fVector[n]) >= f_inputMin[n-1])
        
        for j in range(1, n+1):
            fVector[j] = np.reshape(fVector[j][validFrequenciesIndex], -1)
        
        return fVector
    
def ismember(a, b):
        '''
        Inputs:
            a: vector of floats
               
            
            b: vector of floats
               
            
        Outputs:
            lia: vector of booleans
                 Array of the same size as a containing True where 
               the elements of a are in B and False otherwise.
            
            locb: vector of int        
                  array containing the lowest absolute index in B for each element
                  in A which is a member of B and 0 if there is no such index.
        '''
        lia = np.isin(a, b)
        
        indexes = np.arange(0, len(a), 1)[lia]
        locb = np.zeros_like(a)
        
        for j in range(len(b)):
                locb[indexes[a[indexes] == b[j]]] = j
        
        return lia, locb
    
def inputFFTDegree(X, freqIndex, degree):
        '''    
        % Computes the FFT of the specified degree at the specified frequency combinations.
        %
        %written by: Renato Naville Watanabe 
        %
        %	degreeFFT = inputFFTDegree(X, freqIndex, degree)
        %	
        % Inputs:
        %
        %   X: vector of complex, FFT of the signal.
        %
        %   freqIndex: cell, contains the frequency vectors in which the FFT of the specified degree must be
        %   computed.
        %
        %   degree: integer, degree of the FFT you wish to compute.
        %
        %
        % Outputs:
        %
        %   degreeFFT: vector of complex, the FFT of the specified degree.
        '''
        
        
        if degree > 1:
           degreeFFT = inputFFTDegree(X, freqIndex, degree-1)*X[freqIndex[degree],:]
        else:
           degreeFFT = X[freqIndex[1],:]        
        
        return degreeFFT


def computeDegreeNOFRF(HnFunction, U, Fs, degree, f, fres, f_inputMin, f_inputMax):
        '''
        % Computes the NOFRF of the specified degree in the frequency f.
        %
        % written by: Renato Naville Watanabe 
        %
        % DegreeNOFRF = computeDegreeNOFRF(HnFunction, U, Fs, degree, f, fres, f_inputMin, f_inputMax)
        %	
        %   
        % Inputs:
        %
        %   HnFunction: function, is the function of the the GFRFs of the specified degree.
        %
        %   U: vector of complex, the FFT of the input signal obtained with the computeSignalFFT function.
        %
        %   Fs: float, the sampling frequency, in Hz.
        %
        %   degree: integer,  degree of the NOFRF to be computed.
        % 
        %   f: float, the frequency, in Hz, to have the NOFRF computed.
        %	
        %   fres: float, the frequency resolution of the FFT, in Hz.
        %
        %   f_inputMin: vector of floats, lower frequency limit of the input signal, in Hz.
        %   You can define one value for each degree or simply one value for all
        %   degrees. For example: f_inputMin = [19;19;0;0;19;0] if you will use
        %   GFRFs up to degree six.
        %
        %   f_inputMax: vector of floats, upper frequency limit of the input signal, in Hz.
        %   You can define one value for each degree or simply one value for all
        %   degrees. For example: f_inputMax = [21;21;2;2;21;2] if you will use
        %   GFRFs up to degree six.
        %
        %
        % Output:
        %   
        %   DegreeOFRF: vector of complex,  the NOFRF relative to the specified degree.
        '''
        
        fv = np.arange(-Fs/2, Fs/2, fres)
        
        for i in range(1, degree+1):
            if i == 1:
                fnCall = 'fVector[' + str(i) + ']'
            else:
                fnCall = fnCall + ', fVector[' + str(i) + ']'
                      
        fVector = determineFrequencies(f, fres, degree, f_inputMin, f_inputMax)
        
        freqIndex = dict()
        for i in range(1, degree+1):
            A1, freqIndex[i] = ismember(np.round(fVector[i]/fres).astype(int),
                                        np.round(fv/fres).astype(int))
           
#            if not np.all(A1): 
#                    print(A1[np.logical_not(A1)])
#                    print(freqIndex[i][np.logical_not(A1)])
#                    print(fVector[i][np.logical_not(A1)])
          
        Un = inputFFTDegree(U, freqIndex, degree)
        
        exec('globals()["DegreeNOFRF"] = HnFunction(' + fnCall + ')')
        globals()["DegreeNOFRF"] = np.reshape(globals()["DegreeNOFRF"], (-1, 1))
        globals()['DegreeNOFRF'] = np.sum(globals()['DegreeNOFRF'] * Un, axis=0, keepdims=True)*(fres**(degree-1))/np.sqrt(degree)
        
        return globals()['DegreeNOFRF']

def computeNOFRF(Hn, U, minDegree, maxDegree, Fs, fres, fmin, fmax, f_inputMin, f_inputMax):
        '''
        % Computes the NOFRF (Nonlinear Output Frequency Response Function) for the specified input.
        %
        % written by: Renato Naville Watanabe 
        %
        % [NOFRF, f] = computeNOFRF(Hn, U, minDegree, maxDegree, Fs, fres, fmin, fmax, f_inputMin, f_inputMax)
        %	
        %   
        % Inputs:
        %   
        %   Hn: cell, contains all the GFRFs until the specified degree.
        %
        %   U: vector of complex, the FFT of the input signal obtained with the computeSignalFFT function.
        %
        %   minDegree: integer, the minimal degree to have the NOFRF computed.
        %
        %   maxDegree: integer, the maximal degree to have the NOFRF computed.
        %
        %   Fs: float, sampling frequency, in Hz.
        %
        %   fres: float, frequency resolution of the FFT, in Hz.
        %
        %   fmin: float, lower frequency limit of the NOFRF computation, in Hz.
        %
        %   fmax: float, upper frequency limit of the NOFRF computation, in Hz.
        %
        %   f_inputMin: vector of floats, lower frequency limit of the input signal, in Hz.
        %   You can define one value for each degree or simply one value for all
        %   degrees. For example: f_inputMin = [19;19;0;0;19;0] if you will use
        %   GFRFs up to degree six.
        %
        %   f_inputMax: vector of floats, upper frequency limit of the input signal, in Hz.
        %   You can define one value for each degree or simply one value for all
        %   degrees. For example: f_inputMax = [21;21;2;2;21;2] if you will use
        %   GFRFs up to degree six.
        %
        %
        % Outputs:
        %
        %   NOFRF: vector of complex, the NOFRF of the system for the given input at
        %   each frequency.
        %
        %   f: vector of floats, the vector of frequencies.
        '''
        import sympy as sym
   
        fv = np.arange(-Fs/2, Fs/2, fres)
        
        f_out = np.arange(fmin, fmax, fres)
        
        if type(f_inputMin) == float or type(f_inputMin) == int:
                f_inputMin = f_inputMin*np.ones(maxDegree)
                f_inputMax = f_inputMax*np.ones(maxDegree)
        
        NOFRF = np.zeros((len(f_out), U.shape[1]), dtype=np.complex128)
        f = dict()
        for i in range(minDegree, maxDegree+1):
                if  len(Hn) >= i and Hn[i] != 0:
                    freqList = list()
                    for k in range(1, i+1):
                        f[k] = sym.symbols('f'+str(k))
                        freqList.append(f[k])
                    freqTuple = tuple(freqList)
                        
                    HnFunction = sym.lambdify(freqTuple, Hn[i], 'numpy')
                    
                    for j in range(len(f_out)):
                        if i == 1  and f_inputMin[0]<=f_out[j]<=f_inputMax[0]:
                            validFrequencyIndices = np.abs(fv-f_out[j])<=fres/2
                            if len(U[validFrequencyIndices,:])>0:
                                NOFRF[j,:] = 2*np.mean(HnFunction(np.linspace(f_out[j]-fres/2, f_out[j]+fres/2, 10000))).reshape(1,1)*U[validFrequencyIndices,:]
#                                NOFRF[j,0] = 2*HnFunction(f_out[j])*U[validFrequencyIndices]
                        else:                            
                            NOFRF[j,:] = NOFRF[j,:] + 2*computeDegreeNOFRF(HnFunction, U, Fs, i, f_out[j], fres, f_inputMin, f_inputMax)
 
        f = f_out

        return NOFRF, f
    
def findNoiseModel(beta, D, ustring='u', ystring='y', nstring='n'):
        Dnoise = np.hstack((nstring+'(i-0)', D))
        betanoise = np.vstack((1.0, beta))
        indToRemove = np.reshape(np.array([], dtype=np.int), (-1, 1))
        for j in range(len(Dnoise)):
            if Dnoise[j].find(ustring) != -1:
                indToRemove = np.vstack((indToRemove, j))
        Dnoise = np.delete(Dnoise, indToRemove)
        betanoise = np.delete(betanoise, indToRemove)
        betanoise = np.reshape(betanoise, (-1, 1))
            
        return betanoise, Dnoise

def computeIuy(u, y, Fs, fres, beta_uy, beta_yu, Duy, Dyu,
                                 Hnuy, Hnyu, Hnnuy, Hnnyu, f_inputMin, f_inputMax, maxOrder,
                                 ustring='u', ystring='y'):
        
        import identFunctions as ident       
        
        U, f = computeSignalFFT(u, Fs, fres)       
#        Y, _ = computeSignalFFT(y, Fs, fres)
        
        
        Duy, beta = ident.findStructWithMaxDegree(Duy, beta_uy, maxOrder)
        maxLag_uy = ident.findMaxLagFromStruct(Duy)
        
        n_uy = np.zeros((u.shape[0] - maxLag_uy, u.shape[1]))
        for i in range(u.shape[1]):
            _, n_uy[:,[i]], _ = ident.osaWithStruct(u[:, [i]], y[:, [i]], beta_uy, Duy, degree=maxOrder, 
                                                    ustring=ustring, ystring=ystring)
    
        Nuy, _ = computeSignalFFT(n_uy, Fs, fres)
#        Nyu, _ = computeSignalFFT(n_yu, Fs, fres)
        
        fmin = 0
        fmax = Fs/2
        minDegree = 1
        
        NOFRFuy, _ = computeNOFRF(Hnuy, 2*U, minDegree, maxOrder, Fs, fres,
                                  fmin, fmax, f_inputMin, f_inputMax)
        
#        NOFRFyu, _ = computeNOFRF(Hnuy, 2*Y, minDegree, maxOrder, Fs, fres,
#                                  fmin, fmax, f_inputMin, f_inputMax)
#               
        Hye, _ = computeNOFRF(Hnnuy, 2*Nuy, minDegree, maxOrder, Fs, fres,
                              fmin, fmax, f_inputMin, f_inputMax)
        
#        Hue, _ = computeNOFRF(Hnnyu, 2*Nyu, minDegree, maxOrder, Fs, fres,
#                              fmin, fmax, f_inputMin, f_inputMax)        
##        
        
        
        freq = f >= -fres/2
        f = f[freq]
       
                
        Iuy = np.log((np.abs(Hye)**2 + np.abs(NOFRFuy)**2)/(np.abs(Hye+1e-7)**2))
        Iyy = np.log((np.abs(Hye)**2 + np.abs(NOFRFuy)**2)/(np.abs(NOFRFuy+1e-7)**2))
        
        return Iuy, f, Iyy
    



    


def NPDC(u, y, Fs, fres, beta_uy, beta_yu, Duy, Dyu,
         Hnuy, Hnyu, f_inputMin, f_inputMax, maxOrder, 
         L=9, ustring='u', ystring='y', 
         mfrolsEngine='python', elsEngine='python'):
    
        import identFunctions as ident
        import matplotlib.pyplot as plt
        
        f = np.arange(-Fs/2, Fs/2, fres)
        
        beta_nuy, Dnuy = findNoiseModel(beta_uy, Duy, ustring=ustring,
                                        ystring=ystring, nstring='e')
        
        Hnnuy = computeSystemGFRF(Duy, Fs, beta_uy, maxOrder,
                                  ustring=ustring, ystring=ystring,
                                  noiseFlag=True)
        
        beta_nyu, Dnyu = findNoiseModel(beta_yu, Dyu, ustring=ystring,
                                        ystring=ustring, nstring='f')
        
        Hnnyu = computeSystemGFRF(Dyu, Fs, beta_yu, maxOrder,
                                  ustring=ystring, ystring=ustring,
                                  noiseFlag=True)
        
        Nsegment = len(u)//(L//2 + 1)
        
        uDivided = np.zeros((Nsegment, L*u.shape[1]))
        yDivided = np.zeros((Nsegment, L*u.shape[1]))
         
        for j in range(u.shape[1]):
            for i in range(L):
                uDivided[:, [j*L+i]] = u[i//2*Nsegment:(i+2)//2*Nsegment, [j]]
                yDivided[:, [j*L+i]] = y[i//2*Nsegment:(i+2)//2*Nsegment, [j]]
            
            
        Iuyn, f, Iyyn = computeIuy(uDivided, yDivided,
                                   Fs, fres, beta_uy, beta_yu, Duy, Dyu,
                                   Hnuy, Hnyu, Hnnuy, Hnnyu, f_inputMin, 
                                   f_inputMax, maxOrder=maxOrder,
                                   ustring=ustring, ystring=ystring)
        
        Iyun, f, Iuun = computeIuy(yDivided, uDivided, 
                                   Fs, fres, beta_yu, beta_uy, 
                                   Dyu, Duy, Hnyu, Hnuy, Hnnyu, Hnnuy,
                                   f_inputMin, f_inputMax, maxOrder=maxOrder, 
                                   ustring=ystring, ystring=ustring)
        
        IuynLinear, f, _ = computeIuy(uDivided, yDivided, 
                                      Fs, fres, beta_uy, beta_yu, Duy, Dyu, 
                                      Hnuy, Hnyu, Hnnuy, Hnnyu, f_inputMin, 
                                      f_inputMax, maxOrder=1, 
                                      ustring=ustring, ystring=ystring)
        
        IyunLinear, f, _ = computeIuy(yDivided, uDivided, 
                                      Fs, fres, beta_yu, beta_uy, 
                                      Dyu, Duy, Hnyu, Hnuy, Hnnyu, Hnnuy,
                                      f_inputMin, f_inputMax, maxOrder=1, 
                                      ustring=ystring, ystring=ustring)
                
        Iuy = np.mean(Iuyn, axis=1)
        Iyu = np.mean(Iyun, axis=1)
        Iuu = np.mean(Iuun, axis=1)
        Iyy = np.mean(Iyyn, axis=1)
        IuyLinear = np.mean(IuynLinear, axis=1)
        IyuLinear = np.mean(IyunLinear, axis=1)
        
        # NPDCuy = Iuy/np.sqrt(np.abs(Iuy)**2+np.abs(Iyu)**2+np.abs(Iuu)**2+np.abs(Iyy)**2)
        # NPDCyu = Iyu/np.sqrt(np.abs(Iuy)**2+np.abs(Iyu)**2+np.abs(Iuu)**2+np.abs(Iyy)**2)
        # NPDCuyn = Iuyn/np.sqrt(np.abs(Iuyn)**2+np.abs(Iyun)**2+np.abs(Iuun)**2+np.abs(Iyyn)**2)
        # NPDCyun = Iyun/np.sqrt(np.abs(Iuyn)**2+np.abs(Iyun)**2+np.abs(Iuun)**2+np.abs(Iyyn)**2)
        
        # NPDCuyLinear = IuyLinear/np.sqrt(np.abs(Iuy)**2+np.abs(Iyu)**2+np.abs(Iuu)**2+np.abs(Iyy)**2)
        # NPDCyuLinear = IyuLinear/np.sqrt(np.abs(Iuy)**2+np.abs(Iyu)**2+np.abs(Iuu)**2+np.abs(Iyy)**2)
        # NPDCuynLinear = IuynLinear/np.sqrt(np.abs(Iuyn)**2+np.abs(Iyun)**2+np.abs(Iuun)**2+np.abs(Iyyn)**2)
        # NPDCyunLinear = IyunLinear/np.sqrt(np.abs(Iuyn)**2+np.abs(Iyun)**2+np.abs(Iuun)**2+np.abs(Iyyn)**2)
        
        
        
        idu = np.random.rand(*uDivided.shape).argsort(0)
        uDividedShuffle = uDivided[idu, np.arange(uDivided.shape[1])]
        
        idy = np.random.rand(*yDivided.shape).argsort(0)
        yDividedShuffle = yDivided[idy, np.arange(yDivided.shape[1])]
        

        
        beta_uyShuffle, nyShuffle, DuyShuffle = ident.identifyModel(uDividedShuffle, yDividedShuffle, 0, 0, ustring=ustring,
                                                                    ystring=ystring, nstring='n', L=1,
                                                                    supress=True, method='mols', elsMethod='RLS', 
                                                                    elsMaxIter=2, useStruct=Duy, mfrolsEngine=mfrolsEngine,
                                                                    elsEngine=elsEngine)
        
        beta_yuShuffle, nuShuffle, DyuShuffle = ident.identifyModel(yDividedShuffle, uDividedShuffle, 0, 0, 
                                                                    ustring=ystring, ystring=ustring, nstring='m', L=1, 
                                                                    supress=True, method='mols', elsMethod='RLS', 
                                                                    elsMaxIter=2, useStruct=Dyu)
        
        HnuyShuffle = computeSystemGFRF(DuyShuffle, Fs, beta_uyShuffle, maxOrder, ustring=ustring, ystring=ystring)
        HnyuShuffle = computeSystemGFRF(DyuShuffle, Fs, beta_yuShuffle, maxOrder, ustring=ystring, ystring=ustring)
        
        beta_nuyShuffle, DnuyShuffle = findNoiseModel(beta_uyShuffle, DuyShuffle, ustring=ustring,
                                                      ystring=ystring, nstring='e')
        
        HnnuyShuffle = computeSystemGFRF(DuyShuffle, Fs, beta_uyShuffle, maxOrder,
                                         ustring=ustring, ystring=ystring,
                                         noiseFlag=True)
        
        beta_nyuShuffle, DnyuShuffle = findNoiseModel(beta_yuShuffle, DyuShuffle, ustring=ystring,
                                                      ystring=ustring, nstring='f')
        
        HnnyuShuffle = computeSystemGFRF(DyuShuffle, Fs, beta_yuShuffle, maxOrder,
                                         ustring=ystring, ystring=ustring,
                                         noiseFlag=True)
        
        Nsegment = len(u)//(L//2 + 1)
        
        IuynConf, f, IyynConf = computeIuy(uDividedShuffle, yDividedShuffle,
                                           Fs, fres, beta_uyShuffle, beta_yuShuffle, DuyShuffle, DyuShuffle,
                                           HnuyShuffle, HnyuShuffle, HnnuyShuffle, HnnyuShuffle, f_inputMin, 
                                           f_inputMax, maxOrder=maxOrder,
                                           ustring=ustring, ystring=ystring)
        
        IyunConf, f, IuunConf = computeIuy(yDividedShuffle, uDividedShuffle, 
                                           Fs, fres, beta_yuShuffle, beta_uyShuffle, 
                                           DyuShuffle, DuyShuffle, HnyuShuffle, HnuyShuffle, 
                                           HnnyuShuffle, HnnuyShuffle,
                                           f_inputMin, f_inputMax, maxOrder=maxOrder, 
                                           ustring=ystring, ystring=ustring)
        
        
        # plt.figure()
        # plt.hist(np.max(IuynConf, axis=0), 10)
        # plt.show()
        
        # plt.figure()
        # plt.hist(np.max(IyunConf, axis=0), 10)
        # plt.show()
        
        # IuyConf = np.percentile(np.abs(IuynConf), 50)
        # IyuConf = np.percentile(np.abs(IyunConf), 50)
        IuyConf = np.abs(IuynConf.mean(axis=1))
        IyuConf = np.abs(IyunConf.mean(axis=1))
        
        Iuy = np.mean(Iuyn, axis=1)
        Iyu = np.mean(Iyun, axis=1)
        Iuu = np.mean(Iuun, axis=1)
        Iyy = np.mean(Iyyn, axis=1)
        IuyLinear = np.mean(IuynLinear, axis=1)
        IyuLinear = np.mean(IyunLinear, axis=1)
        
        
        return Iuy, Iyu, f, Iuyn, Iyun, IuyLinear, IyuLinear, IuynLinear, IyunLinear, IuyConf, IyuConf


