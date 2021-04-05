# -*- coding: utf-8 -*-
from scipy.stats import kurtosis, skew, mode

import numpy as np
import pandas as pd

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from log import log
else:
    # uses current package visibility
    from .log import log

# Energy
def energy(data,*args,**kwargs):
    return np.dot(data.T,data)

# Zero Crossing Count
def zcc(data,*args,**kwargs):
    return np.nonzero(np.diff(data > 0))[0].shape[0]

# Mean Crossing Count
def mcc(data,*args,**kwargs):
    return np.nonzero(np.diff((data-np.mean(data)) > 0))[0].shape[0]

def val_FX_list(FX_list):
    if not isinstance(FX_list,list):
        return [str(FX_list)]
    else:
        return [x for x in FX_list if x in FXdict]

def get_FX_list(P):
    return val_FX_list(FEATURES[P.get('FX_sel')])

FXdict = {
  "mean": (np.mean,None),
  "std": (np.std,None),
  #"zcr": (zcc,None),
  "mcr": (mcc,None),
  #"auto_correlation": (lambda x:1,None),
  "kurtosis": (kurtosis,None),
  "skew": (skew,None),
  "min": (np.min,None),
  "max": (np.max,None),
  "median": (np.median,None),
}

FEATURES = {
    'basic': ['mean','median','std','mcr','kurtosis','skew'],
    'all': [*FXdict],
    }

def zerol(data,winsize):
    return np.concatenate((np.zeros((data.shape[0],winsize-1)),data),axis=1)

def zeror(data,winsize):
    return np.concatenate((data,np.zeros((data.shape[0],winsize-1))),axis=1)
    
def mirrorl(data,winsize):
    return np.concatenate((data[:winsize-1:0],data),axis=1)
    
def mirrorr(data,winsize):
    return np.concatenate((data,data[:-winsize-1:-1]),axis=1)

def default(data,winsize):
    log("This padding mode is unknown, zerol is applied",error=True)
    return zerol(data,winsize)

def make_numpy(mat):
    if not isinstance(mat, np.ndarray):
        if isinstance(mat,list):
            return np.array(mat)
        elif isinstance(mat, pd.DataFrame):
            return mat.to_numpy()
        else:
            log("Unknown data type: "+str(type(mat)),error=True)
    return mat

def slidingWindow(P,data,label=None):
    winsize = P.get('winsize')
    jumpsize = P.get('jumpsize')
    FX_list = get_FX_list(P)
    padding = P.get('padding')
    
    if not np.isscalar(winsize) or winsize < 1 or int(winsize) != winsize:
        log("slidingWindow: winsize must be integer and larger or equal to 1",error=True)
        return None
    
    if not np.isscalar(jumpsize) or jumpsize < 1 or int(jumpsize) != jumpsize:
        log("slidingWindow: jumpsize must be integer and larger or equal to 1",error=True)
        return None
    
    data = make_numpy(data)
    
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    
    if data.ndim != 2:
        log("slidingWindow: data must be two-dimensional matrix. data.ndim = %d"%data.ndim,error=True)
        return None
    
    colNo = data.shape[0]
    rowNo = data.shape[1]
    
    if colNo == rowNo:
        log("slidingWindow: data must be a matrix with one dimension (along which the window is sliding) longer than the other one",error=True)
        return None
    
    if colNo > rowNo:
        column = True
        data = np.transpose(data)
    else:
        column = False   

    sdata = data.shape[1]                   # size of the time series
    s = np.ceil(sdata/jumpsize).astype(int) # size of output timeline after sliding window
    FXnr = len(FX_list)                     # number of output features
    Cnr = data.shape[0]                     # number of output channels
    
    # Create the output data
    X = np.empty((Cnr,FXnr,s))
    if label is not None:
        label = make_numpy(label)
        Y = np.empty((s))

    ## Pad the data
    # There are several padding modes:
    # In a sliding window process, the first sliding window of size winsize 
    # could reach to element outside (on the left) of the vector. Similarly
    # the last sliding window could reach to elements outside (on the right) of
    # the end of the vector.
    # Several padding strategies are available to ensure the output vector is
    # of same size as the input:
    # 
    # We pad the vector with null at the front to ensure fast loops later.
    switcher = {
        'zerol': zerol,
        'zeror': zeror,
        'mirrorl': mirrorl,
        'mirrorr': mirrorr
    }
    func = switcher.get(padding,default)
    
    data = func(data,winsize)
    
    # Iterate all the windows
    wi=0    # wi index in the output (window index)
    for w in range(0,sdata,jumpsize):
        
        # Iterate all channels
        for j in range(0,Cnr):
            # Extract the windowed data
            datawin = data[j,w:w+winsize]
            
            # Calculate all features
            for i,FX_name in enumerate(FX_list):
                # Load Feature function
                FX, FXOpts = FXdict[FX_name]
                
                # Compute the feature
                fx = FX(datawin,FXOpts)

                # Update the output vector
                X[j,i,wi] = fx
                
        if label is not None:
            Y[wi] = mode(label[w:w+winsize])[0]       
            
        # increase output index
        wi += 1

    if column:
        X = X.transpose()
    
    if label is not None:
        return X.squeeze(), Y
    else:
        return X.squeeze()
    
if __name__ == "__main__":
    data = np.arange(10)+1
    print(data)
    print(zcc(data))
    print(mcc(data))
    print(energy(data))
    FX, FXOpts = FXdict['auto_correlation']
    print(FX(data))