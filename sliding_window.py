# -*- coding: utf-8 -*-
from scipy.stats import kurtosis, skew, mode

import numpy as np
import pandas as pd

# Raise exception for warnings
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from log import log
else:
    # uses current package visibility
    from .log import log

# -------------------
#  Feature Extraction
# -------------------

def zcc(data,*args,**kwargs):
    ''' Zero crossing count '''
    return np.nonzero(np.diff(data > 0))[0].shape[0]

def mcc(data,*args,**kwargs):
    ''' Mean crossing count '''
    return np.nonzero(np.diff((data-np.mean(data)) > 0))[0].shape[0]

def energy(data,*args,**kwargs):
    ''' Energy of a matrix A*A.T '''
    return np.dot(data.T,data)

def autocorr(data, min_delay=10):
    ''' Statistical correlation with a lag of t '''
    result = np.correlate(data, data,mode='full')
    result = result[min_delay+result.shape[0]//2:]
    idx = np.argmax(result)
    return np.array([result[idx], idx])

def IQR(data,q=[75,25]):
    ''' Interquartile range '''
    return np.subtract(*np.percentile(data, q))

# -------------------
#  List of Features
# -------------------

def val_FX_list(FX_list):
    if not isinstance(FX_list,list):
        return [str(FX_list)]
    else:
        return [x for x in FX_list if x in FXdict]

def get_FX_list(P):
    return val_FX_list(FEATURES[P.get('FX_sel')])

def get_FX_list_len(FX_list):
    return sum(num for _,_,num in [FXdict[Fx] for Fx in FX_list])

FXdict = {
  "mean": (np.mean,None,1),
  "std": (np.std,None,1),
  #"zcr": (zcc,None),
  "mcr": (mcc,None,1),
  "energy": (energy,None,1),
  "auto_correlation": (autocorr,1,2),
  "kurtosis": (kurtosis,None,1),
  "skew": (skew,None,1),
}

QUARTILES = [0,5,10,25,50,75,90,95,100]

# Append quartiles
for q in QUARTILES:
    FXdict["Q%d"%q] = (np.percentile,q,1)

for i in range(len(QUARTILES)-1):
    for j in range(i+1,len(QUARTILES)):
        FXdict["IQR_%d_%d"%(QUARTILES[i], QUARTILES[j])] = (IQR,[QUARTILES[j], QUARTILES[i]],1)
    

FEATURES = {
    'basic': ['mean','median','std','mcr','kurtosis','skew'],
    'all': [*FXdict],
    'auto_correlation': ['auto_correlation'],
    }

# -------------------
#  Padding
# -------------------

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
        data = np.transpose(data)

    sdata = data.shape[1]                   # size of the time series
    s = np.ceil(sdata/jumpsize).astype(int) # size of output timeline after sliding window
    FXnr = get_FX_list_len(FX_list)                 # number of output features
    Cnr = data.shape[0]                     # number of output channels
    
    # Create the output data
    X = np.empty((s,Cnr,FXnr))
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
            i=0
            for FX_name in FX_list:
                # Load Feature function
                FX, FXOpts, L = FXdict[FX_name]
                
                # Compute the feature
                fx = FX(datawin,FXOpts)

                # Update the output vector
                X[wi,j,i:i+L] = fx
                
                i+=L
                
        if label is not None:
            Y[wi] = mode(label[w:w+winsize])[0]       
            
        # increase output index
        wi += 1

    
    if label is not None:
        return X.squeeze(), Y
    else:
        return X.squeeze()
    
if __name__ == "__main__":
    data = np.arange(11)+1
    data = np.array([1,1,1,1,1,1,1,1,1,1,2,3,4,5]*10)
    print("Data:",data)
    for name, (FX, FXOpts, num) in FXdict.items():
        print(name+':',FX(data,FXOpts),num)

    data = [1,2,3,2,1,2,3,2,1,2,3,2,1,2]
    result = np.correlate(data, data,mode='full')
    result = result[result.shape[0]//2:]
    print(result)
    
    print(autocorr(data, min_delay=5))

    from params import Params
    
    P = Params(FX_sel = 'auto_correlation',winsize=5,jumpsize=5)
    print(slidingWindow(P,data,label=None))