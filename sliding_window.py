# -*- coding: utf-8 -*-
import decimal


import numpy as np
import pandas as pd

from scipy.fft import fft
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew, mode


# Raise exception for warnings
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from log import log
else:
    # uses current package visibility
    from .log import log

def round_half_up(x):
    ''' round half up instead of to nearest even integer '''
    return int(decimal.Decimal(x).to_integral_value(rounding=decimal.ROUND_HALF_UP))

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

def FFT_peaks(data,*args,**kwargs):
    ''' Highest FFT Value and Frequency + Ratio between highest and second highest peak '''
    locs,props = find_peaks(data,height=(None, None))
    pks = props['peak_heights']
    
    idx = pks.argsort()[::-1]
    pks,locs = pks[idx],locs[idx]

    if pks.shape[0] == 0:
        pks,locs = data,np.array([0])
    
    elif pks.shape[0] == 1:
        pks = np.concatenate((pks,[np.min(data)]))
    
    finterval = 50/(data.shape[0]-1)
    return np.array([pks[0],locs[0]*finterval,pks[0]/(pks[1]+np.finfo(float).eps)])

def SlideEnergy(x,x2,winlen,skiplen,fs2,finterval):
    nwin = np.fix((fs2-(winlen-skiplen))/skiplen).astype(int)
    y = np.zeros(nwin)
    for n in range(nwin):
        idx_start = round_half_up( (n)*skiplen/finterval )
        idx_end = min((round_half_up( ((n)*skiplen + winlen)/finterval  ) + 1, x.shape[0]))
        y[n] = np.sum(x2[idx_start:idx_end])
    return np.array(y)


def FFT_subbands(data,*args,**kwargs):
    fs2 = 50
    finterval = fs2/(data.shape[0]-1)
    data2 = np.power(data,2)
    data2_sum = data2.sum()
    
    result = np.empty(846)
    idx = 0
    for winlen in [1,2,3,4,5,10,15,20,25]:
        y = SlideEnergy(data,data2,winlen,.5 if winlen==1 else 1.,fs2,finterval)
        result[idx:idx+y.shape[0]] = y
        result[idx+y.shape[0]:idx+2*y.shape[0]] = y/data2_sum
        idx+=2*y.shape[0]
    return result
    
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
    return sum(num for _,_,num,_ in [FXdict[Fx] for Fx in FX_list])

def get_FX_names(indeces=None):
    names = []
    for name in [*FXdict]:
        if name == 'auto_correlation':
            names += ['auto_correlation_peak','auto_correlation_peak_idx']
        elif name == 'Peak_fft':
            names += ['peak_fft','peak_fft_fq','peak_fft_ratio']
        elif name == 'Subband':
            for bandwith in [1,2,3,4,5,10,15,20,25]:
                stepsize = .5 if bandwith==1 else 1.
                center_fqs = np.arange(start=(bandwith/2),stop=(stepsize+50-bandwith/2),step=stepsize)

                for center_fq in center_fqs:
                    names.append(f'bw_{bandwith}_cfq_{center_fq:.1f}')
                    
                for center_fq in center_fqs:
                    names.append(f'bw_{bandwith}_cfq_{center_fq:.1f}_ratio')
        else:
            names.append(name)
    
    names = np.array(names)
    
    if indeces is None:
        return names
    else:
        return names[indeces]


''' name: (func, params, number of return values, fft) '''
FXdict = {
  "mean": (np.mean,None,1,False),
  "std": (np.std,None,1,False),
  #"zcr": (zcc,None,1,False),
  "mcr": (mcc,None,1,False),
  "energy": (energy,None,1,False),
  "auto_correlation": (autocorr,1,2,False),
  "kurtosis": (kurtosis,None,1,False),
  "skew": (skew,None,1,False),
  
  "mean_fft": (np.mean,None,1,True),
  "std_fft": (np.std,None,1,True),
  "energy_fft": (energy,None,1,True),
  "kurtosis_fft": (kurtosis,None,1,True),
  "skew_fft": (skew,None,1,True),
  "DC_fft": (lambda data,_:data[0],None,1,True),
  "Peak_fft": (FFT_peaks,None,3,True),
  "Subband": (FFT_subbands,None,846,True),
}

QUARTILES = [0,5,10,25,50,75,90,95,100]

# Append quartiles
for q in QUARTILES:
    FXdict["Q%d"%q] = (np.percentile,q,1,False)

# Append quartile ranges
for i in range(len(QUARTILES)-1):
    for j in range(i+1,len(QUARTILES)):
        FXdict["IQR_%d_%d"%(QUARTILES[i], QUARTILES[j])] = (IQR,[QUARTILES[j], QUARTILES[i]],1,False)
    

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
            data_win = data[j,w:w+winsize]
            
            # Calculate FFT
            data_fft = fft(data_win)
            data_fft = abs(data_fft[:data_fft.shape[0]//2+1])
            DC_fft = data_fft[0]
            
            # Silencing 0-0.5Hz
            finterval = 50/data_fft.shape[0]
            idxp5 = np.fix(0.5/finterval).astype(int)+1
            data_fft[:idxp5] = 0 
            
            # Calculate all features
            i=0
            for FX_name in FX_list:
                # Load Feature function
                FX, FXOpts, L, rq_fft = FXdict[FX_name]
                
                # Compute the feature
                if FX_name == 'DC_fft':
                    fx = DC_fft
                elif rq_fft:
                    fx = FX(data_fft,FXOpts)
                else:
                    fx = FX(data_win,FXOpts)

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
    data = np.array([0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,])
    data_fft = fft(data)
    data_fft = np.abs(data_fft[:data_fft.shape[0]//2+1]).round(13)
    fs2 = 50
    finterval = fs2/(data_fft.shape[0]-1)
    idxp5 = np.fix(0.5/finterval).astype(int)+1
    data_fft[:idxp5] = 0 
    
    print("Data:",data)
    i=0
    for name, (FX, FXOpts, num, rq_fft) in FXdict.items():
        i+=num
        if rq_fft:
            fx = FX(data_fft,FXOpts)
        else:
            fx = FX(data,FXOpts)
        print(i,name+':',fx,num)


    from params import Params
    
    P = Params(FX_sel = 'all',winsize=5,jumpsize=5)
    #print(slidingWindow(P,data,label=None))
    
    # print(data_fft)
    #slide = SlideEnergy(data_fft,np.power(data_fft,2),1,0.5,fs2,finterval)
    #slide = SlideEnergy(data_fft,np.power(data_fft,2),25,1,fs2,finterval)
    
    # print(data_fft.shape[0])

    # print(slide)
    # print(slide.shape)
    # print(slide[80-1])

    # print(np.power(data_fft,2)[10:11])
    
    #print(get_FX_names(np.random.choice(a=[False, True], size=(908))).shape)
    
