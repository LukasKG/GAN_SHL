# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from numpy.matlib import repmat
import os

if __package__ is None or __package__ == '':
    from log import log
    from sliding_window import slidingWindow
else:
    from .log import log
    from .sliding_window import slidingWindow


ACC_CHANNELS = ["Acc X","Acc Y","Acc Z"]

NAMES_X = ["Time","Acc X","Acc Y","Acc Z",
           "Gyroscope X","Gyroscope Y","Gyroscope Z",
           "Magnetometer X","Magnetometer Y","Magnetometer Z",
           "Orientation w","Orientation x","Orientation y","Orientation z",
           "Gravity X","Gravity Y","Gravity Z",
           "Linear acceleration X","Linear acceleration Y","Linear acceleration Z",
           "Pressure","Altitude","Temperature"]

NAMES_Y = ["Time","Coarse","Fine","Road","Traffic","Tunnels","Social","Food"]

LABELS_SHL = {
        1: "Still",
        2: "Walking",
        3: "Run",
        4: "Bike",
        5: "Car",
        6: "Bus",
        7: "Train",
        8: "Subway",
        }

# Which processed datasets to store
SAVE_DATA = ['SHL','Short']

PATHS = {
    'SHL': '\\SHL_Dataset_preview_v1\\',
    'Short': None,
    'Test': None,
    'Sincos': None,
    'Hash': '\\SHL_processed\\',
        }

def get_path(P,dataset=None):
    ''' Returns the path to the dataset '''
    if dataset is None:
        dataset = P.get('dataset')
    return P.get('data_path')+PATHS[dataset]

def get_labels():
    ''' Returns list with unique labels '''
    return np.fromiter(LABELS_SHL.keys(), dtype=int)

def remove_nan(data,label):
    ''' Remove rows containing NaN values '''
    idx = pd.isnull(data).any(1).to_numpy().nonzero()[0]
    return data.drop(idx).reset_index(drop=True), label.drop(idx).reset_index(drop=True)

def remove_zero(data,label):
    ''' Remove rows labelled 0 '''
    idx = pd.isnull(label.replace(0, np.nan)).any(1).to_numpy().nonzero()[0]
    return data.drop(idx).reset_index(drop=True), label.drop(idx).reset_index(drop=True)

def reduce_labels(data,label,label_remain):
    ''' Remove all but the selected labels '''
    idx = label['Coarse'].isin(label_remain)
    return data[idx].reset_index(drop=True), label[idx].reset_index(drop=True)

def read_day(P,uid='User1',recid='220617'):
    path = get_path(P) + uid + '/' + recid + '/'
    
    X = pd.read_csv(path+P.get('location')+'_Motion.txt',sep=' ',names=NAMES_X)
    Y = pd.read_csv(path+'Label.txt',sep=' ',names=NAMES_Y)
    
    # Select acceleration channels
    data = X[P.get('channels')]
    #data = X[NAMES_X[1:]]
    
    # Select coarse label
    label = Y[["Coarse"]]
    
    data, label = remove_nan(data, label)
    
    data, label = remove_zero(data, label)
    
    # Select only chosen labels
    if P.get('labels') is not None and set(P.get('labels')) != set(get_labels()):
        data, label = reduce_labels(data,label,P.get('labels'))
    
    return data, label


def read_user(P,uid='User1',noise=None):
    assert uid in ['User1', 'User2', 'User3']
    
    if uid == 'User1':
        recids = ['220617','260617','270617']
    elif uid == 'User2':
        recids = ['140617','140717','180717']
    else:
        recids = ['030717','070717','140617']
        
    for i,recid in enumerate(recids):
        if i==0:
            data, label = read_day(P,uid=uid,recid=recid)
        else:
            tmpD, tmpL = read_day(P,uid=uid,recid=recid)
            data = pd.concat([data,tmpD],axis=0)
            label = pd.concat([label,tmpL],axis=0)
    
    # Apply noise
    if noise is None:
        noise = P.get('noise')
    if noise > 0.0:
        data += np.random.normal(0.0, noise, data.shape)
    
    # Convert acceleration channels into magnitude
    if P.get('magnitude'):
        # Calc magnitude
        data['Magnitude'] = np.sum(data[ACC_CHANNELS].to_numpy()**2,axis=1).reshape(-1,1)**.5
        data = data.drop(ACC_CHANNELS, axis=1)

    return data, label

def get_random_signal(length,channels):
    X = np.empty((length,channels))
    
    t = np.linspace(1,length,length)
    
    for ch in range(channels):
        X[:,ch] = np.sin(t) + np.random.normal(scale=0.1, size=len(t))
    
    return X

def read_data(P):
    '''
    Reads the individual data sets for all three users

    Parameters
    ----------
    P.dataset : (Str) Name of the dataset
    P.location : (Str) Name of the sensor location

    Returns
    -------
    [[Data Xi, Labels Yi], ... i ∈ (1,2,3)]

    '''        
    V = []
    noise = P.get('noise')
    
    if P.get('dataset') == 'SHL':
        V = [ read_user(P, uid='User%d'%i, noise=noise) for i in range(1,4) ]
    elif P.get('dataset') == 'Short':
        for _ in range(1,4):
            X = get_random_signal(P.get('dummy_size'),len(P.get('channels')) - (2 if P.get('magnitude') else 0))
            Y = np.empty((P.get('dummy_size'),1))
            for i in range(0,Y.shape[0],500):
                Y[i:i+500] = np.random.choice(P.get('labels'))
                
            if noise>0.0:
                X += np.random.normal(0.0, noise, X.shape)
            V.append([pd.DataFrame(X),pd.DataFrame(Y)])
    elif P.get('dataset') == 'Test':
        L = int(P.get('dummy_size')/12)
        P.set('labels',[1,2,3])
        for _ in range(1,4):
            X = np.concatenate((repmat([1, -1],1,L*2),
                    repmat([1, 0, -1, 0],1,L),
                    repmat([1, 2],1,L*2)),
                   axis = 1).T
            
            Y = np.concatenate((
                    np.array([1]*L*4),
                    np.array([2]*L*4),
                    np.array([3]*L*4)))
            
            if noise>0.0:
                X = np.random.normal(0.0, noise, X.shape) + X
            V.append([pd.DataFrame(X),pd.DataFrame(Y)])
    elif P.get('dataset') == 'Sincos':
        L = int(P.get('dummy_size')/2)
        P.set('labels',[1,2])
        for _ in range(1,4):
            base = np.linspace(0,L,L,dtype=int)
            X = np.concatenate(
                    (
                    np.sin(base).reshape(1,-1),
                    np.cos(base).reshape(1,-1),
                    ),
                   axis = 1).T
            
            Y = np.concatenate((
                    np.array([1]*L),
                    np.array([2]*L)))
            
            if noise>0.0:
                X += np.random.normal(0.0, noise, X.shape)
            V.append([pd.DataFrame(X),pd.DataFrame(Y)])
    return V

def hash_exists(path):
    return os.path.isdir(path)

def load_processed(path):
    assert hash_exists(path)
    
    F = []
    for i in range(3):
        X = pd.read_csv(path+'X%d.csv'%i,header=None).to_numpy()
        Y = pd.read_csv(path+'Y%d.csv'%i,header=None).to_numpy()
        F.append([X,Y])
    
    return F
        
def save_processed(F,path):
    assert not hash_exists(path)

    os.makedirs(path, exist_ok=True)
    
    for i,(X,Y) in enumerate(F):
        df = pd.DataFrame(data=X, index=None, columns=None)
        df.to_csv(path+'X%d.csv'%i,header=False,index=False)
        df = pd.DataFrame(data=Y, index=None, columns=None)
        df.to_csv(path+'Y%d.csv'%i,header=False,index=False)
    
def process_data(P,V=None):
    if V is None:
        V = read_data(P)
    
    F = [slidingWindow(P,X,Y) for (X,Y) in V]
    
    return F

def load_data(P):
    '''
    Checks if the selected dataset-location combination is already extracted.
    If not, the according data is loaded, features extracted, and the result stored.
    Then the selected data and - if available - according labels are loaded and returned.

    Parameters
    ----------
    dataset : name of the dataset
    location : location of the sensor
    FX_sel : selection of features

    Parameters
    ----------
    P.dataset : (Str) Name of the dataset
    P.location : (Str) Name of the sensor location
    P.FX_sel : (Str) Selection of extracted features

    Returns
    -------
    [[Features Xi, Labels Yi], ... i ∈ (1,2,3)]

    '''   

    assert P.get('dataset') in PATHS.keys()
    assert P.get('location') in ['Hand','Hips','Bag','Torso']
    
    assert all(channel in NAMES_X[1:] or channel == 'Magnitude' for channel in P.get('channels'))
    assert all(channel in NAMES_X[1:] for channel in ACC_CHANNELS) or not P.get('magnitude') 
    
    log("Loading dataset %s.. (Location: %s | FX: %s)"%(P.get('dataset'),P.get('location'),P.get('FX_sel')),name=P.get('log_name'))
    
    if P.get('dataset') in SAVE_DATA:
        dataset_hash = P.get_dataset_hash_str()
        hash_path = get_path(P,dataset='Hash') + dataset_hash + '/'
        if hash_exists(hash_path):
            F = load_processed(hash_path)
            log("Loaded processed data (%s)."%dataset_hash,name=P.get('log_name'))
            return F

    V = read_data(P)
    log("Read data.",name=P.get('log_name'))

    F = process_data(P,V)
    log("Processed data.",name=P.get('log_name'))
    
    if P.get('dataset') in SAVE_DATA:
        save_processed(F,hash_path)
        log("Saved processed data (%s)."%dataset_hash,name=P.get('log_name'))
        
    return F
    
  
if __name__ == "__main__":
    from params import Params
    
    dataset = 'SHL'
    #dataset = 'Short'
    
    FX_sel = 'basic'
    #FX_sel = 'all'
    
    labels = None
    #labels = [1,2,3]
    
    P = Params(dataset=dataset,labels=labels,FX_sel=FX_sel)
    
    F = load_data(P)
    
    for i,(X,Y) in enumerate(F):
        print("#--------------#")
        print("User",i+1)
        print("Windows:",X.shape)
        print("Labels:",{int(k):v for k,v in zip(*np.unique(Y, return_counts=True))})
        
