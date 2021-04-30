# -*- coding: utf-8 -*-
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import numpy as np
from sklearn import preprocessing,decomposition
import torch

# if __package__ is None or __package__ == '':
#     import data_source as ds
# else:
#     from . import data_source as ds

def scale_minmax(X):
    ''' Scale data between -1 and 1 to fit the Generators tanh output '''
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
    return scaler.fit_transform(X)

def select_random(X0,Y0=None,ratio=1.0):
    ''' Select random samples based on a given target ratio '''
    idx = np.random.choice(len(X0), size=int(ratio*len(X0)), replace=False)
    X1 = X0[idx]
    if Y0 is None:
        Y1 = None
    else:
        Y1 = Y0[idx]
    return X1, Y1
    
def over_sampling(X,Y,ss='not majority'):
    Y = Y.ravel()
    if isinstance(ss,dict):
        ss = ss.copy()
        for y, c in zip(*np.unique(Y,return_counts=True)):
            ss[y] = max(ss[y],c)
    sampler = SMOTE(sampling_strategy=ss,k_neighbors=5)
    data, labels = sampler.fit_resample(X, Y)
    return data, labels

def under_sampling(X,Y,ss='not minority'):
    Y = Y.ravel()
    if isinstance(ss,dict):
        ss = ss.copy()
        for y, c in zip(*np.unique(Y,return_counts=True)):
            ss[y] = min(ss[y],c)
    sampler = NearMiss(sampling_strategy=ss)
    data, labels = sampler.fit_resample(X, Y)
    return data, labels

def get_one_hot_labels(P,num):
    ''' Turns a list with label indeces into a one-hot label array '''
    labels = np.random.choice(P.get('labels'), size=max(1,num), replace=True, p=None)
    return labels_to_one_hot(P,labels)

def labels_to_one_hot(P,labels):
    ''' Takes a 1d ndarray with categorical labels and encodes them into one hot labels'''
    m = {y:i for i,y in enumerate(sorted(P.get('labels')))}
    Y = np.zeros((labels.shape[0],len(P.get('labels'))))
    try:
        for i,y in enumerate(labels.reshape(-1).astype(int)):
            Y[i,m[y]] = 1
    except TypeError:
        print(labels)
        print(labels.shape)
    return Y

def one_hot_to_labels(P,Y):
    ''' Takes a 2d ndarray or torch tensor with one hot label and decodes them into categorical labels '''
    if torch.is_tensor(Y):
        Y = Y.detach().cpu().numpy()
    return np.array([P.get('labels')[np.where(oh==max(oh))[0][0]] for oh in Y])

def get_tensor(*args,cuda=True):
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    res = []
    for X in args:
        if not torch.is_tensor(X):
            if isinstance(X,np.ndarray):
                X = torch.from_numpy(X).float()
            elif isinstance(X,list):
                X = torch.Tensor(X).float()
        res.append(X.to(device))
    return res

def perform_preprocessing(P_train, datasets, P_val=None):
    if P_val is None: P_val = P_train
    
    X_full = np.concatenate([X for X, _ in datasets])
    
    ''' Perform standardization '''
    #scaler = preprocessing.StandardScaler(copy=False)
    scaler = preprocessing.RobustScaler(copy=False)
    X_full = scaler.fit_transform(X_full)
    
    ''' Perform principle component analysis '''
    if P_train.get('PCA_n_components') is not None:
        pca = decomposition.PCA(n_components=P_train.get('PCA_n_components'), copy=False)
        X_full = pca.fit_transform(X_full)
    
    ''' Scale data between -1 and 1 to fit the Generators tanh output '''
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
    X_full = scaler.fit_transform(X_full)
    
    F = []
    idx = 0
    for i,(_,Y) in enumerate(datasets):
        X = X_full[idx:idx+Y.shape[0]]
        idx+=Y.shape[0]

        # No Under/Oversampling for unlabelled data
        if i!=1:
            P = P_train if i<2 else P_val
            if P.get('sample_no'):
                if isinstance(P.get('sample_no'),tuple): no = P.get('sample_no')[i]
                else: no = P.get('sample_no')
                samples = {k:no for k in P.get('labels')}
                X,Y = over_sampling(X, Y, samples)
                X,Y = under_sampling(X, Y, samples)
                
            elif P.get('undersampling'):
                X, Y = under_sampling(X, Y)
                
            elif P.get('oversampling'):
                X, Y = over_sampling(X, Y)
            
        F.append([X,Y])
    return F

class Permanent_Dataloader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
    def __len__(self):
        return len(self.dataloader)
    def get_next(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader 
            self.iterator = iter(self.dataloader)
            data = next(self.iterator)
        return data

def get_dataloader(P,X,Y=None,batch_size=None):
    # create your datset
    if Y is not None:
        if Y.ndim == 1 or Y.shape[1]==1:
            Y = labels_to_one_hot(P,Y)
        dataset = torch.utils.data.TensorDataset(*get_tensor(X,Y,cuda=P.get('CUDA')))
    else:
        dataset = torch.utils.data.TensorDataset(*get_tensor(X,cuda=P.get('CUDA')))
    
    if batch_size == None:
        batch_size = P.get('batch_size')
    
    # Configure data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        #num_workers=4,
        #pin_memory=True,
    )
    
    return dataloader

def get_perm_dataloader(P,X,Y=None,batch_size=None):
    dataloader = get_dataloader(P,X,Y,batch_size)
    perm_dataloader = Permanent_Dataloader(dataloader)
    return perm_dataloader

def get_all_dataloader(P, datasets, P_val=None):
    F = perform_preprocessing(P, datasets, P_val)

    DL_L = get_dataloader(P, *F[0])
    DL_U_iter = get_perm_dataloader(P, *F[1])
    DL_V = get_dataloader(P, *F[2], batch_size=1024) 
    
    return DL_L, DL_U_iter, DL_V

if __name__ == "__main__":
    from params import Params
    P = Params(
        labels=[2,5],
        
        sample_no = 8,
        undersampling = True,
        oversampling = True,
        )

    
    X = np.array([[0,5,10,0,5,10,0,5,10,0,5,10,0,5,10,0,5,10],[0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2]]).T
    Y = np.array([2,2,5,2,2,5,2,2,5,2,2,5,2,2,5,2,2,5])
    
    get_all_dataloader(P, [[X,Y],[X,Y],[X,Y]])
    