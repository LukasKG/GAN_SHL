# -*- coding: utf-8 -*-
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn import preprocessing
import torch

# if __package__ is None or __package__ == '':
#     import data_source as ds
# else:
#     from . import data_source as ds

def scale_minmax(X):
    ''' Scale data between -1 and 1 to fit the Generators tanh output '''
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
    scaler.fit(X)
    return scaler.transform(X)

def select_random(X0,Y0=None,ratio=1.0):
    ''' Select random samples based on a given target ratio '''
    idx = np.random.choice(len(X0), size=int(ratio*len(X0)), replace=False)
    X1 = X0[idx]
    if Y0 is None:
        Y1 = None
    else:
        Y1 = Y0[idx]
    return X1, Y1
    
def over_sampling(P,X,Y):
    smote = SMOTE(sampling_strategy='not majority',k_neighbors=5)
    data, labels = smote.fit_resample(X, Y.ravel())
    return data, labels

def get_one_hot_labels(P,num):
    ''' Turns a list with label indeces into a one-hot label array '''
    labels = np.random.choice(P.get('labels'), size=num, replace=True, p=None)
    return labels_to_one_hot(P,labels)

def labels_to_one_hot(P,labels):
    ''' Takes a 1d ndarray with categorical labels and encodes them into one hot labels'''
    m = {y:i for i,y in enumerate(sorted(P.get('labels')))}
    Y = np.zeros((labels.shape[0],len(P.get('labels'))))
    for i,y in enumerate(labels.squeeze().astype(int)):
        Y[i,m[y]] = 1
    return Y

def one_hot_to_labels(P,Y):
    ''' Takes a 2d ndarray or torch tensor with one hot label and decodes them into categorical labels '''
    if torch.is_tensor(Y):
        Y = Y.detach().cpu().numpy()
    return np.array([P.get('labels')[np.where(oh==max(oh))[0][0]] for oh in Y])

def get_tensor(*args):
    if torch.cuda.is_available():
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

class Permanent_Dataloader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
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
        dataset = torch.utils.data.TensorDataset(*get_tensor(X,Y))
    else:
        dataset = torch.utils.data.TensorDataset(*get_tensor(X))
    
    if batch_size == None:
        batch_size = P.get('batch_size')
    
    # Configure data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    return dataloader

def get_perm_dataloader(P,X,Y=None,batch_size=None):
    dataloader = get_dataloader(P,X,Y,batch_size)
    perm_dataloader = Permanent_Dataloader(dataloader)
    return perm_dataloader

if __name__ == "__main__":
    from params import Params
    P = Params(labels=[2,3,5])
    labels = np.array([2,2,3,5,2,3])
    Y = labels_to_one_hot(P,labels)
    print(labels)
    print(Y)
    print(one_hot_to_labels(P,Y))