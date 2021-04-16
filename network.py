# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import warnings

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from params import M_PATH, make_dir_mod
else:
    # uses current package visibility
    from .params import M_PATH, make_dir_mod

def hardmax(logits):
    y = F.softmax(logits, dim=-1)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

activation_functions = {
    "relu" : nn.ReLU(inplace=False),
    "leaky" : nn.LeakyReLU(0.01, inplace=False),
    "leaky20" : nn.LeakyReLU(0.2, inplace=False),
    'sig': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'softmax': nn.Softmax(dim=1),
    'hardmax': hardmax,
    }

# -------------------
#  Generator
# -------------------

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_no=1, ac_func='leaky', aco_func='tanh'):
        super(Generator, self).__init__()
        self.ac = activation_functions[ac_func]
        self.aco = activation_functions[aco_func]
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = []
        for _ in range(hidden_no):
            self.hidden.append(nn.Linear(hidden_size, hidden_size))
        self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(hidden_size, output_size)

        self.apply(weights_init_normal)

    def forward(self, x):
        out = self.input(x)
        out = self.ac(out)
        for layer in self.hidden:
            out = layer(out)
            out = self.ac(out)
        out = self.output(out)
        out = self.aco(out)
        return out


# -------------------
#  Discriminator
# -------------------    

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_no=1, ac_func='leaky', aco_func='sig'):
        super(Discriminator, self).__init__()
        self.ac = activation_functions[ac_func]
        self.aco = activation_functions[aco_func]
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = []
        for _ in range(hidden_no):
            self.hidden.append(nn.Linear(hidden_size, hidden_size))
        self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(hidden_size, 1)
        
        self.apply(weights_init_normal)

    def forward(self, x):
        out = self.input(x)
        out = self.ac(out)
        for layer in self.hidden:
            out = layer(out)
            out = self.ac(out)
        out = self.output(out)
        out = self.aco(out)
        return out


# -------------------
#  Classifier
# -------------------

class Classifier(nn.Module):
    ''' Gumbel Softmax (Discrete output is default) '''
    def __init__(self, input_size, hidden_size, num_classes, hidden_no=1, ac_func='relu', aco_func='gumbel', hard=True, tau=1):
        super(Classifier, self).__init__()
        self.ac = activation_functions[ac_func]
        if aco_func == 'gumbel':
            def gumbel(logits): return F.gumbel_softmax(logits, tau=tau, hard=hard, eps=1e-10, dim=1)
            self.aco = gumbel
        else:
            self.aco = activation_functions[aco_func]
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = []
        for _ in range(hidden_no):
            self.hidden.append(nn.Linear(hidden_size, hidden_size))
        self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(hidden_size, num_classes)
        self.soft = activation_functions['softmax']
        
        self.apply(weights_init_normal)
        
    def forward(self, x):
        out = self.input(x)
        out = self.ac(out)
        for layer in self.hidden:
            out = layer(out)
            out = self.ac(out)
        out = self.output(out)
        if self.training:
            out = self.aco(out)
        else:
            out = self.soft(out)
        return out

# -------------------
#  Support Functions
# -------------------

def weights_init_normal(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# -------------------
#  Save/Load Networks
# -------------------

def get_string_name(name,model,num=None):
    assert model in ['G','D','C','R']

    if num is None:
        return '%s_%s'%(name,model)
    else:
        return '%s_n%d_%s'%(name,num,model)
    
def save_Model(name,model):
    make_dir_mod()
    PATH = M_PATH+name+'.pt'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.save(model, PATH)
#    log("Saved model "+name)

def save_R(name,R,num=None):
    save_Model(get_string_name(name,'R',num),R)

def save_GAN(name,G,D,C,R=None,num=None):
    save_Model(get_string_name(name,'G',num),G)
    save_Model(get_string_name(name,'D',num),D)
    save_Model(get_string_name(name,'C',num),C)
    if R is not None:
        save_R(name,R,num)
    
def activate_CUDA(*args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if len(args)>1:
        return [model.to(device) for model in args]
    else:
        return args[0].to(device)
    
def load_G(P):
    input_size, output_size = P.get_IO_shape()
    G = Generator(input_size=P.get('noise_shape')+output_size, hidden_size=P.get('G_hidden'), output_size=input_size, hidden_no=P.get('G_hidden_no'), ac_func=P.get('G_ac_func'), aco_func=P.get('G_aco_func'))
    if P.get('CUDA'): return activate_CUDA(G)
    else: return G
    
def load_D(P):
    input_size, output_size = P.get_IO_shape()
    D = Discriminator(input_size=input_size+output_size, hidden_size=P.get('D_hidden'), hidden_no=P.get('D_hidden_no'), ac_func=P.get('D_ac_func'), aco_func=P.get('D_aco_func'))
    if P.get('CUDA'): return activate_CUDA(D)
    else: return D
    
def load_C(P):
    input_size, output_size = P.get_IO_shape()
    C = Classifier(input_size=input_size, hidden_size=P.get('C_hidden'), num_classes=output_size, hidden_no=P.get('C_hidden_no'), ac_func=P.get('C_ac_func'), aco_func=P.get('C_aco_func'), hard=True, tau=P.get('C_tau'))
    if P.get('CUDA'): return activate_CUDA(C)
    else: return C
 
def load_R(P):
    input_size, output_size = P.get_IO_shape()
    R = Classifier(input_size=input_size, hidden_size=P.get('R_hidden'), num_classes=output_size, hidden_no=P.get('R_hidden_no'), ac_func=P.get('R_ac_func'), aco_func=P.get('R_aco_func'), hard=True, tau=P.get('R_tau'))
    if P.get('CUDA'): return activate_CUDA(R)
    else: return R   
 
def load_GAN(P):
    return load_G(P), load_D(P), load_C(P)

# -------------------
#  Loss
# -------------------

from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor
from typing import Optional

class CrossEntropyLoss_OneHot(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
            super(CrossEntropyLoss_OneHot, self).__init__(weight, size_average, reduce, reduction)
            self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        _, labels = target.max(dim=1)
        return F.cross_entropy(input, labels, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

# -------------------
#  Optimiser
# -------------------

def get_optimiser(P,model,params):
    assert model in ['G','D','C']
    optim = P.get(model+'_optim')
    assert optim in ['Adam','AdamW','SGD']
    
    if optim == 'Adam':
        return torch.optim.Adam(params, lr=P.get(model+'LR'), betas=(P.get(model+'B1'), P.get(model+'B2')))
    elif optim == 'AdamW':
        return torch.optim.AdamW(params, lr=P.get(model+'LR'), betas=(P.get(model+'B1'), P.get(model+'B2')))
    elif optim == 'SGD':
        return torch.optim.SGD(params, lr=P.get(model+'LR'), momentum=P.get(model+'B1'))

# -------------------
#  Clear
# -------------------

def clear(name):
    cleared = False
    if os.path.isdir(M_PATH):
        for fname in os.listdir(M_PATH):
            if fname.startswith(name):
                os.remove(os.path.join(M_PATH, fname))
                cleared = True
    return cleared

def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------------
#  Save/Load Distribution Differences
# -------------------

def save_G_Diff(P,mat):
    make_dir_mod()
    PATH = M_PATH+P.get('name')+'_diff_G.pt'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.save({'diff_G':mat}, PATH)
        
def load_G_Diff(P):
    PATH = M_PATH+P.get('name')+'_diff_G.pt'
    mat = np.zeros((P.get('runs'),int(P.get('epochs')/P.get('save_step'))+1,len(P.get('label'))))
    
    if not os.path.isfile(PATH):
        P.log("Could not find G differences for model \"%s\""%P.get('name'))
    else:
        diff = torch.load(PATH)
        mat = fit_array(mat,diff['diff_G'])
        P.log("Loaded G differences for model \"%s\""%P.get('name'))
    return mat

# -------------------
#  Save/Load Accuracy
# -------------------

def save_R_Acc(P,mat_R):
    make_dir_mod()
    PATH = M_PATH+P.get('name')+'_acc_R.pt'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.save({'mat_R':mat_R}, PATH)
#    log("Saved accuracies of model "+name)
    
def load_R_Acc(P):
    PATH = M_PATH+P.get('name')+'_acc_R.pt'
    mat_R = np.zeros((P.get('runs'),int(P.get('epochs')/P.get('save_step'))+1))
    #mat_R[:,0] = 1.0/pp.get_size(P)[1]
    mat_R[:,0] = 0.0
    
    if not os.path.isfile(PATH):
        P.log("Could not find accuracies for model \"%s\""%P.get('name'))
    else:
        acc = torch.load(PATH)
        mat_R = fit_array(mat_R,acc['mat_R'])
        P.log("Loaded accuracies for model \"%s\""%P.get('name'))
    return mat_R

def save_Acc(P,mat_G,mat_D,mat_C):
    make_dir_mod()
    PATH = M_PATH+P.get('name')+'_acc.pt'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.save({'mat_G':mat_G,'mat_D':mat_D,'mat_C':mat_C}, PATH)
#    log("Saved accuracies of model "+name)
    
def load_Acc(P):
    PATH = M_PATH+P.get('name')+'_acc.pt'
    mat_G = np.zeros((P.get('runs'),int(P.get('epochs')/P.get('save_step'))+1))
    #mat_G[:,0] = 0.5
    mat_G[:,0] = 0.0
    mat_D = np.zeros((P.get('runs'),int(P.get('epochs')/P.get('save_step'))+1))
    #mat_D[:,0] = 0.5
    mat_D[:,0] = 0.0
    mat_C = np.zeros((P.get('runs'),int(P.get('epochs')/P.get('save_step'))+1))
    #mat_C[:,0] = 1.0/pp.get_size(P)[1]
    mat_C[:,0] = 0.0

    if not os.path.isfile(PATH):
        P.log("Could not find accuracies for model \"%s\""%P.get('name'))
    else:
        acc = torch.load(PATH)
        mat_G = fit_array(mat_G,acc['mat_G'])
        mat_D = fit_array(mat_D,acc['mat_D'])
        mat_C = fit_array(mat_C,acc['mat_C'])
        P.log("Loaded accuracies for model \"%s\""%P.get('name'))
    return mat_G, mat_D, mat_C

def fit_array(target,source):
    ''' Fit the content of array 'source' into array 'target' '''
    if source.shape[1]>target.shape[1]:
        source = source[:,:target.shape[1]]
    if source.shape[0]>target.shape[0]:
        source = source[:target.shape[0]]
    target[:source.shape[0],:source.shape[1]] = source
    return target

if __name__ == "__main__":
    print("PyTorch version:",torch.__version__)
    print("      GPU Count:",torch.cuda.device_count())
    print(" Cuda available:",torch.cuda.is_available())
    print("  cudnn enabled:",torch.backends.cudnn.enabled)