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

class Generator_01(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, ac_func='leaky', aco_func='tanh'):
        super(Generator_01, self).__init__()
        self.ac = activation_functions[ac_func]
        self.aco = activation_functions[aco_func]
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.ac(out)
        out = self.fc3(out)
        out = self.aco(out)
        return out


# -------------------
#  Discriminator
# -------------------    

class Discriminator_01(nn.Module):
    def __init__(self, input_size, hidden_size, ac_func='leaky', aco_func='sig'):
        super(Discriminator_01, self).__init__()
        self.ac = activation_functions[ac_func]
        self.aco = activation_functions[aco_func]
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.ac(out)
        out = self.fc3(out)
        out = self.aco(out)
        return out


# -------------------
#  Classifier
# -------------------

class Classifier_01(nn.Module):
    ''' Gumbel Softmax (Discrete output is default) '''
    def __init__(self, input_size, hidden_size, num_classes, ac_func='relu', aco_func='gumbel', hard=True, tau=1, train=True):
        super(Classifier_01, self).__init__()
        self.ac = activation_functions[ac_func]
        if aco_func == 'gumbel':
            def gumbel(logits): return F.gumbel_softmax(logits, tau=tau, hard=hard, eps=1e-10, dim=1)
            self.aco = gumbel
        else:
            self.aco = activation_functions[aco_func]
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        self.train = train
        self.soft = activation_functions['softmax']
    
    def mode_train(self): self.train = True
    def mode_eval(self): self.train = False
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.ac(out)
        out = self.fc3(out)
        if self.train:
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

def new_G(P,input_size,hidden_size,output_size):
    if P.get('G_no') == 1:
        G = Generator_01(input_size, hidden_size, output_size, ac_func=P.get('G_ac_func'), aco_func=P.get('G_aco_func'))
    # elif P.get('G_no') == 2:
    #     G = Generator_02(input_size, hidden_size, output_size)
    # elif P.get('G_no') == 3:
    #     G = Generator_03(input_size, hidden_size, output_size)
    else:
        P.log("No model Generator_%s"%str(P.get('G_no')).zfill(2),error=True)
        return None
    
    G.apply(weights_init_normal)
    if P.get('print_epoch'):
        P.log("Created new generator.")
    # save_Model(get_string_name(P.get('name'),run,'G'),G)
    return G

def new_D(P,input_size,hidden_size):
    if P.get('D_no') == 1:
        D = Discriminator_01(input_size, hidden_size, ac_func=P.get('D_ac_func'), aco_func=P.get('D_aco_func'))
    # elif P.get('D_no') == 2:
    #     D = Discriminator_02(input_size, hidden_size)
    # elif P.get('D_no') == 3:
    #     D = Discriminator_03(input_size, hidden_size)
    else:
        P.log("No model Discriminator_%s"%str(P.get('D_no')).zfill(2),error=True)
        return None
    
    D.apply(weights_init_normal)
    if P.get('print_epoch'):
        P.log("Created new discriminator.")
    # save_Model(get_string_name(P.get('name'),run,'D'),D)
    return D

def new_C(P,input_size,hidden_size,num_classes):
    if P.get('C_no') == 1:
        C = Classifier_01(input_size, hidden_size, num_classes, ac_func=P.get('C_ac_func'), aco_func=P.get('C_aco_func'), hard=True, tau=P.get('C_tau'), train=True)
    # elif P.get('C_no') == 2:
    #     C = Classifier_02(input_size, hidden_size, num_classes)
    # elif P.get('C_no') == 3:
    #     C = Classifier_03(input_size, hidden_size, num_classes)
    else:
        P.log("No model Classifier_%s"%str(P.get('C_no')).zfill(2),error=True)
        return None
    
    C.apply(weights_init_normal)
    if P.get('print_epoch'):
        P.log("Created new classifier.")
    # save_Model(get_string_name(P.get('name'),run,'C'),C)
    return C

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

def save_Ref(name,R,num=None):
    save_Model(get_string_name(name,'R',num),R)

def save_GAN(name,G,D,C,R=None,num=None):
    save_Model(get_string_name(name,'G',num),G)
    save_Model(get_string_name(name,'D',num),D)
    save_Model(get_string_name(name,'C',num),C)
    if R is not None:
        save_Ref(name,R,num)
    
def activate_CUDA(*args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if len(args)>1:
        return [model.to(device) for model in args]
    else:
        return args[0].to(device)
    
def load_Model(P,name):
    PATH = M_PATH+name+'.pt'
    
    if not os.path.isfile(PATH):
        if P.get('print_epoch'):
            P.log("Model \"%s\" does not exist."%PATH,error=False)
        return None
    
    model = torch.load(PATH)
    model.eval()
    
    P.log("Loaded model %s."%name)
    
    return model
    
def load_Pretrain_C(P):
    PATH = 'pretrain/'+P.get('pretrain')
    
    # Check for a pretrained model for the individual run
    if os.path.isfile(PATH+'.pt'):
        model = torch.load(PATH+'.pt')
        P.log("Loaded pretrained classifier %s."%(P.get('pretrain')))
    # Check for a general pretrained model
    elif os.path.isfile(PATH+'.pt'):
        model = torch.load(PATH+'.pt')
        P.log("Loaded pretrained classifier %s."%(P.get('pretrain')))
    # No pretrained model found
    else:
        P.log("Did not find pretrained classifier %s."%(P.get('pretrain')))
        return None
    
    model.eval()
    if P.get('CUDA'):
        return activate_CUDA(model)
    else:
        return model

def load_Ref(P,name=None):
    input_size, output_size = P.get_IO_shape()
    if name is None:
        name = P.get('name')
        
    # Load Classifier
    R = load_Model(P,name+'_R')
    if R is None:
        R = new_C(P, input_size=input_size, hidden_size=P.get('C_hidden'), num_classes=output_size)
        
    if P.get('CUDA'):
        return activate_CUDA(R)
    else:
        return R
    
def load_GAN(P,name=None):
    input_size, output_size = P.get_IO_shape()
    if name is None:
        name = P.get('name')

    # Load Generator
    G = load_Model(P,name+'_G')
    if G is None:
        G = new_G(P, input_size=P.get('noise_shape')+output_size, hidden_size=P.get('G_hidden'), output_size=input_size)
        
    # Load Discriminator
    D = load_Model(P,name+'_D')
    if D is None:
        D = new_D(P, input_size=input_size+output_size, hidden_size=P.get('D_hidden'))
        
    # Load Classifier
    C = load_Model(P,name+'_C')
    if C is None and P.get('pretrain') is not None:
        C = load_Pretrain_C(P)
    if C is None:
        C = new_C(P, input_size=input_size, hidden_size=P.get('C_hidden'), num_classes=output_size)
       
    if P.get('CUDA'):
        return activate_CUDA(G, D, C)
    else:
        return G, D, C

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