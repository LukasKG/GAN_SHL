# -*- coding: utf-8 -*-
from hyperopt import hp, fmin, tpe, Trials, space_eval
from hyperopt.pyll import scope
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch

# from sklearn.exceptions import DataConversionWarning

# # Raise exception for warnings
# import warnings
# warnings.filterwarnings('error', category=DataConversionWarning)

if __package__ is None or __package__ == '':
    import data_source as ds
    import GAN
    from log import log
    from params import Params, save_fig
    from plot_confusion_matrix import plot_confusion_matrix
    import preprocessing as pp
else:
    from . import data_source as ds
    from . import GAN
    from .log import log
    from .params import Params, save_fig
    from .plot_confusion_matrix import plot_confusion_matrix
    from . import preprocessing as pp


def hyperopt_Search(P,param_space,evals=100):
    P.set('R_active',False)
    P.set('save_step',P.get('epochs'))
    log("Params: "+str(P),name='hyperopt')
    F = ds.load_data(P)
    
    if torch.cuda.is_available():
        log("CUDA Training.",name='hyperopt')
    else:
        log("CPU Training.",name='hyperopt')
    
    XTL, YTL = F[0]
    XTU, YTU = F[1]
    XTV, YTV = F[2]
    
    XTL = pp.scale_minmax(XTL)
    XTU = pp.scale_minmax(XTU)
    XTV = pp.scale_minmax(XTV)
    
    DL_U_iter = pp.get_perm_dataloader(P, XTU, pp.labels_to_one_hot(P,YTU))
    DL_V = pp.get_dataloader(P, XTV, pp.labels_to_one_hot(P,YTV), batch_size=1024)
    
    def obj(args):
        P0 = P.copy()
        P0.update(args)
        ACC = None
        for run in range(P.get('runs')):
            XL, YL = XTL, YTL
            count_L = YTL.shape[0]
            if P.get('oversampling'):
                XL, YL = pp.over_sampling(P0, XL, YL)
                log("Oversampling: created %d new labelled samples."%( XL.shape[0]-count_L ),name=P.get('log_name'))
        
            DL_L = pp.get_dataloader(P, XL, pp.labels_to_one_hot(P,YL))
        
            mat_accuracy, G, D, C, _ = GAN.train_GAN(P0, DL_L, DL_U_iter, DL_V, name=P0.get('name')+'_%d'%run)
            
            if ACC is None:
                ACC = np.expand_dims(mat_accuracy,axis=2)
            else:
                ACC = np.concatenate((ACC, np.expand_dims(mat_accuracy,axis=2)),axis=2)
        acc = -np.mean(ACC[2],axis=1)[-1]
        log(f"Perf: {acc:.5f} - Checked Params: ("+", ".join([str(key)+': '+str(val) for key,val in args.items()]),name='hyperopt')
        return -np.mean(ACC[2],axis=1)[-1]
 
    trials = Trials()
    best_param = fmin(obj, param_space, algo=tpe.suggest, max_evals=evals, trials=trials, rstate=np.random.RandomState(42))
    log("Best Params:",name='hyperopt')
    for key,val in space_eval(param_space, best_param).items():
        log(str(key)+': '+str(val),name='hyperopt')

def get_Results(P):
    log("Params: "+str(P),name=P.get('log_name'))
    F = ds.load_data(P)
    
    if torch.cuda.is_available():
        log("CUDA Training.",name=P.get('log_name'))
    else:
        log("CPU Training.",name=P.get('log_name'))
    
    XTL, YTL = F[0]
    XTU, YTU = F[1]
    XTV, YTV = F[2]
    
    XTL = pp.scale_minmax(XTL)
    XTU = pp.scale_minmax(XTU)
    XTV = pp.scale_minmax(XTV)
    
    DL_U_iter = pp.get_perm_dataloader(P, XTU, pp.labels_to_one_hot(P,YTU))
    DL_V = pp.get_dataloader(P, XTV, pp.labels_to_one_hot(P,YTV), batch_size=1024)
    
    ACC = None
    YF = None
    RF = None
    PF = None
    
    # -------------------
    #  Individual runs
    # -------------------
    
    for run in range(P.get('runs')):
        
    
        XL, YL = XTL, YTL
        count_L = YTL.shape[0]
        if P.get('oversampling'):
            XL, YL = pp.over_sampling(P, XL, YL)
            log("Oversampling: created %d new labelled samples."%( XL.shape[0]-count_L ),name=P.get('log_name'))
    
    
        DL_L = pp.get_dataloader(P, XL, pp.labels_to_one_hot(P,YL))
    
        mat_accuracy, G, D, C, R = GAN.train_GAN(P, DL_L, DL_U_iter, DL_V, name=P.get('name')+'_%d'%run)
        
        if ACC is None:
            ACC = np.expand_dims(mat_accuracy,axis=2)
        else:
            ACC = np.concatenate((ACC, np.expand_dims(mat_accuracy,axis=2)),axis=2)
        
        for XV, YV in DL_V:
            
            # Classify Validation data
            PC = C(XV)

            if YF == None:
                YF = YV
                PF = PC
            else:
                YF = torch.cat((YF, YV), 0)
                PF = torch.cat((PF, PC), 0)
                
            if P.get('R_active'):
                if RF == None:
                    RF = R(XV)
                else:
                    RF = torch.cat((RF, R(XV).detach()), 0)
        
    return ACC, (YF, RF, PF)

def evaluate(P):
    P.set('R_active',True)
    ACC, (YF, RF, PF) = get_Results(P)
    
    # -------------------
    #  Plot Accuracy
    # -------------------
    
    timeline = np.arange(0,P.get('epochs')+1,P.get('save_step'))
    
    acc_G = np.mean(ACC[0],axis=1)
    std_G = np.std(ACC[0],axis=1)
    acc_D = np.mean(ACC[1],axis=1)
    std_D = np.std(ACC[1],axis=1)
    acc_C = np.mean(ACC[2],axis=1)
    std_C = np.std(ACC[2],axis=1)
    acc_R = np.mean(ACC[3],axis=1)
    
    fig, ax = plt.subplots()    
    
    legend = []  
    cmap = plt.get_cmap('gnuplot')
    indices = np.linspace(0, cmap.N, 7)
    colors = [cmap(int(i)) for i in indices]

    ax.plot(timeline,acc_C,c=colors[0],linestyle='solid')
    ax.fill_between(timeline, acc_C-std_C, acc_C+std_C, alpha=0.3, facecolor=colors[0])
    legend.append("Accuracy $A_C$")
    
    ax.plot(timeline,acc_D,c=colors[1],linestyle='dashed')
    ax.fill_between(timeline, acc_D-std_D, acc_D+std_D, alpha=0.3, facecolor=colors[1])
    legend.append("Accuracy $A_D$")
    
    ax.plot(timeline,acc_G,c=colors[2],linestyle='dotted')
    ax.fill_between(timeline, acc_G-std_G, acc_G+std_G, alpha=0.3, facecolor=colors[2])
    legend.append("Accuracy $A_G$")
    
    Y_max = 1.15
    ax.plot(timeline,acc_R,c=colors[3],linestyle='dashdot')
    legend.append("Accuracy $A_R$")
    
    perf = np.zeros_like(acc_C)
    perf[0] = 0.0
    perf[1:] = (acc_C[1:]-acc_R[1:])/acc_R[1:]

    ax.plot(timeline,perf+1,c=colors[4],linestyle='solid')
    legend.append("Performance $P_C$")
    
    ax.set_xlim(0.0,P.get('epochs'))
    ax.set_ylim(0.0,Y_max)
    
    ax.legend(legend,fontsize=20)
    ax.set_xlabel('Epoch',fontsize=20)
    ax.set_ylabel('Accuracy',fontsize=20)
        
    ax.grid()
    save_fig(P,'eval',fig)
  
    YF = pp.one_hot_to_labels(P,YF)
    RF = pp.one_hot_to_labels(P,RF)
    PF = pp.one_hot_to_labels(P,PF)
    
    con_mat = confusion_matrix(YF, PF, labels=None, sample_weight=None, normalize=None)
    plot_confusion_matrix(con_mat,P,name='C',title='Confusion matrix')
    
    con_mat = confusion_matrix(YF, RF, labels=None, sample_weight=None, normalize=None)
    plot_confusion_matrix(con_mat,P,name='R',title='Confusion matrix')
    

def basic_search(P,evals=100):
    
    dataset = 'SHL'
    #dataset = 'Short'
    
    R_active = False
    
    print_epoch = False
    
    labels = [2,3]
    labels = None
    
    epochs = 500
    epochs = 100
    
    runs = 5
    
    evals = 100
    
    P = Params(dataset=dataset,R_active=R_active,print_epoch=print_epoch,labels=labels,epochs=epochs,runs=runs)
    
    param_space= {
        'GLR'             : hp.loguniform('GLR', np.log(0.0001), np.log(0.1)),
        'GB1'             : hp.loguniform('GB1', np.log(0.01), np.log(0.99)),
        'DLR'             : hp.loguniform('DLR', np.log(0.0001), np.log(0.1)),
        'DB1'             : hp.loguniform('DB1', np.log(0.01), np.log(0.99)),
        'CLR'             : hp.loguniform('CLR', np.log(0.0001), np.log(0.1)),
        'CB1'             : hp.loguniform('CB1', np.log(0.01), np.log(0.99)),
        
        'G_ac_func'       : hp.choice('G_ac_func',['relu','leaky','leaky20','sig']),
        'G_hidden'        : scope.int(hp.qloguniform('G_hidden', np.log(16), np.log(1024), q=1)),
        
        'D_ac_func'       : hp.choice('D_ac_func',['relu','leaky','leaky20','sig']),
        'D_hidden'        : scope.int(hp.qloguniform('D_hidden', np.log(16), np.log(1024), q=1)),
        
        'C_ac_func'       : hp.choice('C_ac_func',['relu','leaky','leaky20','sig']),
        'C_hidden'        : scope.int(hp.qloguniform('C_hidden', np.log(16), np.log(1024), q=1)),  
    }
    
    hyperopt_Search(P,param_space,evals=evals)
 
def test_search(P,evals=5):
    dataset = 'SHL'
    dataset = 'Short'
    dataset = 'Test'
    
    R_active = False
    
    print_epoch = False
    
    labels = [2,3]
    labels = None
    
    epochs = 500
    epochs = 10

    
    runs = 10
    
    evals = 5
    
    P = Params(dataset=dataset,R_active=R_active,print_epoch=print_epoch,labels=labels,epochs=epochs,runs=runs)
    
    param_space= {
        'GLR'             : hp.loguniform('GLR', np.log(0.0001), np.log(0.1)),
        'GB1'             : hp.loguniform('GB1', np.log(0.01), np.log(0.99)),
        'DLR'             : hp.loguniform('DLR', np.log(0.0001), np.log(0.1)),
        'DB1'             : hp.loguniform('DB1', np.log(0.01), np.log(0.99)),
        'CLR'             : hp.loguniform('CLR', np.log(0.0001), np.log(0.1)),
        'CB1'             : hp.loguniform('CB1', np.log(0.01), np.log(0.99)),
        
        'G_ac_func'       : hp.choice('G_ac_func',['relu','leaky','leaky20','sig']),
        'G_hidden'        : scope.int(hp.qloguniform('G_hidden', np.log(16), np.log(1024), q=1)),
        
        'D_ac_func'       : hp.choice('D_ac_func',['relu','leaky','leaky20','sig']),
        'D_hidden'        : scope.int(hp.qloguniform('D_hidden', np.log(16), np.log(1024), q=1)),
        
        'C_ac_func'       : hp.choice('C_ac_func',['relu','leaky','leaky20','sig']),
        'C_hidden'        : scope.int(hp.qloguniform('C_hidden', np.log(16), np.log(1024), q=1)),  
    }
    
    hyperopt_Search(P,param_space,evals=evals)
 
def basic_evaluate():
    dataset = 'SHL'
    #dataset = 'Short'
    
    print_epoch = True
    
    labels = None
    
    epochs = 2000
    #epochs = 100

    save_step = 10
    
    runs = 10
    
    CB1 = 0.012031153201334216
    CLR = 0.030655766791392727
    DB1 = 0.9861141865537364
    DLR = 0.0006399236851949257
    GB1 = 0.8373519670009939
    GLR = 0.00012814349338124425
    
    G_ac_func = 'leaky20'
    G_hidden = 128
    
    D_ac_func = 'leaky20'
    D_hidden = 128
    
    C_ac_func = 'relu'
    C_hidden = 256
    
    
    P = Params(dataset=dataset,print_epoch=print_epoch,labels=labels,epochs=epochs,save_step=save_step,runs=runs,
               CB1=CB1,CLR=CLR,DB1=DB1,DLR=DLR,GB1=GB1,GLR=GLR,
               G_ac_func=G_ac_func,G_hidden=G_hidden,D_ac_func=D_ac_func,D_hidden=D_hidden,C_ac_func=C_ac_func,C_hidden=C_hidden,
               )
    evaluate(P)
   
def test_evaluate():
    dataset = 'SHL'
    dataset = 'Short'
    
    
    print_epoch = True
    
    labels = [2,3]

    epochs = 10

    save_step = epochs
    
    runs = 1
    
    P = Params(dataset=dataset,print_epoch=print_epoch,labels=labels,epochs=epochs,save_step=save_step,runs=runs)
    evaluate(P)
   
def main():
    
    param_space= {
        'batch_size'      : scope.int(hp.qloguniform('batch_size', np.log(32), np.log(512), q=1)),
        
        'GLR'             : hp.loguniform('GLR', np.log(0.0001), np.log(0.1)),
        'GB1'             : hp.loguniform('GB1', np.log(0.01), np.log(0.99)),
        'DLR'             : hp.loguniform('DLR', np.log(0.0001), np.log(0.1)),
        'DB1'             : hp.loguniform('DB1', np.log(0.01), np.log(0.99)),
        'CLR'             : hp.loguniform('CLR', np.log(0.0001), np.log(0.1)),
        'CB1'             : hp.loguniform('CB1', np.log(0.01), np.log(0.99)),
        
        'G_ac_func'       : hp.choice('G_ac_func',['relu','leaky','leaky20','sig']),
        'G_hidden'        : scope.int(hp.qloguniform('G_hidden', np.log(16), np.log(1024), q=1)),
        
        'D_ac_func'       : hp.choice('D_ac_func',['relu','leaky','leaky20','sig']),
        'D_hidden'        : scope.int(hp.qloguniform('D_hidden', np.log(16), np.log(1024), q=1)),
        
        'C_ac_func'       : hp.choice('C_ac_func',['relu','leaky','leaky20','sig']),
        'C_hidden'        : scope.int(hp.qloguniform('C_hidden', np.log(16), np.log(1024), q=1)),  
    }
    
    P_test = Params(
        dataset = 'Test',
        print_epoch = False,
        epochs = 10,
        save_step = 1,
        runs = 1,
        
        CB1 = 0.07964796742556941,
        CLR = 0.010792158322475784,
        C_ac_func = 'relu',
        C_hidden = 70,
        DB1 = 0.01888049545477495,
        DLR = 0.00015379379813713692,
        D_ac_func = 'leaky',
        D_hidden = 166,
        GB1 = 0.15817792641445283,
        GLR = 0.008189309393710799,
        G_ac_func = 'relu',
        G_hidden = 521,
        ) 
    
    P = Params(
        dataset = 'SHL',
        print_epoch = True,
        epochs = 10,
        save_step = 2,
        runs = 5,
        
        CB1 = 0.012031153201334216,
        CLR = 0.030655766791392727,
        DB1 = 0.9861141865537364,
        DLR = 0.0006399236851949257,
        GB1 = 0.8373519670009939,
        GLR = 0.00012814349338124425,
        
        G_ac_func = 'leaky20',
        G_hidden = 128,
        
        D_ac_func = 'leaky20',
        D_hidden = 128,
        
        C_ac_func = 'relu',
        C_hidden = 256,
        ) 
    
    hyperopt_Search(P_test,param_space,evals=5)
    evaluate(P_test)
    
    #hyperopt_Search(P,param_space,evals=100)
    #evaluate(P)
    
if __name__ == "__main__":
    main()
