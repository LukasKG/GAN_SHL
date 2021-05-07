# -*- coding: utf-8 -*-
from hyperopt import hp, fmin, tpe, Trials, space_eval
from hyperopt.pyll import scope
from math import inf as INF
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch

plt.ioff()

# from sklearn.exceptions import DataConversionWarning

# # Raise exception for warnings
# import warnings
# warnings.filterwarnings('error', category=DataConversionWarning)

if __package__ is None or __package__ == '':
    import data_source as ds
    import GAN
    from metrics import calc_accuracy, calc_f1score
    from network import clear_cache, save_results, load_results
    from sklearn_functions import sklearn_baseline, plt_FX_num
    from params import Params, save_fig, save_trials, load_trials
    from plot_confusion_matrix import plot_confusion_matrix
    import preprocessing as pp
else:
    from . import data_source as ds
    from . import GAN
    from .metrics import calc_accuracy, calc_f1score
    from .network import clear_cache, save_results, load_results
    from .params import Params, save_fig, save_trials, load_trials
    from .plot_confusion_matrix import plot_confusion_matrix
    from . import preprocessing as pp
   
# -------------------
#  Baseline
# -------------------

def pytorch_baseline(P,P_val=None,num=None):
    P.set('epochs_GD',0)
    P.log("Params: "+str(P))
    
    if P.get('CUDA') and torch.cuda.is_available():
        P.log("CUDA Training.")
    else:
        P.log("CPU Training.")
    
    DL_L, _, DL_V = pp.get_all_dataloader(P, ds.get_data(P), P_val)
    P.log(f"Number of batches: Labelled = {len(DL_L)} | Validation = {len(DL_V)}")

        
    ACC = None
    F1S = None
    RF = None
    YF = None
    
    # -------------------
    #  Individual runs
    # -------------------
    
    for run in range(P.get('runs')):
    
        R, mat_accuracy, mat_f1_score = GAN.train_Base(P, DL_L, DL_V, name=P.get('name')+'_%d'%run)

        if ACC is None:
            ACC = mat_accuracy
            F1S = mat_f1_score
        else:
            ACC = np.concatenate((ACC, mat_accuracy),axis=0)
            F1S = np.concatenate((F1S, mat_f1_score),axis=0)

        R.eval()
        with torch.no_grad():
            for XV, YV in DL_V:
                
                if RF == None:
                    RF = R(XV)
                    YF = YV
                else:
                    RF = torch.cat((RF, R(XV).detach()), 0)
                    YF = torch.cat((YF, YV), 0)

    # -------------------
    #  Plot Metrics
    # -------------------
    
    timeline = np.arange(0,(P.get('epochs_GD')+P.get('epochs'))+1,P.get('save_step'))
    
    def get_label(name,model):
        if name == "Accuracy": return "Accuracy $A_%s$"%model;
        elif name == "F1 Score": return "F1 Score $F_%s$"%model;
        else: return "NO_NAME_"+model
       
        
    plt.figure(figsize=(27,9),dpi=300,clear=True)
    fig, ax = plt.subplots()  
    
    cmap = plt.get_cmap('gnuplot')
    indices = np.linspace(0, cmap.N, 7)
    colors = [cmap(int(i)) for i in indices]
        
    for k,(name,mat) in enumerate((('Accuracy',ACC),("F1 Score",F1S))):

        mean_C = np.mean(mat,axis=0)
        std_C = np.std(mat,axis=0)

        P.log(f"R {name} Test: {mean_C[-1]:.2f}")

        ax.plot(timeline,mean_C,c=colors[k],linestyle='solid',label=get_label(name,'R'))
        ax.fill_between(timeline, mean_C-std_C, mean_C+std_C, alpha=0.3, facecolor=colors[k])
            
    Y_max = 1.15
    ax.set_xlim(0.0,(P.get('epochs_GD')+P.get('epochs')))
    ax.set_ylim(0.0,Y_max)

    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Performance')
      
    if num is None: name = 'eval_baseline'
    else: name = 'eval_baseline_'+str(num)
    
    ax.grid()
    save_fig(P,name,fig)
  
    YF = pp.one_hot_to_labels(P,YF)
    RF = pp.one_hot_to_labels(P,RF)

    con_mat = confusion_matrix(YF, RF, labels=None, sample_weight=None, normalize=None)
    plot_confusion_matrix(np.divide(con_mat,P.get('runs')).round().astype(int),P,name=name,title='Confusion matrix',fmt='d')
    
    con_mat = confusion_matrix(YF, RF, labels=None, sample_weight=None, normalize='all')
    plot_confusion_matrix(con_mat,P,name=name+'_normalised',title='Confusion matrix',fmt='0.3f')
    

def hyperopt_Search(P,param_space,objective_func,eval_step=5,max_evals=None):
    P.log("Params: "+str(P),name='hyperopt')
    
    if P.get('CUDA') and torch.cuda.is_available():
        P.log("CUDA Training.",name='hyperopt')
    else:
        P.log("CPU Training.",name='hyperopt')
 
    trials = load_trials(P)
    if trials is None:
        trials = Trials()
    
    while True:
        if max_evals is not None:
            if len(trials.trials) >= max_evals:
                P.log(f"Maximum number of evaluations reached ({len(trials.trials)}/{max_evals})",name='hyperopt')
                try:
                    best_param = fmin(objective_func, param_space, algo=tpe.suggest, max_evals=len(trials.trials), trials=trials, rstate=np.random.RandomState(42))
                    P.log(f"Best Performance: {abs(min(trials.losses())):.5f} - Copy Params: "+", ".join([(key if key[0] != 'C' else 'R'+key[1:])+' = '+ ("'"+val+"'" if isinstance(val,str) else str(val)) for key,val in space_eval(param_space, best_param).items()]),name='hyperopt')
                except:
                    P.log("Couldn't log best performance.")
                break
            evals = min(max_evals,len(trials.trials) + eval_step)
        else:
            evals = len(trials.trials) + eval_step
        evals = max(25,evals)
        
        best_param = fmin(objective_func, param_space, algo=tpe.suggest, max_evals=evals, trials=trials, rstate=np.random.RandomState(42))
        save_trials(P,trials)
        P.log("Best Params:",name='hyperopt')
        for key,val in space_eval(param_space, best_param).items():
            P.log(str(key)+': '+str(val),name='hyperopt')
        P.log(f"Best Performance: {abs(min(trials.losses())):.5f} - Copy Params: "+" ".join([key+' = '+ ("'"+val+"'" if isinstance(val,str) else str(val))+',' for key,val in space_eval(param_space, best_param).items()]),name='hyperopt')   
    
def hyperopt_GAN(P,param_space,eval_step=5,max_evals=None,P_val=None):
    P.set('save_step',INF)
    P.set('R_active',False)
    
    F = ds.get_data(P)
    P.log("Data loaded.")
    
    DL_L, DL_U_iter, DL_V = pp.get_all_dataloader(P,F,P_val)
    P.log(f"Number of batches: Labelled = {len(DL_L)} | Unlabelled = {len(DL_U_iter)} | Validation = {len(DL_V)}")
    
    if P.get('CUDA') and torch.cuda.is_available(): floatTensor = torch.cuda.FloatTensor
    else: floatTensor = torch.FloatTensor

    def obj(args):
        P0 = P.copy()
        P0.update(args)
        P0.log("Check Params: "+", ".join([str(key)+' = '+ ("'"+val+"'" if isinstance(val,str) else str(val)) for key,val in args.items()]),name='hyperopt') 
            
        perf_mat_C = np.empty(shape=(2,P.get('runs'),len(DL_V)))
        perf_mat_D = np.empty(shape=(3,P.get('runs')))
        for run in range(P0.get('runs')):
            G, D, C, _, _ = GAN.train_GAN(P0, DL_L, DL_U_iter, DL_V, name=P0.get('name')+'_%d'%run)
            G.eval();D.eval();C.eval();
            
            with torch.no_grad():
                acc_D = np.empty(shape=(2,len(DL_V)))
                for i,(XV, YV) in enumerate(DL_V):
                    YC = C(XV)
                    perf_mat_C[0,run,i] = calc_f1score(YC, YV, average = 'weighted')
                    perf_mat_C[1,run,i] = calc_accuracy(YC, YV)
                    acc_D[0,i] = calc_accuracy(D(torch.cat((XV,YV),dim=1)),floatTensor(XV.shape[0],1).fill_(1.0))
                    acc_D[1,i] = calc_accuracy(D(torch.cat((XV,YC),dim=1)),floatTensor(XV.shape[0],1).fill_(0.0))
                    
                YV = floatTensor(pp.get_one_hot_labels(P0,num=P0.get('batch_size'))) 
                perf_mat_D[0,run] = calc_accuracy(D(torch.cat((G(torch.cat((floatTensor(np.random.normal(0,1,(P0.get('batch_size'),P0.get('noise_shape')))),YV),dim=1)),YV),dim=1)),floatTensor(P0.get('batch_size'),1).fill_(0.0))
                perf_mat_D[1,run] = np.mean(acc_D[0])
                perf_mat_D[2,run] = np.mean(acc_D[1])
              
        perf = np.mean(perf_mat_C.reshape(2,-1),axis=1)
        val = (
            0.15 * np.mean((0.5-perf_mat_D[0])**2)   # G/D acc ideally is around 50%
            + 0.05 * np.mean((1-perf_mat_D[1])**2)   # D is rewarded for accurately classifying real pairs
            + 0.05 * np.mean((1-perf_mat_D[2])**2)   # D is rewarded for accurately classifying classifier predictions
            - perf[0]                               # C F1 score is most important
            )
        P0.log(f"loss = {val:.5f} [Accuracy D = {np.mean(perf_mat_D):.5f} | vs G = {np.mean(perf_mat_D[0]):.5f} | vs C = {np.mean(perf_mat_D[2]):.5f} | vs real = {np.mean(perf_mat_D[1]):.5f}] [C - F1: {perf[0]:.5f} | Accuracy: {perf[1]:.5f}]",name='hyperopt')
        return val
 
    hyperopt_Search(P,param_space,obj,eval_step=eval_step,max_evals=max_evals)
        

def hyperopt_R(P,param_space,eval_step=5,max_evals=None,P_val=None): 
    P.set('save_step',INF)

    F = ds.get_data(P)
    P.log("Data loaded.")
    
    DL_L, _, DL_V = pp.get_all_dataloader(P,F,P_val)
    P.log(f"Number of batches: Labelled = {len(DL_L)} | Validation = {len(DL_V)}")
    
    def obj(args):
        P0 = P.copy()
        P0.update(args)
        P0.log("Check Params: "+", ".join([str(key)+' = '+ ("'"+val+"'" if isinstance(val,str) else str(val)) for key,val in args.items()]),name='hyperopt')
        
        perf_mat = np.empty(shape=(2,P.get('runs'),len(DL_V)))
        for run in range(P0.get('runs')):

            C, _, _ = GAN.train_Base(P0, DL_L, DL_V, name=P0.get('name')+'_%d'%run)
            C.eval()
            with torch.no_grad():
                for i,(XV, YV) in enumerate(DL_V):
                    perf_mat[0,run,i] = calc_f1score(C(XV), YV, average = 'weighted')
                    perf_mat[1,run,i] = calc_accuracy(C(XV), YV)
              
        perf = np.mean(perf_mat.reshape(2,-1),axis=1)
        P0.log(f"F1: {perf[0]:.5f} | Accuracy: {perf[1]:.5f}",name='hyperopt')
        return -perf[0]
    
    hyperopt_Search(P,param_space,obj,eval_step=eval_step,max_evals=max_evals)  


def hyperopt_GD(P,param_space,eval_step=5,max_evals=None,P_val=None):
    P.set('save_step',INF)
      
    F = ds.get_data(P)
    P.log("Data loaded.")
    
    DL_L, _, DL_V = pp.get_all_dataloader(P,F,P_val)
    P.log(f"Number of batches: Labelled = {len(DL_L)} | Validation = {len(DL_V)}")
    
    if P.get('CUDA') and torch.cuda.is_available(): floatTensor = torch.cuda.FloatTensor
    else: floatTensor = torch.FloatTensor
     
    def obj(args):
        P0 = P.copy()
        P0.update(args)

        perf_mat = np.empty(shape=(2,P.get('runs')))
        for run in range(P0.get('runs')):
            G, D, _, _ = GAN.train_GD(P0, DL_L, DL_V, name=P0.get('name')+'_%d'%run)
            G.eval();D.eval();
            
            with torch.no_grad():
                YV = floatTensor(pp.get_one_hot_labels(P0,num=P0.get('batch_size'))) 
                perf_mat[0,run] = calc_accuracy(D(torch.cat((G(torch.cat((floatTensor(np.random.normal(0,1,(P0.get('batch_size'),P0.get('noise_shape')))),YV),dim=1)),YV),dim=1)),floatTensor(P0.get('batch_size'),1).fill_(0.0))
                perf_mat[1,run] = np.mean([calc_accuracy(D(torch.cat((XV,YV),dim=1)),floatTensor(XV.shape[0],1).fill_(1.0)) for XV, YV in DL_V])
                
        val = np.mean((0.5-perf_mat[0])**2) + 0.1 * np.mean((1-perf_mat[1])**2)
        P0.log(f"loss = {val:.5f} Accuracy D = {np.mean(perf_mat):.5f} | vs G = {np.mean(perf_mat[0]):.5f} | vs real = {np.mean(perf_mat[1]):.5f}",name='hyperopt')
        return val
 
    hyperopt_Search(P,param_space,obj,eval_step=eval_step,max_evals=max_evals)
    
      
def get_Results(P,P_val=None):
    P.log("Params: "+str(P))
    
    ACC = load_results(P,name='acc')
    F1S = load_results(P,name='f1')
    YF = load_results(P,name='YF')
    RF = load_results(P,name='RF')
    PF = load_results(P,name='PF')
    
    if any(mat is None for mat in (ACC,F1S,YF,PF)):
    
        if P.get('CUDA') and torch.cuda.is_available():
            P.log("CUDA Training.")
        else:
            P.log("CPU Training.")
        
        DL_L, DL_U_iter, DL_V = pp.get_all_dataloader(P, ds.get_data(P), P_val)
        P.log(f"Number of batches: Labelled = {len(DL_L)} | Unlabelled = {len(DL_U_iter)} | Validation = {len(DL_V)}")    
    
        ACC = None
        F1S = None
        YF = None
        RF = None
        PF = None
        
        # -------------------
        #  Individual runs
        # -------------------
        
        for run in range(P.get('runs')):
        
            G, D, C, mat_accuracy, mat_f1_score = GAN.train_GAN(P, DL_L, DL_U_iter, DL_V, name=P.get('name')+'_%d'%run)
            
            if P.get('R_active'):
                R, acc_BASE, f1_BASE = GAN.train_Base(P, DL_L, DL_V, name=P.get('name')+'_%d'%run)
                mat_accuracy = np.concatenate((mat_accuracy,acc_BASE))
                mat_f1_score = np.concatenate((mat_f1_score,f1_BASE))
            if ACC is None:
                ACC = np.expand_dims(mat_accuracy,axis=2)
                F1S = np.expand_dims(mat_f1_score,axis=2)
            else:
                ACC = np.concatenate((ACC, np.expand_dims(mat_accuracy,axis=2)),axis=2)
                F1S = np.concatenate((F1S, np.expand_dims(mat_accuracy,axis=2)),axis=2)
                
            C.eval()
            if P.get('R_active'):
                R.eval()
                
            with torch.no_grad():
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
        
        save_results(P, ACC, name='acc')
        save_results(P, F1S, name='f1')
        save_results(P,YF,name='YF')
        save_results(P,PF,name='PF')
        if RF is not None:
            save_results(P, RF, name='RF')
        P.log("Saved Accuracy, F1 Score and predictions.")
    else:
        P.log("Loaded Accuracy, F1 Score and predictions.")
        
    return ACC, F1S, (YF, RF, PF)

def evaluate(P,P_val=None):
    P.set('R_active',True)
    ACC, F1S, (YF, RF, PF) = get_Results(P,P_val)

    # -------------------
    #  Plot Metrics
    # -------------------

    def get_label(name,model):
        if name == "Accuracy": return "Accuracy $A_%s$"%model;
        elif name == "F1 Score": return "F1 Score $F_%s$"%model;
        else: return "NO_NAME"+model

    for eval_mode in ['full','GAN']:
        
        if eval_mode == 'GAN':
            ACC = ACC[:,P.get('epochs_GD'):]
            F1S = F1S[:,P.get('epochs_GD'):]

        timeline = np.arange(0,ACC.shape[1]*P.get('save_step'),P.get('save_step'))
        
        for name,mat in (('Accuracy',ACC),("F1 Score",F1S)):
        
            mean_G = np.mean(mat[0],axis=1)
            std_G = np.std(mat[0],axis=1)
            mean_D = np.mean(mat[1],axis=1)
            std_D = np.std(mat[1],axis=1)
            mean_C = np.mean(mat[2],axis=1)
            std_C = np.std(mat[2],axis=1)
            mean_R = np.mean(mat[3],axis=1)
    
            plt.figure(figsize=(27,9),dpi=300,clear=True)
            fig, ax = plt.subplots()    
             
            cmap = plt.get_cmap('gnuplot')
            indices = np.linspace(0, cmap.N, 7)
            colors = [cmap(int(i)) for i in indices]
        
            ax.plot(timeline,mean_C,c=colors[0],linestyle='solid',label=get_label(name,'C'))
            ax.fill_between(timeline, mean_C-std_C, mean_C+std_C, alpha=0.3, facecolor=colors[0])
            
            ax.plot(timeline,mean_D,c=colors[1],linestyle='dashed',label=get_label(name,'D'))
            ax.fill_between(timeline, mean_D-std_D, mean_D+std_D, alpha=0.3, facecolor=colors[1])
            
            ax.plot(timeline,mean_G,c=colors[2],linestyle='dotted',label=get_label(name,'G'))
            ax.fill_between(timeline, mean_G-std_G, mean_G+std_G, alpha=0.3, facecolor=colors[2])
            
            Y_max = 1.15
            ax.plot(timeline,mean_R,c=colors[3],linestyle='dashdot',label=get_label(name,'R'))
            
            # perf = np.zeros_like(mean_C)
            # perf[0] = 0.0
            # perf[1:] = (mean_C[1:]-mean_R[1:])/mean_R[1:]
        
            # ax.plot(timeline,perf+1,c=colors[4],linestyle='solid')
            # legend.append("Performance $P_C$")
            
            ax.set_xlim(0.0,timeline.shape[0]-1)
            ax.set_ylim(0.0,Y_max)
            
            # ax.legend(legend,fontsize=20)
            # ax.set_xlabel('Epoch',fontsize=20)
            # ax.set_ylabel('Accuracy',fontsize=20)
            
            ax.legend()
            ax.set_xlabel('Epoch')
            ax.set_ylabel(name)
                
            ax.grid()
            save_fig(P,f'eval_{eval_mode}_'+('acc' if name == 'Accuracy' else 'f1'),fig)
  
    YF = pp.one_hot_to_labels(P,YF)
    RF = pp.one_hot_to_labels(P,RF)
    PF = pp.one_hot_to_labels(P,PF)

    for Y, name in [(PF,'C'),(RF,'R')]:
        con_mat = confusion_matrix(YF, Y, labels=None, sample_weight=None, normalize=None)
        plot_confusion_matrix(np.divide(con_mat,P.get('runs')).round().astype(int),P,name=name,title='Confusion matrix',fmt='d')
        
        con_mat = confusion_matrix(YF, Y, labels=None, sample_weight=None, normalize='all')
        plot_confusion_matrix(con_mat,P,name=name+'_normalised',title='Confusion matrix',fmt='0.3f')

   
def mrmr(K=908,log=True,down_to=183,dataset='SHL'):
    import pandas as pd
    from sklearn.feature_selection import f_regression
    from preprocessing import under_sampling
    
    from sliding_window import get_FX_names
   
    P = Params(name='mRMR_reduced_user1',dataset=dataset,FX_sel='all',cross_val='user',log_name='mrmr')
    V = ds.load_data(P)
    
    samples = {k:down_to for k in P.get('labels')}
    F = [under_sampling(X0, Y0, samples) for X0, Y0 in V]
    
    X = np.concatenate([X0 for X0,_ in F])
    Y = np.concatenate([Y0 for _,Y0 in F])
   
    P.log(f"X: {X.shape}, Y: {Y.shape}")
    P.log(', '.join([ f"{int(val)}: {count}" for val,count in zip(*np.unique(Y,return_counts=True))]))
   
    X = pd.DataFrame(X, columns = get_FX_names())
    Y = pd.Series(Y.ravel())
    
    F = pd.Series(f_regression(X, Y)[0], index = X.columns)
    corr = pd.DataFrame(.00001, index = X.columns, columns = X.columns)
    
    # initialize list of selected features and list of excluded features
    selected = []
    not_selected = X.columns.to_list()
    
    # repeat K times
    for i in range(K):
      
        # compute (absolute) correlations between the last selected feature and all the (currently) excluded features
        if i > 0:
            last_selected = selected[-1]
            corr.loc[not_selected, last_selected] = X[not_selected].corrwith(X[last_selected]).abs().clip(.00001)
            
        # compute FCQ score for all the (currently) excluded features (this is Formula 2)
        score = F.loc[not_selected] / corr.loc[not_selected, selected].mean(axis = 1).fillna(.00001)
        
        # find best feature, add it to selected and remove it from not_selected
        best = score.index[score.argmax()]
        
        if log:
            P.log(str(i+1).rjust(3,' ')+f': {best} (Score: {score[best]:.4f})')
        selected.append(best)
        not_selected.remove(best)
        
    indeces = [X.columns.get_loc(c) for c in selected]
        
    if log:
        P.log(str(selected))
        P.log(str(indeces))

    return selected, indeces

# -------------------
#  Hyperopt GAN Search
# -------------------

def hyper_GAN_3_5(P_args):
    P_search = P_args.copy()
    P_search.set_keys(
        name = 'Hyper_GAN_3.5',
        dataset = 'SHL_ext',
        
        epochs = 100,
        runs = 5,
        
        batch_size = 512,
        FX_num = 150,
        
        FX_sel = 'all',
        cross_val = 'combined',
        
        sample_no = 512,
        undersampling = False,
        oversampling = False,
        
        C_aco_func = 'gumbel',
        
        D_optim = 'SGD', 

        G_optim = 'SGD',
        
        ) 
    
    param_space={
        'GD_ratio'        : hp.uniform('GD_ratio', 0, 0.9),
        
        'CLR'             : hp.loguniform('CLR', np.log(0.00001), np.log(0.1)),
        'CB1'             : hp.loguniform('CB1', np.log(0.001), np.log(0.99)),
        
        'C_ac_func'       : hp.choice('C_ac_func',['relu','leaky','leaky20','sig']),
        'C_hidden'        : scope.int(hp.qloguniform('C_hidden', np.log(16), np.log(4096), q=1)),
        'C_hidden_no'     : scope.int(hp.quniform('C_hidden_no', 1, 8, q=1)), 
        'C_optim'         : hp.choice('C_optim',['AdamW','SGD']),
        'C_tau'           : hp.loguniform('C_tau', np.log(0.01), np.log(10.)),
        
        'GLR'             : hp.loguniform('GLR', np.log(0.00001), np.log(0.1)),
        'GB1'             : hp.loguniform('GB1', np.log(0.001), np.log(0.99)),
        'DLR'             : hp.loguniform('DLR', np.log(0.00001), np.log(0.1)),
        'DB1'             : hp.loguniform('DB1', np.log(0.001), np.log(0.99)),

        'G_ac_func'       : hp.choice('G_ac_func',['relu','leaky','leaky20','sig']),
        'G_hidden'        : scope.int(hp.qloguniform('G_hidden', np.log(16), np.log(4096), q=1)),
        'G_hidden_no'     : scope.int(hp.quniform('G_hidden_no', 1, 8, q=1)), 
        
        'D_ac_func'       : hp.choice('D_ac_func',['relu','leaky','leaky20','sig']),
        'D_hidden'        : scope.int(hp.qloguniform('D_hidden', np.log(16), np.log(4096), q=1)),
        'D_hidden_no'     : scope.int(hp.quniform('D_hidden_no', 1, 8, q=1)), 
    }   
    
    hyperopt_GAN(P_search,param_space,eval_step=5,max_evals=None,P_val=P_search.copy().set_keys(
        sample_no = None,
        undersampling = False,
        oversampling = False,))
 
# -------------------
#  Hyperopt Baseline Search
# -------------------   

def hyper_R_1_3(P_args):
    P_search = P_args.copy()
    P_search.set_keys(
        name = 'Hyper_R_1.3',
        dataset = 'SHL_ext',
        
        epochs = 100,
        runs = 5,
        
        batch_size = 512,
        
        FX_sel = 'all',
        FX_num = 150,
        cross_val = 'combined',
        
        sample_no = 512,
        undersampling = False,
        oversampling = False,
        
        R_aco_func = 'softmax',
        R_tau = None,
        ) 
    
    param_space={
        'RLR'             : hp.loguniform('RLR', np.log(0.00001), np.log(0.1)),
        'RB1'             : hp.loguniform('RB1', np.log(0.001), np.log(0.99)),
        
        'R_ac_func'       : hp.choice('R_ac_func',['relu','leaky','leaky20','sig']),
        'R_hidden'        : scope.int(hp.qloguniform('R_hidden', np.log(16), np.log(4096), q=1)),
        'R_hidden_no'     : scope.int(hp.quniform('R_hidden_no', 1, 8, q=1)), 
        'R_optim'         : hp.choice('R_optim',['AdamW','SGD']),
    } 
    
    hyperopt_GAN(P_search,param_space,eval_step=5,max_evals=None,P_val=P_search.copy().set_keys(
        sample_no = None,
        undersampling = False,
        oversampling = False,))

# -------------------
#  Hyperopt Baseline Search
# -------------------   

def hyper_GD_1_3(P_args):
    P_search = P_args.copy()
    P_search.set_keys(
        name = 'Hyper_GD_1.3',
        dataset = 'SHL_ext',

        epochs_GD = 100,
        runs = 5,
        
        batch_size = 512,
        FX_num = 150,
        
        FX_sel = 'all',
        cross_val = 'combined',
        
        sample_no = None,
        undersampling = True,
        oversampling = False,
        
        G_optim = 'SGD',
        
        D_optim = 'SGD',
        ) 
    
    param_space={
        'GLR'             : hp.loguniform('GLR', np.log(0.00001), np.log(0.1)),
        'GB1'             : hp.loguniform('GB1', np.log(0.001), np.log(0.99)),
        'DLR'             : hp.loguniform('DLR', np.log(0.00001), np.log(0.1)),
        'DB1'             : hp.loguniform('DB1', np.log(0.001), np.log(0.99)),

        'G_ac_func'       : hp.choice('G_ac_func',['relu','leaky','leaky20','sig']),
        'G_hidden'        : scope.int(hp.qloguniform('G_hidden', np.log(16), np.log(4096), q=1)),
        'G_hidden_no'     : scope.int(hp.quniform('G_hidden_no', 0, 9, q=1)), 
        
        'D_ac_func'       : hp.choice('D_ac_func',['relu','leaky','leaky20','sig']),
        'D_hidden'        : scope.int(hp.qloguniform('D_hidden', np.log(16), np.log(4096), q=1)),
        'D_hidden_no'     : scope.int(hp.quniform('D_hidden_no', 0, 9, q=1)), 
    }
    
    hyperopt_GAN(P_search,param_space,eval_step=5,max_evals=None,P_val=P_search.copy().set_keys(
        sample_no = None,
        undersampling = False,
        oversampling = False,))


def main():
    import argparse
    from params import DEFAULT_PARAMS as default
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-clear', dest='CLEAR', action='store_true')
    parser.set_defaults(CLEAR=False)
    
    parser.add_argument('-test', dest='TEST', action='store_true')
    parser.set_defaults(TEST=False)
    
    parser.add_argument('-eval', dest='EVAL', action='store_true')
    parser.set_defaults(EVAL=False)
    
    parser.add_argument('-eval_r', dest='BASE', action='store_true')
    parser.add_argument('-eval_c', dest='BASE', action='store_true')
    parser.set_defaults(BASE=False)
    
    parser.add_argument('-search', dest='SEARCH', action='store_true')
    parser.set_defaults(SEARCH=False)
    
    parser.add_argument('-search_r', dest='SEARCH_C', action='store_true')
    parser.add_argument('-search_c', dest='SEARCH_C', action='store_true')
    parser.set_defaults(SEARCH_C=False)
    
    parser.add_argument('-search_gd', dest='SEARCH_GD', action='store_true')
    parser.set_defaults(SEARCH_GD=False)
    
    parser.add_argument('-mrmr', dest='MRMR', action='store_true')
    parser.set_defaults(MRMR=False)
    
    parser.add_argument('-sklearn', dest='SKLEARN', action='store_true')
    parser.set_defaults(SKLEARN=False)

    parser.add_argument('-fx_num', type=int, dest='FX_NUM')
    parser.set_defaults(FX_NUM=None)
    
    parser.add_argument('-cuda', dest='CUDA', action='store_true')
    parser.add_argument('-cpu', dest='CUDA', action='store_false')
    parser.set_defaults(CUDA=default['CUDA'])
    
    parser.add_argument('-data_path', type=str, dest='data_path')
    parser.add_argument('-datapath', type=str, dest='data_path')
    parser.set_defaults(data_path=default['data_path'])
    
    parser.add_argument('-dataset', type=str, dest='dataset')
    parser.set_defaults(dataset=default['dataset'])
    
    parser.add_argument('-print', dest='PRINT', action='store_true')
    parser.set_defaults(PRINT=default['print_epoch'])
    
    parser.add_argument('-basic', dest='BASIC', action='store_true')
    parser.add_argument('-no_basic', dest='BASIC', action='store_false')
    parser.set_defaults(BASIC=default['C_basic_train'])
    
    parser.add_argument('-max_evals', type=int, dest='max_evals')
    parser.set_defaults(max_evals=None)
    
    args = parser.parse_args()
    P_args = Params(
        data_path = args.data_path,
        CUDA = args.CUDA,
        print_epoch = args.PRINT,
        C_basic_train = args.BASIC,
        dataset = args.dataset,
        )
    
    P_test = P_args.copy().set_keys(
        name = 'Test',
        dataset = 'Test',

        epochs = 10,
        GD_ratio = 0.5,

        save_step = 1,
        runs = 1,
        
        FX_sel = 'all',
        cross_val = 'user',
        
        sample_no = None,
        undersampling = False,
        oversampling = False,
        
        CB1 = 0.02482259369526197, 
        CLR = 0.00033565485364740803, 
        C_ac_func = 'relu', 
        C_hidden = 92, 
        C_optim = 'AdamW', 
        DB1 = 0.1294935579262613, 
        DLR = 0.010144020667237321, 
        D_ac_func = 'leaky', 
        D_hidden = 317, 
        D_optim = 'AdamW', 
        GB1 = 0.023718651003136713,
        GLR = 0.0005411668775518598, 
        G_ac_func = 'relu', 
        G_hidden = 140, 
        G_optim = 'SGD', 
        batch_size = 110
        ) 
    
    P = P_args.copy().set_keys(
        name = 'eval',

        epochs = 1500,
        save_step = 3,
        runs = 3,
        
        FX_sel = 'all',
        
        cross_val = 'combined',
        
        sample_no = 11136,
        undersampling = False,
        oversampling = False,
        
        User_L = 1,
        User_U = 2,
        User_V = 3,
        
        batch_size = 512,
        FX_num = 150, 
        
        CB1 = 0.9575294338712442, 
        CLR = 0.00019648736861945274, 
        C_ac_func = 'relu', 
        C_hidden = 3663, 
        C_hidden_no = 2, 
        C_optim = 'AdamW', 
        C_tau = 7.81219653857222, 
        
        DB1 = 0.040659167249518575, 
        DLR = 0.007276690625627446, 
        D_ac_func = 'relu', 
        D_hidden = 392, 
        D_hidden_no = 7, 
        
        GB1 = 0.7121429015539442, 
        GD_ratio = 0.15667317600612712, 
        GLR = 0.00010176168444138969, 
        G_ac_func = 'relu', 
        G_hidden = 2360, 
        G_hidden_no = 4,
        
        RB1 = 0.0202574056023114, 
        RLR = 0.0024569186376477503, 
        R_ac_func = 'sig', 
        R_hidden = 893, 
        R_hidden_no = 4, 
        R_optim = 'SGD',

        ) 
    

    if args.CLEAR:
        clear_cache()

    if args.TEST:
        param_space={'GD_ratio': hp.uniform('GD_ratio', 0, 0.9)} 
        
        P_test.set_keys(name='Test_CUDA',CUDA = True,)
        evaluate(P_test)
        hyperopt_GAN(P_test.copy(),param_space,eval_step=2,max_evals=5)
        
        P_test.set_keys(name='Test_CPU',CUDA = False,)
        evaluate(P_test)
        hyperopt_GAN(P_test.copy(),param_space,eval_step=2,max_evals=5)
    
    if args.EVAL:

        P.set_keys(name='eval_complete', sample_no = None, undersampling = False, oversampling = False, )
        evaluate(P,P.copy().set_keys( sample_no = None, undersampling = False, oversampling = False, ))
        
        P.set_keys(name='eval_undersampling', sample_no = 11136, undersampling = False, oversampling = False, )
        evaluate(P,P.copy().set_keys( sample_no = None, undersampling = False, oversampling = False, ))
        
        # for cross_val in ['user','none']:
        #     for basic_train in [True,False]:
        #            P.set_keys(
        #                 name = '_'.join(['eval','C',('Complete' if basic_train else 'GAN'),cross_val,'cross']),
        #                 C_basic_train = basic_train,
        #                 cross_val = cross_val,
        #                 )
        #            evaluate(P,P.copy().set_keys( sample_no = None, undersampling = False, oversampling = False, ))

    if args.BASE:
        P_base1 = P.copy().set_keys( name='eval_c_normal', dataset='SHL', epochs=1500, sample_no=400, undersampling=False, oversampling=False, )
        P_base2 = P.copy().set_keys( name='eval_c_extended', dataset='SHL_ext', epochs=200, sample_no=None, undersampling=True, oversampling=False, )
        
        pytorch_baseline(P_base1,P_base1.copy().set_keys( sample_no = None, undersampling = False, oversampling = False, )) 
        pytorch_baseline(P_base2,P_base2.copy().set_keys( sample_no = None, undersampling = False, oversampling = False, ))
    

    indeces = None
    if args.MRMR:
        down_to = 183
        try: 
            selected, indeces = mrmr(down_to=down_to,dataset='SHL_ext')
        except Exception as e:
            print(e)
            selected, indeces = mrmr(down_to=down_to,dataset='SHL')
        
    if args.SKLEARN:
        sklearn_baseline(P)
        
    if args.FX_NUM is not None:
        P_fx_num = P.copy().set_keys( name='fx_num', dataset='SHL_ext', runs=8, sample_no=512, undersampling=False, oversampling=False, )
        plt_FX_num(P_fx_num,max_n=args.FX_NUM,P_val=P_fx_num.copy().set_keys(sample_no=None, undersampling=False, oversampling=False,),indeces=indeces)

    if args.SEARCH:
        hyper_GAN_3_5(P_args)
    
    if args.SEARCH_C:
        hyper_R_1_3(P_args)
    
    if args.SEARCH_GD:
        hyper_GD_1_3(P_args)
    
if __name__ == "__main__":
    main()