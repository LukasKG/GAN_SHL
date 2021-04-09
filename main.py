# -*- coding: utf-8 -*-
from hyperopt import hp, fmin, tpe, Trials, space_eval
from hyperopt.pyll import scope
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
    from params import Params, save_fig, save_trials, load_trials
    from plot_confusion_matrix import plot_confusion_matrix
    import preprocessing as pp
else:
    from . import data_source as ds
    from . import GAN
    from .params import Params, save_fig, save_trials, load_trials
    from .plot_confusion_matrix import plot_confusion_matrix
    from . import preprocessing as pp

def hyperopt_C(eval_step=25,max_evals=None):
    import network
    epochs = 100
    
    P = Params(
        name = 'Hyperopt_C',
        dataset = 'SHL',
        CUDA = False,
        
        print_epoch = False,
        epochs = epochs,
        save_step = epochs+1,
        runs = 5,
        
        FX_sel = 'basic',
        Cross_val = 'user',
        
        User_L = 3,
        User_U = 2,
        User_V = 1,
        
        C_basic_train = False,
        
        sample_no = None,
        undersampling = True,
        oversampling = False,
        
        ) 
    
    param_space= {
        'epochs'          : scope.int(hp.uniform('epochs',10,200)),
        'batch_size'      : scope.int(hp.qloguniform('batch_size', np.log(32), np.log(512), q=1)),

        'CLR'             : hp.loguniform('CLR', np.log(0.00001), np.log(0.1)),
        'CB1'             : hp.loguniform('CB1', np.log(0.001), np.log(0.99)),
        'C_tau'           : hp.loguniform('C_tau', np.log(0.01), np.log(10.)),
        
        
        'C_ac_func'       : hp.choice('C_ac_func',['relu','leaky','leaky20','sig']),
        'C_aco_func'      : hp.choice('C_aco_func',['gumbel','gumbel_custom']),
        'C_hidden'        : scope.int(hp.qloguniform('C_hidden', np.log(16), np.log(1024), q=1)),  
        'C_optim'         : hp.choice('C_optim',['Adam','AdamW','SGD']),
    }


    DL_L, DL_U_iter, DL_V = pp.get_all_dataloader(P, ds.ds.get_data(P)) 
    input_size, output_size = P.get_IO_shape()
    
    if P.get('CUDA') and torch.cuda.is_available():
        device = torch.device('cuda')
        P.log("Cuda Training")
    else:
        device = torch.device('cpu')
        P.log("CPU Training")
    
    def obj(args):
        P0 = P.copy()
        P0.update(args)
        acc_mat = np.empty(shape=P.get('runs'))
        for run in range(P0.get('runs')):
            
            C = network.new_C(P0,input_size=input_size,hidden_size=P0.get('C_hidden'),num_classes=output_size)
            C_Loss = torch.nn.BCELoss()
            C.to(device)
            C_Loss.to(device)
            optimizer_C = network.get_optimiser(P0,'C',C.parameters())
            
            for epoch in range(P.get('epochs')):
                for X1, Y1 in DL_L:
                    optimizer_C.zero_grad()
                    P1 = C(X1)
                    loss = C_Loss(P1, Y1)
                    loss.backward()
                    optimizer_C.step()
            C.eval()
            with torch.no_grad():
                acc_mat[run] = np.mean([GAN.get_accuracy(C(XV), YV) for (XV, YV) in DL_V])
              
        acc = np.mean(acc_mat)
        P0.log(f"Perf: {acc:.5f} - Checked Params: "+", ".join([str(key)+' = '+ ("'"+val+"'" if isinstance(val,str) else str(val)) for key,val in args.items()]),name='hyperopt')
        return -acc
    
    trials = load_trials(P)
    if trials is None:
        trials = Trials()
    
    while True:
        if max_evals is not None:
            if len(trials.trials) >= max_evals:
                P.log(f"Maximum number of evaluations reached ({max_evals})",name='hyperopt')
                break
        evals = len(trials.trials) + eval_step

        best_param = fmin(obj, param_space, algo=tpe.suggest, max_evals=evals, trials=trials, rstate=np.random.RandomState(42))
        P.log("Best Params:",name='hyperopt')
        for key,val in space_eval(param_space, best_param).items():
            P.log(str(key)+': '+str(val),name='hyperopt')
        
        save_trials(P,trials)
    
def pytorch_baseline(P):
    import torch
    import network
    
    #P.set('CUDA',False)
    P.set('C_aco_func','gumbel')
   
    DL_L, DL_U_iter, DL_V = pp.get_all_dataloader(P, ds.get_data(P)) 
    
    input_size, output_size = P.get_IO_shape()
    C = network.new_C(P,input_size=input_size,hidden_size=P.get('C_hidden'),num_classes=output_size)
    C_Loss = network.CrossEntropyLoss_OneHot()
    
    
    optim = 'Adam'
    
    if optim == 'Adam':
        optimizer_C = torch.optim.Adam(C.parameters(), lr=0.001, betas=(0.9,0.999))
    elif optim == 'AdamW':
        optimizer_C = torch.optim.AdamW(C.parameters(), lr=0.001, betas=(0.9,0.999))
    elif optim == 'SGD':
        optimizer_C = torch.optim.SGD(C.parameters(), lr=0.001, momentum=0.9)
        
    if P.get('CUDA') and torch.cuda.is_available():
        device = torch.device('cuda')
        C.cuda()
        C_Loss.cuda()
        P.log("Cuda Training")
    else:
        device = torch.device('cpu')
        P.log("CPU Training")
    
    for epoch in range(200):
        running_loss_C = 0.0
        C.train()
        for i, (X1, Y1) in enumerate(DL_L, 1):
            optimizer_C.zero_grad()
            P1 = C(X1)
            loss = C_Loss(P1, Y1)
            loss.backward()
            optimizer_C.step()
            running_loss_C += loss.item()
        loss_C = running_loss_C/len(DL_L) 
        with torch.no_grad():
            acc_C_G = np.mean([GAN.get_accuracy(C(XV), YV) for (XV, YV) in DL_V])
            C.eval()
            acc_C_S = np.mean([GAN.get_accuracy(C(XV), YV) for (XV, YV) in DL_V])
        P.log(f"Epoch {epoch+1}: Loss = {loss_C:.4f} | Accuracy Gumbel = {acc_C_G:.4f} | Accuracy Softmax = {acc_C_S:.4f}")

   
def sklearn_baseline(P):
    from sklearn.neural_network import MLPClassifier as MLP
    from sklearn.ensemble import RandomForestClassifier

    F = pp.perform_preprocessing(P, ds.get_data(P))
    
    x_train, y_train = F[0]
    x_test, y_test = F[2]
    
    mlp = MLP(hidden_layer_sizes=(100,100),max_iter=500)
    
    mlp.fit(x_train, y_train.ravel())
    
    score = mlp.score(x_train, y_train.ravel())
    P.log(f"MLP Acc Train: {score:.2f}")
    
    score = mlp.score(x_test, y_test.ravel())
    P.log(f"MLP Acc Test: {score:.2f}")
    
    rfc = RandomForestClassifier()
    
    rfc.fit(x_train, y_train.ravel())
    
    score = rfc.score(x_train, y_train.ravel())
    P.log(f"RFC Acc Train: {score:.2f}")
    
    score = rfc.score(x_test, y_test.ravel())
    P.log(f"RFC Acc Test: {score:.2f}")
    
def hyperopt_Search(P,param_space,eval_step=25,max_evals=None):
    P.set('R_active',False)
    P.set('save_step',P.get('epochs')+1)
    P.log("Params: "+str(P),name='hyperopt')
    
    if P.get('CUDA') and torch.cuda.is_available():
        P.log("CUDA Training.",name='hyperopt')
    else:
        P.log("CPU Training.",name='hyperopt')
    
    F = ds.get_data(P)
    P.log("Data loaded.")
    
    def obj(args):
        P0 = P.copy()
        P0.update(args)

        DL_L, DL_U_iter, DL_V = pp.get_all_dataloader(P0, ds.select_features(F,P0.get('FX_indeces')))
        
        mat_acc = np.empty((P0.get('runs')))
        for run in range(P0.get('runs')):
            _, _, _, C = GAN.train_GAN(P0, DL_L, DL_U_iter, DL_V, name=P0.get('name')+'_%d'%run)
            C.eval()
            with torch.no_grad():
                mat_acc[run] = np.mean([GAN.get_accuracy(C(XV),YV) for XV, YV in DL_V])
            
        acc = np.mean(mat_acc)
        P0.log(f"Perf: {acc:.5f} - Checked Params: "+", ".join([str(key)+' = '+ ("'"+val+"'" if isinstance(val,str) else str(val)) for key,val in args.items()]),name='hyperopt')
        return -acc
 
    trials = load_trials(P)
    if trials is None:
        trials = Trials()
    
    while True:
        if max_evals is not None:
            if len(trials.trials) >= max_evals:
                P.log(f"Maximum number of evaluations reached ({max_evals})",name='hyperopt')
                break
            evals = min(max_evals,len(trials.trials) + eval_step)
        else:
            evals = len(trials.trials) + eval_step

        best_param = fmin(obj, param_space, algo=tpe.suggest, max_evals=evals, trials=trials, rstate=np.random.RandomState(42))
        P.log("Best Params:",name='hyperopt')
        for key,val in space_eval(param_space, best_param).items():
            P.log(str(key)+': '+str(val),name='hyperopt')
        P.log(f"Best Performance: {abs(max(trials.losses())):.5f} - Copy Params: "+", ".join([str(key)+' = '+ ("'"+val+"'" if isinstance(val,str) else str(val)) for key,val in space_eval(param_space, best_param).items()]),name='hyperopt')
        save_trials(P,trials)

    
def get_Results(P,P_val=None):
    P.log("Params: "+str(P))
    
    if P.get('CUDA') and torch.cuda.is_available():
        P.log("CUDA Training.")
    else:
        P.log("CPU Training.")
    
    DL_L, DL_U_iter, DL_V = pp.get_all_dataloader(P, ds.get_data(P))
    
    if P_val is not None:
        P.log("Load Validation data.")
        _, _, DL_V = pp.get_all_dataloader(P_val, ds.get_data(P_val))
    
    ACC = None
    YF = None
    RF = None
    PF = None
    
    # -------------------
    #  Individual runs
    # -------------------
    
    for run in range(P.get('runs')):
    
        mat_accuracy, G, D, C = GAN.train_GAN(P, DL_L, DL_U_iter, DL_V, name=P.get('name')+'_%d'%run)
        
        if P.get('R_active'):
            acc_BASE, R = GAN.train_Base(P, DL_L, DL_U_iter, DL_V, name=P.get('name')+'_%d'%run)
            mat_accuracy = np.concatenate((mat_accuracy,acc_BASE))
            
        if ACC is None:
            ACC = np.expand_dims(mat_accuracy,axis=2)
        else:
            ACC = np.concatenate((ACC, np.expand_dims(mat_accuracy,axis=2)),axis=2)
        
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
        
    return ACC, (YF, RF, PF)

def evaluate(P,P_val=None):
    P.set('R_active',True)
    ACC, (YF, RF, PF) = get_Results(P,P_val)
    
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
    
    plt.figure(figsize=(27,9),dpi=300,clear=True)
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
    
    # perf = np.zeros_like(acc_C)
    # perf[0] = 0.0
    # perf[1:] = (acc_C[1:]-acc_R[1:])/acc_R[1:]

    # ax.plot(timeline,perf+1,c=colors[4],linestyle='solid')
    # legend.append("Performance $P_C$")
    
    ax.set_xlim(0.0,P.get('epochs'))
    ax.set_ylim(0.0,Y_max)
    
    # ax.legend(legend,fontsize=20)
    # ax.set_xlabel('Epoch',fontsize=20)
    # ax.set_ylabel('Accuracy',fontsize=20)
    
    ax.legend(legend)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
        
    ax.grid()
    save_fig(P,'eval',fig)
  
    YF = pp.one_hot_to_labels(P,YF)
    RF = pp.one_hot_to_labels(P,RF)
    PF = pp.one_hot_to_labels(P,PF)

    for Y, name in [(PF,'C'),(RF,'R')]:
        con_mat = confusion_matrix(YF, Y, labels=None, sample_weight=None, normalize=None)
        plot_confusion_matrix(np.divide(con_mat,P.get('runs')).round().astype(int),P,name=name,title='Confusion matrix',fmt='d')
        
        con_mat = confusion_matrix(YF, Y, labels=None, sample_weight=None, normalize='all')
        plot_confusion_matrix(con_mat,P,name=name+'_normalised',title='Confusion matrix',fmt='0.3f')

   
def mrmr(K=None,log=True):
    import pandas as pd
    from sklearn.feature_selection import f_regression
    
    from sliding_window import get_FX_names
    
    if K is None:
        K = 908
    
    P = Params(dataset='SHL',FX_sel='all',cross_val='user')
    F = ds.load_data(P)
    
    X = np.concatenate([X0 for X0,_ in F])
    Y = np.concatenate([Y0 for _,Y0 in F])
   
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
 
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', dest='CUDA', action='store_true')
    parser.add_argument('-cpu', dest='CUDA', action='store_false')
    parser.set_defaults(CUDA=True)
    args = parser.parse_args()
    
    
    param_space= {
        'FX_num'          : scope.int(hp.quniform('FX_num', 1, 908, q=1)),
        'batch_size'      : scope.int(hp.qloguniform('batch_size', np.log(16), np.log(512), q=1)),
        
        'GLR'             : hp.loguniform('GLR', np.log(0.00001), np.log(0.1)),
        'GB1'             : hp.loguniform('GB1', np.log(0.001), np.log(0.99)),
        'DLR'             : hp.loguniform('DLR', np.log(0.00001), np.log(0.1)),
        'DB1'             : hp.loguniform('DB1', np.log(0.001), np.log(0.99)),
        'CLR'             : hp.loguniform('CLR', np.log(0.00001), np.log(0.1)),
        'CB1'             : hp.loguniform('CB1', np.log(0.001), np.log(0.99)),
        
        'G_ac_func'       : hp.choice('G_ac_func',['relu','leaky','leaky20','sig']),
        'G_hidden'        : scope.int(hp.qloguniform('G_hidden', np.log(16), np.log(2048), q=1)),
        'G_hidden_no'     : scope.int(hp.quniform('G_hidden_no', 0, 5, q=1)), 
        'G_optim'         : hp.choice('G_optim',['AdamW','SGD']),
        
        'D_ac_func'       : hp.choice('D_ac_func',['relu','leaky','leaky20','sig']),
        'D_hidden'        : scope.int(hp.qloguniform('D_hidden', np.log(16), np.log(2048), q=1)),
        'D_hidden_no'     : scope.int(hp.quniform('D_hidden_no', 0, 5, q=1)), 
        'D_optim'         : hp.choice('D_optim',['AdamW','SGD']),
        
        'C_ac_func'       : hp.choice('C_ac_func',['relu','leaky','leaky20','sig']),
        #'C_aco_func'      : hp.choice('C_aco_func',['gumbel','hardmax','softmax']),
        'C_hidden'        : scope.int(hp.qloguniform('C_hidden', np.log(16), np.log(2048), q=1)),
        'C_hidden_no'     : scope.int(hp.quniform('C_hidden_no', 0, 5, q=1)), 
        'C_optim'         : hp.choice('C_optim',['AdamW','SGD']),
        'C_tau'           : hp.loguniform('C_tau', np.log(0.01), np.log(10.)),
    }
    
    P_search = Params(
        name = 'Hyper_GAN_3.0',
        dataset = 'SHL',
        CUDA = args.CUDA,
        
        print_epoch = False,
        epochs = 100,
        runs = 5,
        
        FX_sel = 'all',
        Cross_val = 'user',
        
        User_L = 3,
        User_U = 2,
        User_V = 1,
        
        C_basic_train = False,
        
        sample_no = None,
        undersampling = True,
        oversampling = False,
        
        ) 
    
    P_test = Params(
        name = 'Test',
        dataset = 'Test',
        CUDA = args.CUDA,
        
        print_epoch = False,
        epochs = 5,
        save_step = 1,
        runs = 1,
        
        FX_sel = 'all',
        Cross_val = 'user',
        
        C_basic_train = False,
        
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
    
    P = Params(
        name = 'evaluation',
        dataset = 'SHL',
        CUDA = args.CUDA,
        
        #PCA_n_components = 0.85,
        
        print_epoch = False,
        epochs = 2000,
        save_step = 2,
        runs = 5,
        
        FX_sel = 'all',
        FX_num = 11,
        
        Cross_val = 'user',
        
        C_basic_train = False,
        
        sample_no = None,
        undersampling = False,
        oversampling = False,
        
        User_L = 1,
        User_U = 2,
        User_V = 3,
        
        CB1 = 0.022687365760039192, 
        CLR = 0.0011569226643110444, 
        C_ac_func = 'sig', 
        C_hidden = 138, 
        C_hidden_no = 2, 
        C_optim = 'AdamW', 
        C_tau = 1.456271875874318,
        
        DB1 = 0.027028929852659037, 
        DLR = 0.01174597928660055, 
        D_ac_func = 'relu', 
        D_hidden = 931, 
        D_hidden_no = 0, 
        D_optim = 'SGD', 
        GB1 = 0.002328454596567183, 
        GLR = 0.007870570011454539, 
        G_ac_func = 'leaky20', 
        G_hidden = 25, 
        G_hidden_no = 1, 
        G_optim = 'SGD', 
        
        batch_size = 161
        ) 
    
    #mrmr()

    
    TEST = False
    EVAL = False
    SEARCH = False
    
    # TEST = True
    # EVAL = True
    SEARCH = True
    
    if TEST:
        P_test.set_keys(CUDA = True,)
        evaluate(P_test)
        hyperopt_Search(P_test,param_space,eval_step=2,max_evals=5)
        P_test.set_keys(CUDA = False,)
        evaluate(P_test)
        hyperopt_Search(P_test,param_space,eval_step=2,max_evals=5)
    
    if EVAL:
        P_val = P.copy()
        P_val.set_keys(
            sample_no = None,
            undersampling = False,
            oversampling = False,
            )

        P.set_keys(
            name = 'eval_C_GAN_no_cross',
            C_basic_train = False,
            Cross_val = 'none',
            )
        evaluate(P,P_val)
        
        P.set_keys(
            name = 'eval_C_Complete_user_cross',
            C_basic_train = True,
            Cross_val = 'user',
            )
        evaluate(P,P_val)

    if SEARCH:
        hyperopt_Search(P_search,param_space)
    
    # P.set_keys(
    #     name = 'eval_gumbel',
    #     C_aco_func = 'gumbel',
    #     )
    # evaluate(P,P_val)
    # P.set_keys(
    #     name = 'eval_softmax',
    #     C_aco_func = 'softmax',
    #     )
    # evaluate(P,P_val)
    # P.set_keys(
    #     name = 'eval_hardmax',
    #     C_aco_func = 'hardmax',
    #     )
    # evaluate(P,P_val) 
    
    
    #F = pp.perform_preprocessing(P, ds.get_data(P))
    #sklearn_baseline(P)
    #pytorch_baseline(P)
    #hyperopt_C(eval_step=25,max_evals=None)
    
    
if __name__ == "__main__":
    main()
