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


def get_Data(P):
    F = ds.load_data(P)
    
    if P.get('Cross_val') == 'user':
        return [F[P.get('User_L')-1], F[P.get('User_U')-1], F[P.get('User_V')-1]]
    
     
    if P.get('Cross_val') == 'none':
        X = np.concatenate([X for X,_ in F])
        Y = np.concatenate([Y for _,Y in F])
        return [[X,Y], [X,Y], [X,Y]]

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


    DL_L, DL_U_iter, DL_V = pp.get_all_dataloader(P, get_Data(P)) 
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
            C.mode_eval()
            acc_mat[run] = np.mean([GAN.get_accuracy(C(XV), YV) for (XV, YV) in DL_V])
              
        acc = np.mean(acc_mat)
        P0.log(f"Perf: {acc:.5f} - Checked Params: "+", ".join([str(key)+' = '+str(val) for key,val in args.items()]),name='hyperopt')
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
    
    P.set('CUDA',False)
    P.set('C_aco_func','gumbel')
   
    DL_L, DL_U_iter, DL_V = pp.get_all_dataloader(P, get_Data(P)) 
    
    input_size, output_size = P.get_IO_shape()
    C = network.new_C(P,input_size=input_size,hidden_size=P.get('C_hidden'),num_classes=output_size)
    C_Loss = torch.nn.BCELoss()
    
    
    optim = 'Adam'
    
    if optim == 'Adam':
        optimizer_C = torch.optim.Adam(C.parameters(), lr=0.001, betas=(0.9,0.999))
    elif optim == 'AdamW':
        optimizer_C = torch.optim.AdamW(C.parameters(), lr=0.001, betas=(0.9,0.999))
    elif optim == 'SGD':
        optimizer_C = torch.optim.SGD(C.parameters(), lr=0.001, momentum=0.9)
        
    if P.get('CUDA') and torch.cuda.is_available():
        device = torch.device('cuda')
        C_Loss.cuda()
        P.log("Cuda Training")
    else:
        device = torch.device('cpu')
        P.log("CPU Training")
    
    for epoch in range(200):
        running_loss_C = 0.0
        C.mode_train()
        for i, (X1, Y1) in enumerate(DL_L, 1):
            optimizer_C.zero_grad()
            P1 = C(X1)
            loss = C_Loss(P1, Y1)
            loss.backward()
            optimizer_C.step()
            running_loss_C += loss.item()
        loss_C = running_loss_C/len(DL_L) 
        acc_C_G = np.mean([GAN.get_accuracy(C(XV), YV) for (XV, YV) in DL_V])
        C.mode_eval()
        acc_C_S = np.mean([GAN.get_accuracy(C(XV), YV) for (XV, YV) in DL_V])
        P.log(f"Epoch {epoch}: Loss = {loss_C:.4f} | Accuracy Gumbel = {acc_C_G:.4f} | Accuracy Softmax = {acc_C_S:.4f}")

   
def sklearn_baseline(P):
    from sklearn.neural_network import MLPClassifier as MLP
    
    F = get_Data(P)
    
    x_train, y_train = F[0]
    x_test, y_test = F[2]
    
    clf = MLP(hidden_layer_sizes=(100,100),max_iter=500)
    
    clf.fit(x_train, y_train.ravel())
    
    score = clf.score(x_train, y_train.ravel())
    P.log(f"Acc Train: {score:.2f}")
    
    score = clf.score(x_test, y_test.ravel())
    P.log(f"Acc Test: {score:.2f}")
    
def hyperopt_Search(P,param_space,eval_step=25,max_evals=None):
    P.set('R_active',False)
    P.set('save_step',P.get('epochs')+1)
    P.log("Params: "+str(P),name='hyperopt')
    
    if P.get('CUDA') and torch.cuda.is_available():
        P.log("CUDA Training.",name='hyperopt')
    else:
        P.log("CPU Training.",name='hyperopt')
    
    DL_L, DL_U_iter, DL_V = pp.get_all_dataloader(P, get_Data(P))
    
    def obj(args):
        P0 = P.copy()
        P0.update(args)
        
        mat_acc = np.empty((P0.get('runs')))
        for run in range(P0.get('runs')):
            _, _, _, C, _ = GAN.train_GAN(P0, DL_L, DL_U_iter, DL_V, name=P0.get('name')+'_%d'%run)
            C.mode_eval()
            mat_acc[run] = np.mean([GAN.get_accuracy(C(XV),YV) for XV, YV in DL_V])
            
        acc = np.mean(mat_acc)
        P.log(f"Perf: {acc:.5f} - Checked Params: "+", ".join([str(key)+' = '+str(val) for key,val in args.items()]),name='hyperopt')
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
        
        save_trials(P,trials)

    
def get_Results(P,P_val=None):
    P.log("Params: "+str(P))
    
    if P.get('CUDA') and torch.cuda.is_available():
        P.log("CUDA Training.")
    else:
        P.log("CPU Training.")
    
    DL_L, DL_U_iter, DL_V = pp.get_all_dataloader(P, get_Data(P))
    
    if P_val is not None:
        _, _, DL_V = pp.get_all_dataloader(P_val, get_Data(P_val))
    
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
        
        C.mode_eval()
        if P.get('R_active'):
            R.mode_eval()
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
    
    perf = np.zeros_like(acc_C)
    perf[0] = 0.0
    perf[1:] = (acc_C[1:]-acc_R[1:])/acc_R[1:]

    ax.plot(timeline,perf+1,c=colors[4],linestyle='solid')
    legend.append("Performance $P_C$")
    
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

    
   
def main():
    
    param_space= {
        'batch_size'      : scope.int(hp.qloguniform('batch_size', np.log(16), np.log(512), q=1)),
        
        'GLR'             : hp.loguniform('GLR', np.log(0.00001), np.log(0.1)),
        'GB1'             : hp.loguniform('GB1', np.log(0.001), np.log(0.99)),
        'DLR'             : hp.loguniform('DLR', np.log(0.00001), np.log(0.1)),
        'DB1'             : hp.loguniform('DB1', np.log(0.001), np.log(0.99)),
        'CLR'             : hp.loguniform('CLR', np.log(0.00001), np.log(0.1)),
        'CB1'             : hp.loguniform('CB1', np.log(0.001), np.log(0.99)),
        
        'G_ac_func'       : hp.choice('G_ac_func',['relu','leaky','leaky20','sig']),
        'G_hidden'        : scope.int(hp.qloguniform('G_hidden', np.log(16), np.log(1024), q=1)),
        'G_hidden_no'     : scope.int(hp.quniform('G_hidden_no', 0, 4, q=1)), 
        'G_optim'         : hp.choice('G_optim',['AdamW','SGD']),
        
        'D_ac_func'       : hp.choice('D_ac_func',['relu','leaky','leaky20','sig']),
        'D_hidden'        : scope.int(hp.qloguniform('D_hidden', np.log(16), np.log(1024), q=1)),
        'D_hidden_no'     : scope.int(hp.quniform('D_hidden_no', 0, 4, q=1)), 
        'D_optim'         : hp.choice('D_optim',['AdamW','SGD']),
        
        'C_ac_func'       : hp.choice('C_ac_func',['relu','leaky','leaky20','sig']),
        #'C_aco_func'      : hp.choice('C_aco_func',['gumbel','hardmax','softmax']),
        'C_hidden'        : scope.int(hp.qloguniform('C_hidden', np.log(16), np.log(1024), q=1)),
        'C_hidden_no'     : scope.int(hp.quniform('C_hidden_no', 0, 4, q=1)), 
        'C_optim'         : hp.choice('C_optim',['AdamW','SGD']),
        'C_tau'           : hp.loguniform('C_tau', np.log(0.01), np.log(10.)),
    }
    
    P_search = Params(
        name = 'Hyper_GAN',
        dataset = 'SHL',
        CUDA = False,
        
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
        print_epoch = False,
        epochs = 1,
        save_step = 1,
        runs = 1,
        
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
        name = 'eval_softmax',
        dataset = 'SHL',
        CUDA = False,
        
        print_epoch = False,
        epochs = 500,
        save_step = 10,
        runs = 5,
        
        FX_sel = 'all',
        Cross_val = 'none',
        
        C_basic_train = False,
        
        sample_no = 800,
        undersampling = False,
        oversampling = False,
        
        User_L = 1,
        User_U = 2,
        User_V = 3,
        
        CB1 = 0.010305926728118187, 
        CLR = 0.0017731978111430147, 
        C_ac_func = 'sig', 
        C_hidden = 254, 
        C_hidden_no = 2, 
        C_optim = 'AdamW', 
        C_tau = 5.576378501612518, 
        
        DB1 = 0.031153071611443442, 
        DLR = 0.013359442807976967, 
        D_ac_func = 'leaky', 
        D_hidden = 757, 
        D_hidden_no = 1, 
        D_optim = 'SGD', 
        
        GB1 = 0.9293227790745483, 
        GLR = 0.003310001178590957, 
        G_ac_func = 'relu', 
        G_hidden = 158, 
        G_hidden_no = 3, 
        G_optim = 'SGD', 
        
        batch_size = 21
        ) 
    
    #hyperopt_Search(P_test,param_space,eval_step=2,max_evals=5)
    # evaluate(P_test)
    
    # P_val = P.copy()
    # P_val.set_keys(
    #     sample_no = None,
    #     undersampling = False,
    #     oversampling = False,
    #     )
    
    # evaluate(P,P_val)
    # P.set_keys(
    #     name = 'eval_gumbel',
    #     C_aco_func = 'gumbel',
    #     )
    # evaluate(P,P_val)
    # P.set_keys(
    #     name = 'eval_hardmax',
    #     C_aco_func = 'hardmax',
    #     )
    # evaluate(P,P_val)
    
    
    hyperopt_Search(P_search,param_space)
    
    #sklearn_baseline(P)
    #pytorch_baseline(P)
    #hyperopt_C(eval_step=25,max_evals=None)
    
    
if __name__ == "__main__":
    main()
