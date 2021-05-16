# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
import torch

    
if __package__ is None or __package__ == '':
    import data_source as ds
    import GAN
    from main import plot_evaluation
    from network import save_results, load_results
    from params import Params
    import preprocessing as pp
else:
    from . import data_source as ds
    from . import GAN
    from .main import plot_evaluation
    from .network import save_results, load_results
    from .params import Params
    from . import preprocessing as pp
    
def run_cross_val(P,V=None):
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
        
        F = pp.perform_preprocessing(P, ds.get_data(P,V), P.copy().set_keys( sample_no = None, undersampling = False, oversampling = False, ))
        
        X, Y = F[0]
        XV, YV = F[2]
        x_test, y_test = XV, YV.ravel()
        DL_V = pp.get_dataloader(P, XV, YV, batch_size=1024) 
        
        #DL_L, DL_U_iter, DL_V = pp.get_all_dataloader(P, ds.get_data(P,V), P_val)
        #P.log(f"Number of batches: Labelled = {len(DL_L)} | Unlabelled = {len(DL_U_iter)} | Validation = {len(DL_V)}")    
    
        ACC = None
        F1S = None
        YF = None
        RF = None
        PF = None
        
        # Baseline Results
        res = np.empty(shape=(P.get('runs'),8))
        
        # -------------------
        #  Individual runs
        # -------------------
        skf = StratifiedKFold(n_splits=P.get('runs'),shuffle=True,random_state=42)
        for run, (train_index, test_index) in enumerate(skf.split(X, Y)):
        
            DL_L = pp.get_dataloader(P, X[test_index], Y[test_index])
            DL_U_iter = pp.get_perm_dataloader(P, X[train_index], Y[train_index])    
            P.log(f"Number of batches: Labelled = {len(DL_L)} | Unlabelled = {len(DL_U_iter)} | Validation = {len(DL_V)}")
            
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
        
            # Baseline
            x_train, y_train = X[test_index], Y[test_index].ravel()
            
            ''' Random Forest Classifier '''
            clf = RandomForestClassifier()
            clf.fit(x_train, y_train)
            
            y_pred = clf.predict(x_train) 
            res[run,0] = accuracy_score(y_train,y_pred)
            res[run,1] = f1_score(y_train,y_pred,average='macro')
            
            y_pred = clf.predict(x_test)
            res[run,2] = accuracy_score(y_test,y_pred)
            res[run,3] = f1_score(y_test,y_pred,average='macro')

            
            ''' Gaussian Naive Bayes '''
            clf = GaussianNB()
            clf.fit(x_train, y_train)
            
            y_pred = clf.predict(x_train) 
            res[run,4] = accuracy_score(y_train,y_pred)
            res[run,5] = f1_score(y_train,y_pred,average='macro')
            
            y_pred = clf.predict(x_test)
            res[run,6] = accuracy_score(y_test,y_pred)
            res[run,7] = f1_score(y_test,y_pred,average='macro')
            
        
        save_results(P, ACC, name='acc')
        save_results(P, F1S, name='f1')
        save_results(P,YF,name='YF')
        save_results(P,PF,name='PF')
        if RF is not None:
            save_results(P, RF, name='RF')
        P.log("Saved Accuracy, F1 Score and predictions.")
        
        # Baseline Evaluation
    
        res = np.mean(res,axis=0)
           
        P.log("")
        P.log(f"RFC Acc Train: {res[0]:.5f}")
        P.log(f"RFC  F1 Train: {res[1]:.5f}")
        
        P.log("")
        P.log(f"RFC Acc  Test: {res[2]:.5f}")
        P.log(f"RFC  F1  Test: {res[3]:.5f}")
        
        P.log("")
        P.log(f"GNB Acc Train: {res[4]:.5f}")
        P.log(f"GNB  F1 Train: {res[5]:.5f}")
        
        P.log("")
        P.log(f"GNB Acc  Test: {res[6]:.5f}")
        P.log(f"GNB  F1  Test: {res[7]:.5f}")
        P.log("")
        
    else:
        P.log("Loaded Accuracy, F1 Score and predictions.")
        
        
    plot_evaluation(P, ACC, F1S, YF, PF, RF, epoch_lst=list(range(50,P.get('epochs'),50)))


if __name__ == "__main__":
    import argparse
    from params import DEFAULT_PARAMS as default
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, dest='data_path')
    parser.add_argument('-datapath', type=str, dest='data_path')
    parser.set_defaults(data_path=default['data_path'])
    args = parser.parse_args()
    
    P = Params(
        name = 'eval_cross_eval_user1_bs',
        data_path = args.data_path,

        epochs = 300,
        save_step = 1,
        runs = 10,
        
        dataset = 'SHL_ext',
        
        FX_sel = 'all',
        R_active = True,
        cross_val = 'combined',
        
        sample_no = 11136,
        undersampling = False,
        oversampling = False,
        
        User_L = 1,
        User_U = 2,
        User_V = 3,
        
        batch_size = 512,
        FX_num = 150, 
        
        GD_ratio = 0,
        
        RB1 = 0.8661148142428583, 
        RLR = 8.299645247840653e-05, 
        R_ac_func = 'leaky20', 
        R_hidden = 1790, 
        R_hidden_no = 2, 
        R_optim = 'AdamW', 
        
        CB1 = 0.8661148142428583, 
        CLR = 8.299645247840653e-05, 
        C_ac_func = 'leaky20', 
        C_hidden = 1790, 
        C_hidden_no = 2, 
        C_optim = 'AdamW', 
        C_tau = 2.833757972503762, 
        
        DB1 = 0.04397295845368007, 
        DLR = 0.0243252689035249, 
        D_ac_func = 'leaky', 
        D_hidden = 113, 
        D_hidden_no = 6, 
        
        GB1 = 0.6201555853224091, 
        GLR = 0.006959406242448824, 
        G_ac_func = 'relu', 
        G_hidden = 318, 
        G_hidden_no = 5,
        
        ) 

    #V = ds.load_data(P)
    run_cross_val(P)