# -*- coding: utf-8 -*-
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

if __package__ is None or __package__ == '':
    import data_source as ds
    from params import Params, save_fig
    import preprocessing as pp
else:
    from . import data_source as ds
    from .params import Params, save_fig
    from . import preprocessing as pp


def sklearn_baseline(P):
    P.log(P)
        
    F = pp.perform_preprocessing(P, ds.get_data(P))
    
    x_train, y_train = F[0]
    x_test, y_test = F[2]
    y_train, y_test = y_train.ravel(), y_test.ravel()
    
    P.log('cross_val: '+str(P.get('cross_val')))
    P.log('   FX_num: '+str(P.get('FX_num')))
    
    ''' Multi-layer Perceptron '''
    mlp = MLP(hidden_layer_sizes=(100,100),max_iter=500)
    mlp.fit(x_train, y_train)

    y_pred = mlp.predict(x_train) 
    P.log(f"MLP Acc Train: {accuracy_score(y_train,y_pred):.2f}")
    P.log(f"MLP  F1 Train: {f1_score(y_train,y_pred,average='weighted'):.2f}")
    
    y_pred = mlp.predict(x_test)
    P.log(f"MLP Acc  Test: {accuracy_score(y_test,y_pred):.2f}")
    P.log(f"MLP  F1  Test: {f1_score(y_test,y_pred,average='weighted'):.2f}")
    P.log(F"MLP Iterations = {mlp.n_iter_}")
    
    ''' Random Forest Classifier '''
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    
    y_pred = rfc.predict(x_train)
    P.log(f"RFC Acc Train: {accuracy_score(y_train,y_pred):.2f}")
    P.log(f"RFC  F1 Train: {f1_score(y_train,y_pred,average='weighted'):.2f}")
    
    y_pred = rfc.predict(x_test)
    P.log(f"RFC Acc  Test: {accuracy_score(y_test,y_pred):.2f}")
    P.log(f"RFC  F1  Test: {f1_score(y_test,y_pred,average='weighted'):.2f}")


def plt_FX_num(P,max_n=908,P_val=None,indeces=None):
    
    if indeces is not None:max_n = min(max_n,len(indeces))
    
    P.set('FX_indeces',None)
    V = ds.get_data(P)
    
    for i,(X,Y) in enumerate(V):
        count_string = ', '.join([ f"{int(val)}: {count}" for val,count in zip(*np.unique(Y,return_counts=True))])
        P.log(f"V[{i+1}]: {X.shape} - {Y.shape} ({count_string})")
    
    def train_model(X,y,seed):
        mlp = MLP(hidden_layer_sizes=(150,150),max_iter=2000, random_state=seed)
        return mlp.fit(X, y)
    
    mat = np.empty(shape=(3,max_n))
    FX = np.arange(1,max_n+1,1)
    for fx in FX:
        if indeces is None:P.set('FX_num',fx)
        else: P.set('FX_indeces',indeces[:fx])

        V0 = ds.select_features(V,P.get('FX_indeces'))
        F0 = pp.perform_preprocessing(P, V0, P_val)
        
        x_train, y_train = F0[0]
        x_test, y_test = F0[2]

        model_list = Parallel(n_jobs=8)(delayed(train_model)(x_train, y_train.ravel(), seed) for seed in range(P.get('runs')))

        res = np.empty(shape=(3,P.get('runs')))
        for run,mlp in enumerate(model_list):
            res[0,run] = accuracy_score(y_test.ravel(),mlp.predict(x_test))
            res[1,run] = f1_score(y_test.ravel(),mlp.predict(x_test),average='weighted')
            res[2,run] = mlp.n_iter_
    
        mat[:,fx-1] = np.mean(res,axis=1)
        P.log(f"Fx_num = {fx}: [Acc = {mat[0,fx-1]:.2f}] [F1 = {mat[1,fx-1]:.2f}] [{mat[2,fx-1]:.2f} iterations]")
        
    plt.figure(figsize=(27,9),dpi=300,clear=True)
    fig, ax = plt.subplots()
    
    ax.plot(FX,mat[0],linestyle='solid',label='Accuracy')
    ax.plot(FX,mat[1],linestyle='solid',label='F1 Score')
    
    ax.legend()
    ax.set_xlabel('FX_num')
    ax.set_ylabel('Performance')
    
    ax.set_xlim(1,max_n)
    ax.grid()
    save_fig(P,'eval_fx_num',fig)
    
    ax.plot(FX,mat[2]/np.max(mat[2]),linestyle='solid',label='Iterations')
    ax.legend()
    save_fig(P,'eval_fx_num_iterations',fig)
    
if __name__ == "__main__":
    fx_num = 5
    P_fx_num = Params( name='fx_num', dataset='SHL_ext', sample_no=512, undersampling=False, oversampling=False, )
    plt_FX_num(P_fx_num,max_n=fx_num,P_val=P_fx_num.copy().set_keys(sample_no=None, undersampling=False, oversampling=False,))