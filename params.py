# -*- coding: utf-8 -*-

# -------------------
#  Path Handling
# -------------------

import os

P_PATH = 'pic/'
M_PATH = 'models/'
S_PATH = 'prediction/'
T_PATH = 'tree/'
H_PATH = 'hyper/'

def make_dir_pic():os.makedirs(P_PATH, exist_ok=True)
def make_dir_mod():os.makedirs(M_PATH, exist_ok=True)
def make_dir_pre():os.makedirs(S_PATH, exist_ok=True)
def make_dir_tre():os.makedirs(T_PATH, exist_ok=True)
def make_dir_hyp():os.makedirs(H_PATH, exist_ok=True)


def save_file(file,path):
    with open(path, 'wb') as f:
        pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)

def load_file(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

# -------------------
#  Images
# -------------------

import matplotlib as mpl
import matplotlib.pyplot as plt

# File format for vector graphics
FILE_FORMAT_V = '.pdf'

# File format for pixel graphics
FILE_FORMAT_P = '.png'

def save_fig(P,name,fig,close=False):
    make_dir_pic()
    if P.get('name')[0] == '/':
        path = P_PATH + P.get('name') + name
    else:
        path = P_PATH + P.get('name') + '_' + name
    os.makedirs(path.rsplit('/', 1)[0], exist_ok=True)
    plt.rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    fig.savefig( path+FILE_FORMAT_V, dpi=300 )
    fig.savefig( path+FILE_FORMAT_P, dpi=300 )
    mpl.rcParams.update(mpl.rcParamsDefault)
    if close:
        plt.close(fig)


# -------------------
#  Decision Trees
# -------------------

def save_tree(P,name,tree):
    from sklearn.tree import export_text
    make_dir_tre()
    text_representation = export_text(tree)
    with open("tree/%s.log"%name, "w+") as fout:
        fout.write(text_representation)

# -------------------
#  Hyperopt Trials
# -------------------

def save_trials(P,trials,name=None):
    make_dir_hyp()
    if name is None:
        name = P.get('name')
    PATH = H_PATH+name+'.p'
    save_file(file=trials,path=PATH)
    
def load_trials(P,name=None):
    if name is None:
        name = P.get('name')  
    return load_file(H_PATH+name+'.p')
     
# -------------------
#  Parameters
# -------------------

import copy
import hashlib
import pickle

if __package__ is None or __package__ == '':
    # uses current directory visibility
    import data_source as ds
    from log import log as writeLog
    from sliding_window import get_FX_list, get_FX_list_len, get_best_n_features
else:
    # uses current package visibility
    from . import data_source as ds
    from .log import log as writeLog
    from .sliding_window import get_FX_list, get_FX_list_len, get_best_n_features


DEFAULT_PARAMS = {
        'name'            : "Missing_Name",             # Name to save files under
        'log_name'        : 'log',                      # Name of the logfile
        'dummy_size'      : 100000,                     # Length of the dummy signal (only for dataset 'Short')
        'noise'           : 0.0,                        # Standard deviation of gaussian noise added to the signals
        'data_path'       : '/media/lgunthermann/HDD/data',                  # Path to the datasets
        'CUDA'            : True,                       # True: utilises CUDA if available
        
        'dataset'         : "SHL_ext",                  # Name of the dataset to be used
        'location'        : 'Hips',                     # Body location of the sensor (Hand,Hips,Bag,Torso)
        'labels'          : None,                       # Class labels
        'channels'        : 'acc',                      # Sensor channels to be selected
        'magnitude'       : True,                       # True: Calculates the magnitude of acceleration
        
        'cross_val'       : 'user',                     # Crossvalidation mode, 'user' = as set in the individual users, 'user_x': days for user x seperatly, 'none': all data together
        'User_L'          : 1,                          # User for the Labelled data
        'User_U'          : 2,                          # User for the Unlabelled data
        'User_V'          : 3,                          # User for the Validation data
        
        'FX_sel'          : 'all',                      # Which features to extract
        'FX_indeces'      : None,                       # Which features to select after extraction. None = all
        'FX_num'          : None,                       # If given, select the n best features (overwrites 'FX_indeces')
        
        'padding'         : 'zerol',                    # Padding type for the sliding window
        'winsize'         : 500,                        # Size of the sliding window
        'jumpsize'        : 500,                        # Jump range of the sliding window
        
        'print_epoch'     : False,                      # True: Print individual epochs
        'save_GAN'        : False,                      # True: Save the network models
        'pretrain'        : None,                       # If given: Name of the pretrained model to be loaded.
        'runs'            : 10,                         # Number of runs
        
        'sample_no'       : None,                       # Not None: number of samples to reduce/increase all classes to
        'undersampling'   : False,                      # True: undersample all majority classes
        'oversampling'    : False,                      # True: oversample all minority classes  
        'PCA_n_components': None,                       # Number of components for PCA
        
        'epochs'          : 500,                        # Number of regular training epochs
        'epochs_GD'       : 0,                          # Number of G/D training epochs
        'epochs_GAN'      : 0,                          # Number of GAN training epochs
        'GD_ratio'        : 0.0,                        # If given, divides epochs into epochs_GD and epochs_GAN
        'save_step'       : 10,                         # Number of epochs after which results are stored
        'batch_size'      : 512,                        # Number of samples per batch
        'noise_shape'     : 100,                        # Size of random noise Z
        
        'G_label_sample'  : True,                       # True: randomly sample input labels for G | False: use current sample batch as input
        'G_label_factor'  : 1,                          # Size factor of the input for G in relation to current batch
        'C_basic_train'   : True,                       # True: The classifier is trained on real data | False: the classifier is only trained against the discriminator
        'R_active'        : True,                       # True: a reference classifier is used as baseline
        'D_fake_step'     : 1,                          # Every n epochs fake positive training will be performed on the discriminator
        
        'CLR'             : 0.003,                      # Classifier: Learning rate
        'CB1'             : 0.9,                        # Classifier: Decay rate for first moment estimates
        'CB2'             : 0.999,                      # Classifier: Decay rate for second-moment estimates
        'C_hidden'        : 256,                        # Classifier: Number of nodes in the hidden layers
        'C_hidden_no'     : 1,                          # Classifier: Number of hidden layers   
        'C_ac_func'       : 'relu',                     # Classifier: Type of activation function for the hidden layers
        'C_aco_func'      : 'gumbel',                   # Classifier: Type of activation function for the output layer
        'C_tau'           : 1,                          # Classifier: Temperature of gumbel softmax
        'C_optim'         : 'AdamW',                    # Classifier: Optimiser
        'C_drop'          : 0.0,                        # Classifier: Dropout probability for each hidden layer
        
        'DLR'             : 0.0125,                     # Discriminator: Learning rate
        'DB1'             : 0.75,                       # Discriminator: Decay rate for first moment estimates
        'DB2'             : 0.999,                      # Discriminator: Decay rate for second-moment estimates
        'D_hidden'        : 128,                        # Discriminator: Number of nodes in the hidden layers
        'D_hidden_no'     : 1,                          # Discriminator: Number of hidden layers     
        'D_ac_func'       : 'leaky',                    # Discriminator: Type of activation function for the hidden layers
        'D_aco_func'      : 'sig',                      # Discriminator: Type of activation function for the output layer
        'D_optim'         : 'SGD',                      # Discriminator: Optimiser
        'D_drop'          : 0.0,                        # Discriminator: Dropout probability for each hidden layer
        
        'GLR'             : 0.0005,                     # Generator: Learning rate
        'GB1'             : 0.5,                        # Generator: Decay rate for first moment estimates
        'GB2'             : 0.999,                      # Generator: Decay rate for second-moment estimates 
        'G_hidden'        : 128,                        # Generator: Number of nodes in the hidden layers
        'G_hidden_no'     : 1,                          # Generator: Number of hidden layers   
        'G_ac_func'       : 'leaky',                    # Generator: Type of activation function for the hidden layers
        'G_aco_func'      : 'tanh',                     # Generator: Type of activation function for the output layer    
        'G_optim'         : 'SGD',                      # Generator: Optimiser
        'G_drop'          : 0.0,                        # Generator: Dropout probability for each hidden layer
        
        'RLR'             : 0.003,                      # Baseline Classifier: Learning rate
        'RB1'             : 0.9,                        # Baseline Classifier: Decay rate for first moment estimates
        'RB2'             : 0.999,                      # Baseline Classifier: Decay rate for second-moment estimates
        'R_hidden'        : 256,                        # Baseline Classifier: Number of nodes in the hidden layers
        'R_hidden_no'     : 1,                          # Baseline Classifier: Number of hidden layers   
        'R_ac_func'       : 'relu',                     # Baseline Classifier: Type of activation function for the hidden layers
        'R_aco_func'      : 'softmax',                  # Baseline Classifier: Type of activation function for the output layer
        'R_tau'           : 1,                          # Baseline Classifier: Temperature of gumbel softmax
        'R_optim'         : 'AdamW',                    # Baseline Classifier: Optimiser
        'R_drop'          : 0.0,                        # Baseline Classifier: Dropout probability for each hidden layer
        }

class Params:
    def __init__(self, P=None, **kwargs):
        if P is None:
            self.params = DEFAULT_PARAMS
            given = locals()['kwargs']
            saved = load_params(given.get('name','missingNo'))
            
            if saved is None:
                saved = DEFAULT_PARAMS

            for key in DEFAULT_PARAMS:
                val = given.get(key,None)
                if val is None:
                    val = saved.get(key,None)
                    if val is None:
                        continue
                self.set(key,val)
            if self.get('labels') is None:
                self.set('labels',ds.get_labels())
        else:
            self.params = P

    def update_channels(self):
        if self.get('channels') == 'all':
            ch = ds.NAMES_X[1:]
            ch.remove('Pressure')
            self.set('channels',ch)
        elif self.get('channels') == 'acc':
            self.set('channels',ds.ACC_CHANNELS.copy())
        return self

    def get_channel_list(self):
        self.update_channels()
        param_lst = copy.deepcopy(self.params.get('channels'))
        if self.params.get('magnitude') and 'Magnitude' not in param_lst:
            param_lst.append('Magnitude')
            for chl in ds.ACC_CHANNELS[::-1]:
                if chl in param_lst:
                    param_lst.remove(chl)
        return param_lst
    
    def get_IO_shape(self):
        ''' Returns the input shape and number of output classes of a dataset '''
        FX_len = 908 if self.get('FX_indeces') is None else len(self.get('FX_indeces'))
        FX_len = min(FX_len,get_FX_list_len(get_FX_list(self)))
        X = len(self.get_channel_list()) * FX_len
        Y = len(self.get('labels'))
        return [X,Y]  

    def get_dataset_hash(self):
        keys = ['dataset','location','labels','noise','magnitude','winsize','jumpsize','padding']
        
        value = ''.join([str(self.get(key)) for key in keys])
        value += str(self.get_channel_list())
        value += str(get_FX_list(self))

        return hashlib.blake2s(value.encode('utf-8')).hexdigest()

    def get_dataset_hash_str(self):
        return str(self.get_dataset_hash())

    def log(self,txt:str,save:bool=True,error:bool=False,name:str=None):
        writeLog(txt,save=save,error=error,name=(self.get('name')+'_log'))

    def save(self):
        make_dir_mod()
        PATH = M_PATH + self.params['name'] + '_params.pkl'
        save_file(file=self.params,path=PATH)
            
    def get(self, key):
        return self.params.get(key,None)
    
    def set(self,key,val):
        assert key in DEFAULT_PARAMS.keys()
        self.params[key] = val
        
        if key=='channels': self.update_channels()
        elif key=='FX_num' and val is not None: self.set('FX_indeces',get_best_n_features(val))
        elif key in ['GD_ratio','epochs'] and val is not None:
            epochs_GD = int(round(self.get('GD_ratio')*self.get('epochs')))
            epochs_GAN = self.get('epochs')-self.get('epochs_GD')
            self.params['epochs_GD'] = epochs_GD
            self.params['epochs_GAN'] = epochs_GAN
        elif key=='epochs_GD' and val is not None:
            self.set('GD_ratio',((self.get('epochs_GD'))/self.get('epochs')))
        elif key=='epochs_GAN' and val is not None:
            self.set('GD_ratio',(1-(self.get('epochs_GAN'))/self.get('epochs')))
        return self
    
    def set_keys(self,**kwargs):
        for key, val in locals()['kwargs'].items():
            self.set(key,val)
        return self
        
    def update(self, dic):
        for key, val in dic.items():
            self.set(key,val)
        return self
            
    def inc(self,key):
        self.params[key] += 1
        return self
    
    def copy(self):
        return Params(copy.deepcopy(self.params))
    
    def __str__(self):
        return str(self.params)
    
def load_params(name):
    return load_file(M_PATH + name + '_params.pkl')
    
if __name__ == "__main__":
    channels = 'all'
    channels = ["Acc X","Acc Y","Acc Z","Gyroscope X"]
    channels = 'acc'
    
    magnitude = False
    magnitude = True
    
    FX_sel = 'all'

    P = Params(channels=channels,magnitude=magnitude,FX_sel=FX_sel)
    # print(P.get_channel_list())
    # print(P.get_dataset_hash())
    # print(hex(P.get_dataset_hash()))
    print(P.get_dataset_hash_str())
    print(P.get_IO_shape())

