# -*- coding: utf-8 -*-

# -------------------
#  Path Handling
# -------------------

import os

P_PATH = 'pic/'
M_PATH = 'models/'
S_PATH = 'prediction/'
T_PATH = 'tree/'

def make_dir_pic():os.makedirs(P_PATH, exist_ok=True)
def make_dir_mod():os.makedirs(M_PATH, exist_ok=True)
def make_dir_pre():os.makedirs(S_PATH, exist_ok=True)
def make_dir_tre():os.makedirs(T_PATH, exist_ok=True)

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
#  Parameters
# -------------------

import copy
import hashlib
import pickle

if __package__ is None or __package__ == '':
    # uses current directory visibility
    import data_source as ds
    from sliding_window import get_FX_list
else:
    # uses current package visibility
    from . import data_source as ds
    from .sliding_window import get_FX_list

DEFAULT_PARAMS = {
        'name'            : "Missing_Name",             # Name to save files under
        'log_name'        : 'log',                      # Name of the logfile
        'dummy_size'      : 100000,                     # Length of the dummy signal (only for dataset 'Short')
        'noise'           : 0.0,                        # Standard deviation of gaussian noise added to the signals
        
        'dataset'         : "SHL",                      # Name of the dataset to be used
        'location'        : 'Hips',                     # Body location of the sensor (Hand,Hips,Bag,Torso)
        'labels'          : None,                       # Class labels
        
        'channels'        : 'acc',                      # Sensor channels to be selected
        'magnitude'       : True,                       # True: Calculates the magnitude of acceleration
        'padding'         : 'zerol',                    # Padding type for the sliding window
        'FX_sel'          : 'basic',                    # Which features to extract
        'winsize'         : 500,                        # Size of the sliding window
        'jumpsize'        : 500,                        # Jump range of the sliding window
        
        'print_epoch'     : True,                       # True: Print individual epochs
        'save_GAN'        : False,                      # True: Save the network models
        'pretrain'        : None,                       # If given: Name of the pretrained model to be loaded.
        'runs'            : 10,                         # Number of runs
        
        'epochs'          : 500,                        # Number of training epochs
        'save_step'       : 10,                         # Number of epochs after which results are stored
        'oversampling'    : True,                       # True: oversample all minority classes
        'batch_size'      : 128,                        # Number of samples per batch
        'noise_shape'     : 100,                        # Size of random noise Z

        'G_no'            : 1,                          # Model number of the new generator
        'D_no'            : 1,                          # Model number of the new discriminator
        'C_no'            : 1,                          # Model number of the new classifier
        
        'G_label_sample'  : True,                       # True: randomly sample input labels for G | False: use current sample batch as input
        'G_label_factor'  : 1,                          # Size factor of the input for G in relation to current batch
        'C_basic_train'   : True,                       # True: The classifier is trained on real data | False: the classifier is only trained against the discriminator
        'R_active'        : True,                       # True: a reference classifier is used as baseline
        
        'GLR'             : 0.0005,                     # Generator learning rate
        'GB1'             : 0.5,                        # Generator decay rate for first moment estimates
        'GB2'             : 0.999,                      # Generator decay rate for second-moment estimates
        'DLR'             : 0.0125,                     # Discriminator learning rate
        'DB1'             : 0.75,                       # Discriminator decay rate for first moment estimates
        'DB2'             : 0.999,                      # Discriminator decay rate for second-moment estimates
        'CLR'             : 0.003,                      # Classifier learning rate
        'CB1'             : 0.9,                        # Classifier decay rate for first moment estimates
        'CB2'             : 0.999,                      # Classifier decay rate for second-moment estimates
        
        'G_hidden'        : 128,                        # Generator: Number of nodes in the hidden layers
        'G_ac_func'       : 'leaky',                    # Generator: Type of activation function for the hidden layers
        'G_aco_func'      : 'tanh',                     # Generator: Type of activation function for the output layer
        
        'D_hidden'        : 128,                        # Discriminator: Number of nodes in the hidden layers
        'D_ac_func'       : 'leaky',                    # Discriminator: Type of activation function for the hidden layers
        'D_aco_func'      : 'sig',                      # Discriminator: Type of activation function for the output layer
        
        'C_hidden'        : 256,                        # Classifier: Number of nodes in the hidden layers
        'C_ac_func'       : 'relu',                     # Classifier: Type of activation function for the hidden layers
        'C_aco_func'      : 'gumbel',                   # Classifier: Type of activation function for the output layer   
        }

class Params:
    def __init__(self, P=None, **kwargs):
        if P is None:
            given = locals()['kwargs']
            saved = load_params(given.get('name','missingNo'))
            
            if saved is None:
                saved = DEFAULT_PARAMS
                
            self.params = {}
            for key in DEFAULT_PARAMS:
                val = given.get(key,None)
                if val is None:
                    val = saved.get(key,None)
                    if val is None:
                        val = DEFAULT_PARAMS.get(key,None)
                self.set(key,val)
            if self.get('labels') is None:
                self.set('labels',ds.get_labels())
        else:
            self.params = P

    def update(self, dic):
        for key, val in dic.items():
            self.set(key,val)

    def update_channels(self):
        if self.get('channels') == 'all':
            ch = ds.NAMES_X[1:]
            ch.remove('Pressure')
            self.set('channels',ch)
        elif self.get('channels') == 'acc':
            self.set('channels',ds.ACC_CHANNELS.copy())

    def get_channel_list(self):
        self.update_channels()
        param_lst = self.params.get('channels')
        if self.params.get('magnitude') and 'Magnitude' not in param_lst:
            param_lst.append('Magnitude')
            for chl in ds.ACC_CHANNELS[::-1]:
                if chl in param_lst:
                    param_lst.remove(chl)
        return param_lst
    
    def get_IO_shape(self):
        ''' Returns the input shape and number of output classes of a dataset '''
        X = len(self.get_channel_list()) * len(get_FX_list(self))
        Y = len(self.get('labels'))
        return [X,Y]  

    def get_dataset_hash(self):
        keys = ['dataset','location','channels','labels','noise','magnitude','winsize','jumpsize','FX_sel','padding']
        
        value = ''.join([str(self.get(key)) for key in keys])

        return hashlib.blake2s(value.encode('utf-8')).hexdigest()

    def get_dataset_hash_str(self):
        return str(self.get_dataset_hash())

    def save(self):
        make_dir_mod()
        PATH = M_PATH + self.params['name'] + '_params.pkl'
        with open(PATH, 'wb') as f:
            pickle.dump(self.params, f, pickle.HIGHEST_PROTOCOL)
            
    def get(self, key):
        return self.params.get(key,None)
    
    def set(self,key,val):
        assert key in DEFAULT_PARAMS.keys()
        self.params[key] = val
        if key=='channels': self.update_channels()
    
    def inc(self,key):
        self.params[key] += 1
    
    def copy(self):
        return Params(copy.deepcopy(self.params))
    
    def __str__(self):
        return str(self.params)
    
def load_params(name):
    PATH = M_PATH + name + '_params.pkl'
    if not os.path.isfile(PATH):
        return None
    with open(PATH, 'rb') as f:
        return pickle.load(f)

    
if __name__ == "__main__":
    channels = 'all'
    channels = ["Acc X","Acc Y","Acc Z","Gyroscope X"]
    channels = 'acc'
    
    magnitude = False
    magnitude = True
    
    P = Params(channels=channels,magnitude=magnitude)
    # print(P.get_channel_list())
    # print(P.get_dataset_hash())
    # print(hex(P.get_dataset_hash()))
    print(P.get_dataset_hash_str())
    print(P.get_IO_shape())

