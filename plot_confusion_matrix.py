# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sn

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from params import save_fig
else:
    # uses current package visibility
    from .params import save_fig

def plot_confusion_matrix(cm,P,labels=None,name='C',title='Confusion Matrix'):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    """
    if labels is None:
        labels = P.get('labels')
    
    fig, ax = plt.subplots()
    ax = sn.heatmap(cm, annot=True,fmt='d',xticklabels=labels, yticklabels=labels,cmap = sn.cubehelix_palette(len(labels))) # , fmt='.1%', vmin=0, vmax=1
    
    #cbar = ax.collections[0].colorbar
    #cbar.set_ticks([0, .5, 1])
    #cbar.set_ticklabels(['0%', '50%', '100%'])
    
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    
    #fig.tight_layout()
    save_fig(P,'con_mat_'+name,fig,close=True)