# -*- coding: utf-8 -*-
"""
@author: George Kafentzis
Hyfe Inc.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import seaborn as sns
import numpy as np
from sklearn import metrics
import itertools

def plot_confusion_matrix(y_true, y_pred, classes, cn='', normalize=True, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(5,5))
    cm = metrics.confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j],2), fontsize="x-large", horizontalalignment="center", color="black" if cm[i, j] > thresh else "black")
    plt.ylabel('Real class')
    plt.xlabel('Predicted class')
    plt.title(cn)
    plt.tight_layout()
    plt.show()

def plot_ROC(ytest, score_test, cn = ''):
    plt.figure(figsize=(5,5))   
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    fprs, tprs, thresholds = metrics.roc_curve(ytest, score_test)
    AUC = metrics.auc(fprs, tprs)
    plt.plot(fprs, tprs, color='k', label=r'ROC (AUC = %0.3f)' % (AUC), lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title(cn)
    plt.grid()
    plt.tight_layout()
    
    plt.show()

def plot_histogram(ytest, score_test, name_clases, cn=''):
    plt.figure(figsize=(5,5))
    p1=np.where(np.hstack(ytest)==0)[0]
    p2=np.where(np.hstack(ytest)==1)[0]
    sns.histplot(np.hstack(score_test)[p1], label = ['0'], kde=True, stat="density", common_norm=False)
    sns.histplot(np.hstack(score_test)[p2], label = ['1'], kde=True, color='k', stat="density", common_norm=False)
    plt.grid()
    plt.legend()
    plt.xlabel("Classification threshold")
    plt.ylabel('Normalized count')
    plt.title(cn)
    plt.tight_layout()
    plt.show()
    #plt.figure(figsize=(5,5))
    #sns.distplot(np.hstack(score_test)[p1], label=name_clases[0])
    #sns.distplot(np.hstack(score_test)[p2], label=name_clases[1], color="k")
    #plt.grid()
    #plt.legend()
    #plt.xlabel("Classification threshold")
    #plt.ylabel('Normalized count')
    #plt.tight_layout()
    #plt.show()
    
    


def sensivity_specifity_cutoff(y_true, y_score):
    '''Find data-driven cut-off for classification
    
    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
    
    Parameters
    ----------
    
    y_true : array, shape = [n_samples]
        True binary labels.
        
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).
        
    References
    ----------
    
    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.
    
    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.
    
    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]