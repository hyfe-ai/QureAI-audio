# -*- coding: utf-8 -*-
"""
@author: George Kafentzis
Hyfe Inc.
"""
import numpy as np
import librosa as lr
from scipy.stats import kurtosis, skew
from features import *
from itertools import product


eps = 10e-8

def feature_extraction(filename, opt):
# Feature extraction 
    
    #print('Performing feature extraction in filename', filename)
    X = np.array([])
    sig, fs = lr.load(filename, sr=opt.sr)
    
    win, step = 0.05, 0.025
    [f, fn] = fbf_feature_extraction(sig, fs, int(fs * win), int(fs * step), deltas=False)
    
    ds = []
    featb = []
    # Run librosa's deltas
    if opt.deltas == True:
        for idx in range(len(f)):
            feat = f[idx,:]
            featb.append(feat[~np.isnan(feat)])
            dfeat = lr.feature.delta(featb[idx])
            ds.append(dfeat)
            
    
    # Stack features and compute global statistics
    for idx in range(0, len(f)):
        X = np.hstack((X, get_stats(f[idx])))
        
    if opt.deltas == True:
        for idx in range(0, len(ds)):
            X = np.hstack((X, get_stats(ds[idx])))

    # Return the global stats matrix
    F = X.reshape(-1,1)
    
    names = ['_mean', '_std', '_med', '_skew', '_kurt']
    
    feat_names = [a + b for a, b in product(fn, names)]
        
    return F, feat_names


def get_stats(F):
    
    stats = [np.mean(F), np.std(F), np.median(F), skew(F), kurtosis(F)]
            #np.percentile(F, 1), np.percentile(F, 99), np.abs(np.percentile(F, 99) - np.percentile(F, 1))]
    
    return stats