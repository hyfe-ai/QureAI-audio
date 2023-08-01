# -*- coding: utf-8 -*-
"""
@author: George Kafentzis
Hyfe Inc.
"""

import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from feats_extraction import *
from typing import NamedTuple
import h5py
from sklearn.model_selection import StratifiedGroupKFold
from sklearn import preprocessing
from plot_func import *
from sklearn.metrics import roc_auc_score
from classifiers_audio import *

class Options(NamedTuple):
    classifier: str = 'LR' #SVM, NN: choose your classifier
    verbose: bool = False
    deltas: bool = False
    compute_feats: bool = True
    featsel: str = 'Ftest'
    recreate_file: bool = True
    sr: float = 16000
    
opt = Options()


df = pd.read_csv('output.csv')
condition = pd.read_csv('medical.csv')
condition['Normal vs. Abnormal Finding'].replace(['Normal', 'Abnormal'], [0, 1], inplace=True)

# Keep coughs only
df2 = df[df['is_cough'] == True] 

condition_list = []
for idx in df2.index:
    uID = df2['user_id'][idx]
    Cond = condition[condition['Patient ID'] == uID]['Normal vs. Abnormal Finding']
    condition_list.append(int(Cond))
    
df_final = df2.assign(GT=condition_list)
df_final = df_final.reset_index(drop=True)

############### FEATURE EXTRACTION ################
isFile_feats = os.path.isfile('./feats.h5')

Feats = []
if isFile_feats == False or opt.compute_feats == True:
    Feats = []
    for waveID in tqdm(range(0, len(df_final)), desc='Computing features...'):
        file = df_final.wav_file[waveID]
        F, fn = feature_extraction(file, opt)
        Feats.append(F)
    
    h5f = h5py.File('./feats.h5', 'w')
    h5f.create_dataset('Feats', data=np.asarray(Feats, dtype='float32'))
    h5f.close()
    FeatsDF = pd.DataFrame(data=np.squeeze(Feats), columns=fn)
else:
    h5f = h5py.File('./feats.h5', 'r')
    Feats = h5f['Feats'][:]
    h5f.close()
    

X = np.squeeze(np.asarray(Feats))
y = np.asarray(df_final.GT.to_list())
groups = df_final['user_id']
le = preprocessing.LabelEncoder()
groups_enc = le.fit_transform(groups)
#groups_or = le.inverse_transform(groups_enc)

cv = StratifiedGroupKFold(n_splits=4)
dfclass = []

for i, (train_ix, test_ix) in enumerate(cv.split(X, y, groups=groups_enc)):
    # select rows
    Xtrain, Xtest = X[train_ix], X[test_ix]
    ytrain, ytest = y[train_ix], y[test_ix]
    groups_cv = groups_enc[train_ix]
    groups_test = groups_enc[test_ix]
    print(f"Fold {i}:")
    
    scaler = preprocessing.StandardScaler().fit(Xtrain) 
    Xtrain = scaler.transform(Xtrain) 
    Xtest = scaler.transform(Xtest)

    if opt.classifier == 'LR':
        scoresLR, model= classifyLR(Xtrain, ytrain, groups_cv, opt)
    elif opt.classifier == 'SVM':
        scoresSVM, model= classifySVM(Xtrain, ytrain, groups_cv, opt)
    elif opt.classifier == 'NN':
        scoreNN, model = classifyNN(Xtrain, ytrain, groups_cv, opt)
    
    ypred = model.predict(Xtest) # test predictions
    classname = 'XRay prediction'
    
    if opt.classifier == 'SVM':
        score_test = model.decision_function(Xtest)
        plot_ROC(ytest, score_test, cn=classname)
        Probs = model.predict_proba(Xtest) # get probabilities
    else:
        Probs = model.predict_proba(Xtest) # get probabilities
        plot_ROC(ytest, Probs[:,1], cn=classname)
        score_test = Probs[:, 1]
    
    
    # display the results
    plot_confusion_matrix(ytest, ypred, cn=classname, classes=["Normal", "Abnormal"], normalize=False)
    
    print("Original ROC area: {:0.3f}".format(roc_auc_score(ytest, score_test)))
    
    n_bootstraps = 2000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(ypred), len(ypred))
        if len(np.unique(ytest[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(ytest[indices], score_test[indices])
        bootstrapped_scores.append(score)
        #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Computing the lower and upper bound of the 95% confidence interval
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
        confidence_lower, confidence_upper))
    
    
    # PER PARTICIPANT #####################################################
    #######################################################################
    avg_part_prob = []
    binary_part_prob = []
    binary_part_truth = []
    # Participant based classification: collecting stuff
    for groupid in np.unique(groups_test):
        idd = (groups_test == groupid)
        part = groups_test[idd]
        part_test = ytest[groups_test == part[0]]
        part_pred = ypred[groups_test == part[0]]
        part_pred_prob = Probs[idd]
        PosProb = np.mean(part_pred_prob[:,1]) # Avg. Prob. being TB-positive
        avg_part_prob.append(PosProb) # Collect probabilities of TB-positive
        binary_part_prob.append(PosProb >= 0.5)
        binary_part_truth.append(part_test[0])
        
        
    binary_part_prob = np.asarray(binary_part_prob)
    binary_part_truth = np.asarray(binary_part_truth)
    avg_part_prob = np.asarray(avg_part_prob)
    
    # CI stuff
    plot_ROC(binary_part_truth, avg_part_prob, cn='Abnormal XRay prediction per participant')
    print("Per Participant ROC area: {:0.3f}".format(roc_auc_score(binary_part_truth, avg_part_prob)))

    n_bootstraps = 2000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(binary_part_prob), len(binary_part_prob))
        if len(np.unique(binary_part_truth[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(binary_part_truth[indices], avg_part_prob[indices])
        bootstrapped_scores.append(score)
        #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Computing the lower and upper bound of the 95% confidence interval
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
        confidence_lower, confidence_upper))