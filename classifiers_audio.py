# -*- coding: utf-8 -*-
"""
@author: George Kafentzis
Hyfe Inc.
"""
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier



def classifyNN(X, y, groups, opt):
    
    cv_performance = False
    
    Xtrain = X
    ytrain = y
    
    #
    # define grid search
    parameters = {'NN__alpha': (10.0 ** -np.arange(0, 7)).tolist()}

    # define pipeline
    nn_pip = Pipeline([('NN', MLPClassifier(solver='adam', hidden_layer_sizes=(1024, 512, 128, 32), max_iter = 1000, random_state=1))])
    
    # Set up inner CV procedure 
    cvGS = StratifiedGroupKFold(n_splits=5)

    
    print('Starting parameter tuning...\n')
    # Hyper parameter tuning
    clf = GridSearchCV(nn_pip, parameters, cv=cvGS, verbose=1, scoring='f1_weighted', refit = True)

    # prepare the outer CV procedure
    cvK = StratifiedGroupKFold(n_splits=3)

    # train the LR with the optimal hyper-parameters
    scoring = {"AUC": "roc_auc", 
               "BAcc": "balanced_accuracy", 
               "F1": "f1_weighted", 
               'Prec': 'precision_weighted', 
               'Rec': 'recall_weighted',
               'sensitivity': metrics.make_scorer(metrics.recall_score),
               'specificity': metrics.make_scorer(metrics.recall_score,pos_label=0)}
    
    scores = []
    
    if cv_performance == True:
        print('Starting Crossvalidation\n')
        scores = cross_validate(clf, Xtrain, ytrain, groups=groups, scoring=scoring, cv=cvK, n_jobs=-1)
    # report performance
        print('ROC-AUC: %.3f (%.3f)' % (np.mean(scores['test_AUC']), np.std(scores['test_AUC'])))
        print('Balanced Accuracy: %.3f (%.3f)' % (np.mean(scores['test_BAcc']), np.std(scores['test_BAcc'])))
        print('Precision (weighted): %.3f (%.3f)' %  (np.mean(scores['test_Prec']), np.std(scores['test_Prec'])))
        print('Recall (weighted): %.3f (%.3f)' % (np.mean(scores['test_Rec']), np.std(scores['test_Rec'])))
        print('F1-score: %.3f (%.3f)' % (np.mean(scores['test_F1']), np.std(scores['test_F1'])))
        print('Sensitivity: %.3f (%.3f)' % (np.mean(scores['test_sensitivity']), np.std(scores['test_sensitivity'])))
        print('Specificity: %.3f (%.3f)' % (np.mean(scores['test_specificity']), np.std(scores['test_specificity'])))
    
    # Fit the whole
    clf.fit(Xtrain, ytrain, groups=groups) # train the LR pipeline
    
    return scores, clf.best_estimator_





def classifyLR(X, y, groups, opt):
    
    cv_performance = False
    
    Xtrain = X
    ytrain = y
    
    #
    # define grid search
    parameters = {'LR__solver':['lbfgs'],
                  'LR__penalty': ['l2'],
                  'LR__C': np.arange(0.001, 5, 1000).tolist()}

    # define pipeline
    lr_pip = Pipeline([('LR', LogisticRegression(class_weight='balanced', max_iter=10000))])
    
    # Set up inner CV procedure 
    cvGS = StratifiedGroupKFold(n_splits=5)

    
    print('Starting parameter tuning...\n')
    # Hyper parameter tuning
    clf = GridSearchCV(lr_pip, parameters, cv=cvGS, verbose=1, scoring='f1_weighted', refit = True)

    # prepare the outer CV procedure
    cvK = StratifiedGroupKFold(n_splits=3)

    # train the LR with the optimal hyper-parameters
    scoring = {"AUC": "roc_auc", 
               "BAcc": "balanced_accuracy", 
               "F1": "f1_weighted", 
               'Prec': 'precision_weighted', 
               'Rec': 'recall_weighted',
               'sensitivity': metrics.make_scorer(metrics.recall_score),
               'specificity': metrics.make_scorer(metrics.recall_score,pos_label=0)}
    
    scores = []
    
    if cv_performance == True:
        print('Starting Crossvalidation\n')
        scores = cross_validate(clf, Xtrain, ytrain, groups=groups, scoring=scoring, cv=cvK, n_jobs=-1)
    # report performance
        print('ROC-AUC: %.3f (%.3f)' % (np.mean(scores['test_AUC']), np.std(scores['test_AUC'])))
        print('Balanced Accuracy: %.3f (%.3f)' % (np.mean(scores['test_BAcc']), np.std(scores['test_BAcc'])))
        print('Precision (weighted): %.3f (%.3f)' %  (np.mean(scores['test_Prec']), np.std(scores['test_Prec'])))
        print('Recall (weighted): %.3f (%.3f)' % (np.mean(scores['test_Rec']), np.std(scores['test_Rec'])))
        print('F1-score: %.3f (%.3f)' % (np.mean(scores['test_F1']), np.std(scores['test_F1'])))
        print('Sensitivity: %.3f (%.3f)' % (np.mean(scores['test_sensitivity']), np.std(scores['test_sensitivity'])))
        print('Specificity: %.3f (%.3f)' % (np.mean(scores['test_specificity']), np.std(scores['test_specificity'])))
    
    # Fit the whole
    clf.fit(Xtrain, ytrain, groups=groups) # train the LR pipeline

    return scores, clf.best_estimator_


def classifySVM(X, y, groups, opt):
    
    cv_performance = False
    
    Xtrain = X
    ytrain = y

    # randomized search cross-validation to optimize hyper-parameters of SVM 
    parameters = {'SVM__C': [0.01, 0.1, 1, 2, 3, 4, 5], 
                  'SVM__gamma': [0.01, 0.001, 0.0001, 0.00001]}

    
    # define pipeline
    svm_pip = Pipeline([('SVM', svm.SVC(kernel='rbf', class_weight='balanced', probability=True))])

    cvGS = StratifiedGroupKFold(n_splits=5)
    
    clf = GridSearchCV(svm_pip, parameters, cv=cvGS, verbose=1, scoring='f1_weighted', refit = True)


    # prepare the cross-validation procedure
    cvK = StratifiedGroupKFold(n_splits=3)

    # train the SVM with the optimal hyper-parameters
    #model = svm.SVC(kernel='rbf', C=Copt, gamma=gammaopt, class_weight='balanced') 
    scoring = {"AUC": "roc_auc", 
               "BAcc": "balanced_accuracy", 
               "F1": "f1_weighted", 
               'Prec': 'precision_weighted', 
               'Rec': 'recall_weighted',
               'sensitivity': metrics.make_scorer(metrics.recall_score),
               'specificity': metrics.make_scorer(metrics.recall_score,pos_label=0)}
    scores = []
    if cv_performance == True:
        #scores = []
        scores = cross_validate(clf, Xtrain, ytrain, groups=groups, scoring=scoring, cv=cvK, n_jobs=-1)
    
        # report performance
        print('ROC-AUC: %.3f (%.3f)' % (np.mean(scores['test_AUC']), np.std(scores['test_AUC'])))
        print('Balanced Accuracy: %.3f (%.3f)' % (np.mean(scores['test_BAcc']), np.std(scores['test_BAcc'])))
        print('Precision (weighted): %.3f (%.3f)' %  (np.mean(scores['test_Prec']), np.std(scores['test_Prec'])))
        print('Recall (weighted): %.3f (%.3f)' % (np.mean(scores['test_Rec']), np.std(scores['test_Rec'])))
        print('F1-score: %.3f (%.3f)' % (np.mean(scores['test_F1']), np.std(scores['test_F1'])))
        print('Sensitivity: %.3f (%.3f)' % (np.mean(scores['test_sensitivity']), np.std(scores['test_sensitivity'])))
        print('Specificity: %.3f (%.3f)' % (np.mean(scores['test_specificity']), np.std(scores['test_specificity'])))
    
    # Fit the whole
    clf.fit(Xtrain, ytrain, groups=groups) # train the SVM
    
    return scores, clf.best_estimator_