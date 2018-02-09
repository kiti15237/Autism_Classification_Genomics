import numpy as np
import pandas as pd
from random import shuffle
from collections import defaultdict

"""
RA, 10/27/2017
class_SibKFold.py
V1
=============================================================
Defines a basic SibKFold object:
Given a dataframe of inputs, X, and a number of folds, k,
Returns indicies of a random partition of X into k subsets
so that siblings are always in the same subset.
=============================================================
"""

# TODO:
# - [ ] integrate with EvalLR so that I don't have to read in ped file every time

"""
    SibKfold Object:
    Partitions the data into k subsets by sibling
    Class variables:
        - X (pandas df) Input features for all examples
        - y(1D pandas df) 0/1 labels for all examples
        - model_scores (pandas df) stores train and test scores
        - lr (sk linear model)
        - Xtrain, Xtest (np arrays) store train and test data
        - ytrain, ytrain (np arrays) store train and test labels
"""
class SibKFold(object):
    def __init__(self, n_splits, X, shuffle = True):
        self.shuffle = shuffle
        self.k = n_splits
        self.X = X
        self.siblings = self.initSibs()
        
    def initSibs(self):
        subjects = self.X.index.values.tolist()
        
        # read ssc ped 
        ssc_ped = pd.read_csv("data/ssc.ped", 
                              sep = "\t",
                             header = None, 
                             usecols = [0,1])
        ssc_ped.columns = ['fid','sibid']
        ssc_ped = ssc_ped.loc[ssc_ped['sibid'].isin(subjects)]
        
        # read Agre ped
        agre_ped = pd.read_csv("data/v34.vcf.ped", 
                              sep = "\t", 
                             usecols = [0,1])
        agre_ped.columns = ['fid','sibid']
        agre_ped = agre_ped.loc[agre_ped['sibid'].isin(subjects)]
        
        ped = pd.concat([ssc_ped, agre_ped], axis = 0, ignore_index = True)
        
        # transcribe into dictionary
        sibs = defaultdict(list)
        for i in range(ped.shape[0]):
            famid = str(ped.fid[i])
            sibs[famid].append(subjects.index(ped.sibid[i]))
            
        return sibs
    
    def split2(self, X):
        group_kfold = GroupKFold(n_splits= 6)
        group_kfold.get_n_splits(X, y, groups)

    def printSibs(self):
        print(self.siblings)

    def split(self, X):
        
        fam_ids = list(self.siblings.keys())
        
        if self.shuffle:
            shuffle(fam_ids)
        
        # init lists of indicies for folds
        fold_sizes = np.zeros(self.k)
        fold_inds = defaultdict(list)
       
        # populate fold_inds, a dict of indicies of each partition
        for fam_id in fam_ids:
            
            # find smallest fold
            smallest_fold = np.argmin(fold_sizes)
            fold_inds[smallest_fold] += self.siblings[fam_id]
            
            fold_sizes[smallest_fold] += len(self.siblings[fam_id])
            
        # format output as a list of tuples of train/test data for each fold
        result = []
        for f in range(self.k):
            # make list of all indicies not in the fth fold
            train_folds = list(range(self.k))
            train_folds = [i for i in train_folds if i!=f]
            train_inds = sum([fold_inds[i] for i in train_folds], [])
            
            # apend train, test to result
            result.append((train_inds,fold_inds[f]))
            
        return result