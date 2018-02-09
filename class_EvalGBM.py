import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import f1_score, make_scorer, roc_curve, accuracy_score, confusion_matrix, roc_auc_score
from class_SibKFold import SibKFold

"""
RA, 10/27/2017
class_EvalGBM.py
V2
=============================================================
Defines a basic EvalGBM object:
Given several input parameters, builds a gradient booted classifier
model, runs k-fold cross validation, and outputs the results
=============================================================
"""

# TODO:
# - [X] test
# - [X] test multiclass functionality
# - [ ] take additional parameters such as scorer, c, etc.
# - [ ] generalize to other classifier types (e.g. SVM)
# - [ ] add function to calculate wald statistic for all coefficients
# - [X] BUGFIX: evaluate() sometimes throws a weird formatting error
# - [X] implement ROC printer
# - [ ] add AU-ROC
# - [ ] add precision-recall
# - [ ] add class function(s) for setting train and test data manually

"""
    EvalLR Object:
    Builds and evaluates a logistic regression model
    Class variables:
        - X (pandas df) Input features for all examples
        - y(1D pandas df) 0/1 labels for all examples
        - model_scores (pandas df) stores train and test scores
        - lr (sk linear model)
        - Xtrain, Xtest (np arrays) store train and test data
        - ytrain, ytrain (np arrays) store train and test labels
"""

class EvalGBM(object):
    def __init__(self, X, y, n_estimators = 100, max_depth = 3, min_samples_leaf = 2, metric = "roc"):
        # basic class variables
        self.X = X
        self.y = y
        
        self.gbm = ensemble.GradientBoostingClassifier(n_estimators = n_estimators,
                                                      max_depth = max_depth,
                                                      min_samples_leaf = min_samples_leaf)
        # test and train data
        self.Xtrain = None
        self.ytrain = None
        self.Xtest = None
        self.ytest = None
        self.metric = metric

    # print the counts of labels in a label dataset
    def summarize_data(self, labels):
        print labels.ix[:,0].value_counts()
        
    def resample(self, X, y, verbose, oversample = True):
        # subset data by label
        y = pd.DataFrame(y)
        neg = X[y.values == 0]
        pos = X[y.values == 1]

        # resample X to 1:1 ratio
        n_subj = X.shape[0]
        
        if oversample:
            if verbose:
                print "Oversampling from negative controls to 1:1 balance"
            desired = pos.shape[0]
            pos_resamp = pos
        else:
            desired = n_subj/2
            pos_resamp = pos.sample(n = desired, replace = True)
            
        neg_resamp = neg.sample(n = desired, replace = True)

        X_resamp = pd.concat([neg_resamp, pos_resamp])

        # create new labels for resampled data
        neg_y = pd.Series(np.zeros(desired))
        neg_y = neg_y.reindex(neg_resamp.index.values.tolist(), fill_value = 0)

        pos_y = pd.Series(np.zeros(desired))
        pos_y = pos_y.reindex(pos_resamp.index.values.tolist(), fill_value = 1)

        y_resamp = pd.concat([neg_y, pos_y])

        return X_resamp, y_resamp

    # trains on self.Xtrain and self.ytrain
    # evaluates on self.Xtest and self.ytest
    # returns train and test error as a tuple
    def evaluate(self, makeROC, verbose):
        # fit to train data
        self.gbm.fit(self.Xtrain, self.ytrain)
        
        test_probs = self.gbm.predict_proba(self.Xtest)[:,1]
        train_probs = self.gbm.predict_proba(self.Xtrain)[:,1]

        if self.metric == 'roc':
            # calculate and store roc scores
            testscore = roc_auc_score(self.ytest, test_probs)
            trainscore = roc_auc_score(self.ytrain, train_probs)
            
        else:
            fpr, tpr, thresholds = roc_curve(self.ytrain, train_probs, pos_label = 1)
            f1s = [f1_score(self.ytrain, (train_probs>t).astype(int), average = 'binary') for t in thresholds]
            f_i = np.argmax(np.asarray(f1s))
            if verbose:
                print "\nOptimum threshold to maximize f1:", thresholds[f_i]
            trainscore = f1_score(self.ytrain, (train_probs>thresholds[f_i]).astype(int), average = 'binary')
            testscore = f1_score(self.ytest, (test_probs>thresholds[f_i]).astype(int), average = 'binary')
            
        
        if verbose:
            print "Train Score: %f Test Score: %f" % (trainscore, testscore)
            print self.printMetrics()
        
        if makeROC:
            self.showROC()
        
        return testscore, trainscore

    # runs k-fold cross validation by calling self.evaluate()
    # returns a matrix of model scores
    def kfold(self, k, makeROC = False, verbose = True, resample = False):

        # use KFold from sklearn to split data
        kf = SibKFold(n_splits= k, X = self.X, shuffle = True)

        # make model_scores dataframe to fill in
        model_scores = pd.DataFrame(index = range(1, k+1),
                                    columns = ["Train_score", "Test_score"])
        i = 1
        
        coef = np.zeros(self.X.shape[1])

        for train, test in kf.split(self.X):

            # assign test and train sets
            self.Xtrain, self.Xtest = self.X.iloc[train], self.X.iloc[test]
            self.ytrain, self.ytest = self.y.iloc[train], self.y.iloc[test]
            
            # resample if desired:
            if resample:
                self.Xtrain, self.ytrain = self.resample(self.Xtrain, self.ytrain, verbose)
                self.Xtest, self.ytest = self.resample(self.Xtest, self.ytest, verbose)

            # evaluate and store results
            if verbose:
                print "\nRunning cross validation for fold %d:" % i
                print "==========================================="
                
            testscore, trainscore = self.evaluate(makeROC, verbose)

            model_scores.at[i, 'Test_score'] = testscore
            model_scores.at[i, 'Train_score'] = trainscore

            i += 1

        return model_scores
    
    def printMetrics(self):
        test_probs = self.gbm.predict_proba(self.Xtest)
        train_probs = self.gbm.predict_proba(self.Xtrain)
        fpr, tpr, thresholds = roc_curve(self.ytrain, train_probs[:,1], pos_label = 1)
        
        # find threshold that maximizes accuracy
        accs = [accuracy_score(self.ytrain, (train_probs[:,1]>t).astype(int)) for t in thresholds]
        a_i = np.argmax(np.asarray(accs))
        print "AU-ROC", roc_auc_score(self.ytest, test_probs[:,1])

        print "\nOptimum threshold to maximize training accuracy:", thresholds[a_i]
        print "F1:", f1_score(self.ytest, (test_probs[:,1]>thresholds[a_i]).astype(int), average = 'binary')
        print "Accuracy:", accuracy_score(self.ytest, (test_probs[:,1]>thresholds[a_i]).astype(int))
        print "Confusion Matrix:\n", confusion_matrix(self.ytest, (test_probs[:,1]>thresholds[a_i]).astype(int))

        # find threshold that maximizes f1
        f1s = [f1_score(self.ytrain, (train_probs[:,1]>t).astype(int), average = 'binary') for t in thresholds]
        f_i = np.argmax(np.asarray(f1s))

        print "\nOptimum threshold to maximize f1:", thresholds[f_i]
        print "F1:", f1_score(self.ytest, (test_probs[:,1]>thresholds[f_i]).astype(int), average = 'binary')
        print "Accuracy:", accuracy_score(self.ytest, (test_probs[:,1]>thresholds[f_i]).astype(int))
        print "Confusion Matrix:\n", confusion_matrix(self.ytest, (test_probs[:,1]>thresholds[f_i]).astype(int))
    
    # makes an ROC plot and shows it
    def showROC(self):
        fpr, tpr, thresholds = self.getROCdata()
        plt.figure()
        lw = 2
        
        plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label = 'ROC curve')
            
        plt.plot([0,1], [0,1], color = 'navy', lw = lw, linestyle = '--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.legend(loc='lower right')
        plt.show()
        plt.figure(figsize = (10,10))
        
    def getROCdata(self):
        test_probs = self.gbm.predict_proba(self.Xtest)
        
        fpr, tpr, thresholds = roc_curve(self.ytest, test_probs[:,1], pos_label = 1)
                       
        return fpr, tpr, thresholds
    
    def setTrain(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        
    def setTest(self, Xtest, Ytest):
        self.Xtest = Xtest
        self.Ytest = Ytest
