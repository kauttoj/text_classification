# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:23:33 2017

http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html
"""

from mlxtend.classifier import StackingClassifier

import numpy as np
import re
from pandas import DataFrame
from time import time
import pickle
import matplotlib.pyplot as plt

from nltk import tokenize
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.decomposition import TruncatedSVD, PCA

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from itertools import tee, islice

from sklearn.base import BaseEstimator, TransformerMixin

from collections import defaultdict

from sklearn.neighbors import KNeighborsClassifier

import scipy
#from vowpalwabbit import pyvw
#from vowpalwabbit.sklearn_vw import VWClassifier

import os
import sys
#HOME=os.getcwd()
#os.chdir(HOME+'\\..')
#sys.path.insert(0,os.getcwd())
#os.chdir(HOME)
#import get_my_data

def median(arr,ind):                
    try:
        arr=sorted(arr)
        return arr[len(arr)//2]
    except:                    
        return arr[ind]
def average_params(params,ind):
    average_params = params[0].copy() 
    k=0                        
    for key in average_params.keys():
        k+=1
        #print('key',k,':',key)
        val=[]
        for a in params:
            val.append(a[key])            
        average_params[key]=[median(val,ind)]
    return average_params

def get_metrics(true_labels, predicted_labels):
    
    a = np.round(metrics.accuracy_score(true_labels,predicted_labels),3)
    print('Accuracy:',a)
    
    p = np.round(metrics.precision_score(true_labels,predicted_labels,average='weighted'),3) 
    print('Precision:',p)
    
    r = np.round(metrics.recall_score(true_labels,predicted_labels,average='weighted'),3)    
    print('Recall:', r)
    
    s = np.round(metrics.f1_score(true_labels,predicted_labels,average='weighted'),3)
    print('F1 Score:', s)
    
    cm = metrics.confusion_matrix(true_labels, predicted_labels)
    print(cm)
    
    return a,p,r,s,cm

#    """
#    MODEL
#    """
#    SVM_model = svm.SVC(kernel = 'linear')
#    Forest_model = RandomForestClassifier()
#    SGD_model = SGDClassifier(penalty='l2')
#    Bayes_model=MultinomialNB()
#    logreg_model = LogisticRegression()
#    
#    """
#    PARAMETERS
#    """
#    parameters = {
#        'vect__max_df': [0.75],
#        'vect__max_features': [40000],
#        'vect__ngram_range': [(1,2)],  # n-grams
#        'vect__stop_words': ['english'],
#        'tfidf__use_idf': [True],
#        'tfidf__norm': ['l2'],
#        #'SGD__alpha': (0.0001,0.00001, 0.000001),
#        #'SGD__penalty': ['l2'],
#        #'SGD__loss': ['hinge'],
#        #'SGD__n_iter': [100],
#        #'svm__C': numpy.logspace(-2,5,7).tolist(),
#        #'bayes__alpha': [0.75]
#        #'forest__n_estimators':[10,15,20,25]
#        'logreg__C': numpy.logspace(4,6,5).tolist(),
#    }
    
"""
PIPELINE
"""
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    

def show_most_informative_features(model, text=None, n=20):
    
    from operator import itemgetter
    """
    Accepts a Pipeline with a classifer and a TfidfVectorizer and computes
    the n most informative features of the model. If text is given, then will
    compute the most informative features for classifying that text.

    Note that this function will only work on linear models with coefs_
    """
    # Extract the vectorizer and the classifier from the pipeline
    vectorizer = model.named_steps['vectorizer']
    classifier = model.named_steps['classifier']

    # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {} model.".format(
                classifier.__class__.__name__
            )
        )

    if text is not None:
        # Compute the coefficients for the text
        tvec = model.transform([text]).toarray()
    else:
        # Otherwise simply use the coefficients
        tvec = classifier.coef_

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        key=itemgetter(0), reverse=True
    )

    topn  = zip(coefs[:n], coefs[:-(n+1):-1])

    # Create the output string to return
    output = []

    # If text, add the predicted value to the output.
    if text is not None:
        output.append("\"{}\"".format(text))
        output.append("Classified as: {}".format(model.predict([text])))
        output.append("")

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}{: >15}    {:0.4f}{: >15}".format(cp, fnp, cn, fnn)
        )

    return "\n".join(output)

# TruncatedSVD is for large sparse datasets which cannot be centered without making the memory usage explode.

class prepare_data(BaseEstimator, TransformerMixin):    
        
    def __init__(self, Params=None,n_components=100):
        self.Params = Params
        self.n_components=n_components
        self.decomposer = TruncatedSVD(n_components=n_components)
        #self.decomposer = NMF(n_components=n_components)
        self.tfidf = TfidfTransformer()
            
    def fit(self,X, y=None):
        dat = self.tfidf.fit_transform(X[:,0:-self.Params['n_custom_features']])   
        if self.Params['Compression']==1:
            self.decomposer.fit(dat)
        return self    
            
    def transform(self,X):                    
        dat = self.tfidf.transform(X[:,0:-self.Params['n_custom_features']])   
        if self.Params['Compression']==1:
            dat = self.decomposer.transform(dat)  
            dat = np.concatenate((dat,X[:,-self.Params['n_custom_features']:].todense()),axis=1)
        else:
            dat = scipy.sparse.hstack([dat,X[:,-self.Params['n_custom_features']:]])            
        return dat

    # def fit_transform(self,X, y=None):
    #     self.fit(X,y)
    #     return self.transform(X)


class prepare_data_embedded(BaseEstimator, TransformerMixin):
    def __init__(self, Params=None, Feat = None):
        self.Params = Params
        self.pipe = None
        self.n_components = None
        self.inds = None
        self.Feat = Feat

    def fit(self, X, y=None):
        self.inds = np.where(np.sum(X[:,0:-self.Params['n_custom_features']],axis=0)>0)[0]
        X = self.Feat['X_transformer'][self.inds, :]
        self.inds = self.inds.tolist()
        self.n_components = X.shape[1]-5
        self.pipe = Pipeline([('scaler', StandardScaler()), ('pca',PCA(n_components=self.n_components))])
        self.pipe.fit(X)
        return self

    def transform(self, X):
        XX = np.zeros((X.shape[0],self.n_components),dtype=np.float32)
        for row in range(XX.shape[0]):
            ind = X[row,0:-self.Params['n_custom_features']].nonzero()[0]
            x = 0
            k = 0
            for elem in ind:
                if elem in self.inds:
                    xx = self.Feat['X_transformer'][elem,:].reshape(1, -1)
                    x += xx*X[row,elem]
                    k += X[row,elem]

            x /= k
            x = self.pipe.transform(x)

            XX[row,0:self.n_components] = x
        XX = np.concatenate((XX, X[:, -self.Params['n_custom_features']:]), axis=1)
        return XX


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

# and a tf-idf version of the same

class TfidfEmbeddingVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, Params=None, Feat = None):
        self.Params = Params
        self.word2weight = None
        self.Feat = Feat
        self.wordlist = None
        self.pca = None

    def fit(self, X, y):
        self.dim = self.Params['embedding_dimension']
        tfidf = TfidfVectorizer(analyzer = lambda x: x)  # skip all word processing
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(lambda : max_idf,[(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        wordlist = []
        for words in X:
            for w in words:
                if w not in wordlist and self.Feat['word2vec'][w] is not None:
                    wordlist.append(w)
        self.wordlist = wordlist

        if self.Params['WordEmbedding_PCA']==1:
            #self.scaler = StandardScaler()
            self.pca = PCA(n_components = self.dim)
            X = [self.Feat['word2vec'][x].reshape(1, -1) for x in wordlist]
            X = np.concatenate(X,axis=0)
            #X = self.fit_transform(X)
            self.pca.fit(X)
        return self

    def transform(self, X):
        res = np.zeros((len(X),self.dim),dtype=np.float32)
        for i,words in enumerate(X):
            x = 0
            k = 0.0        
            for w in words:
                if w in self.wordlist:
                    x+=self.Feat['word2vec'][w]*self.word2weight[w]
                    k+=1.0
            if self.Params['WordEmbedding_PCA'] == 1:
                res[i,:] = self.pca.transform(x.reshape(1, -1) / k)
            else:
                res[i,:] = x/k
        return res

def main(FEAT=None,Params=None):
                   
    if Params['TargetType'] == 'regression':
        if Params['Algorithm'] == 'SGD':
            model = SGDClassifier(penalty='l2',loss='squared_loss')     
            parameters = {'penalty':['l1','l2']}
    else:
        if Params['Algorithm'] == 'SVM':
            model = svm.SVC(kernel = 'linear',C = 0.01)
        if Params['Algorithm'] == 'NaiveBayes':
             model = MultinomialNB(alpha = 1.0)
             parameters = {'MODEL__alpha':[0.5,1.0]}
        if Params['Algorithm'] == 'RandomForest':
             model = RandomForestClassifier(n_estimators=100)
             parameters = {'MODEL__n_estimators':[50,100,150]}
        if Params['Algorithm'] == 'Neighbors':
            model = KNeighborsClassifier(n_neighbors=5,weights="uniform",algorithm ='auto', leaf_size = 30, p = 2, metric ='minkowski')
            parameters = {'MODEL__n_neighbors':[3,5],'MODEL__leaf_size':[15,30,50]}
        if Params['Algorithm'] == 'SGD':
            model = SGDClassifier(loss='log',penalty='l2')
            parameters = {'MODEL__penalty':['l1','l2'],'MODEL__alpha':[0.00005,0.0001],'MODEL__n_iter':[10]}
        if Params['Algorithm'] == 'Logistic':
            model = LogisticRegression()
            parameters = {'MODEL__penalty': ['l2'],'MODEL__C':[0.3,0.5,0.7]}
        if Params['Algorithm'] == 'ExtraTrees':   
            model = ExtraTreesClassifier(n_estimators=100)   
            parameters = {'MODEL__n_estimators':[50,100,150]}
        if Params['Algorithm'] == 'Ensemble':
             raise('not present!')
    #
    if Params['Compression']==1:
        parameters['PREPARATOR__n_components'] = [200]
        
    #decomposer = LatentDirichletAllocation(n_topics=10, max_iter=10,learning_method='online',learning_offset=50.,random_state=1)
    #decomposer = NMF(n_components=50, random_state=1,alpha=.1, l1_ratio=.5)
       
    """
    X = data.iloc[:]['text'].values
    y = data.iloc[:]['mylabel'].values.astype(str)
    
    dat = vect.fit_transform(X)
    dat = tfidf.fit_transform(dat)
    dat = decomposer.fit_transform(dat)  
    
    for a in numpy.unique(y):
        plt.scatter(dat[y==a,0],dat[y==a,1])
    """
     
    """
    START LOOP
    """
    k_fold = StratifiedKFold(n_splits=Params['CV-folds'],shuffle=True, random_state=666)
        
    accuracys=[]
    precisions=[]
    recalls=[]
    scores=[]
    
    confusion = np.array([[0, 0], [0, 0]])    
    
    best_score=(-1,-1)
    
    #Y = [y for y in FEAT['Y'][:,0].tolist()]
    X = FEAT['X']
    Params['n_custom_features']=0
    if Params['UseCustomFeatures']==1:
        try:
            X = scipy.sparse.hstack([X,FEAT['X_custom']])
            X = X.tocsr()
        except:
            X = np.hstack((X, FEAT['X_custom']))
        Params['n_custom_features'] = FEAT['X_custom'].shape[1]

    Y = FEAT['Y'].ravel()

    if Params['WordEmbedding'] == 1:
        PIPE = Pipeline([("PREPARATOR", TfidfEmbeddingVectorizer(Params,FEAT)),("MODEL", model)])
    else:
        PIPE = Pipeline([("PREPARATOR", prepare_data(Params)),("MODEL", model)])

    incorrect_samples = []
    best_parameters=[]
    k=0        
    print('\n---- Starting first loops ----')
    for train_indices, test_indices in k_fold.split(Y,Y):
        k+=1
        print('...fold',k)

        if Params['WordEmbedding'] == 1:
            XX = [X[i] for i in train_indices]
        else:
            XX = X[train_indices,:]
        YY = Y[train_indices]
        
        t0 = time()                
        
        grid_search = GridSearchCV(PIPE, parameters, n_jobs=3, verbose=1,cv=10)        
        grid_search.fit(X=XX,y=YY)
        best_parameters.append(grid_search.best_estimator_.get_params())        
        #PIPE.fit(X=XX,y=YY)
        
        #best_params = MODEL.best_params_
        
        print('.....training done in %0.3fs' % (time() - t0))

        if Params['WordEmbedding'] == 1:
            XX = [X[i] for i in test_indices]
        else:
            XX = X[test_indices,:]
        YY = Y[test_indices]                
        
        #XX = PREPARATOR.transform(XX)
        predictions = grid_search.predict(X=XX)
        #predictions = PIPE.predict(X=XX)

        incorrect_samples.append([x[0] for x in zip(test_indices,predictions,YY) if x[1]!=x[2]])

        accuracy,precision,recall,score,confusion1 = get_metrics(YY, predictions)
        
        accuracys.append(accuracy)
        scores.append(score)
        precisions.append(precision)
        recalls.append(recall)
        
        confusion += confusion1
        
        if score>best_score[0]:
            best_score=(score,k)          
        
        print('')
    
    scores_old = scores
    confusion_old = confusion
    
    print('\nTotal texts classified:', len(FEAT['Y']))
    print('Score:', sum(scores_old)/len(scores_old))
    print('Confusion matrix:')
    print(confusion_old)                    
    print(best_parameters)
    print(incorrect_samples)
