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

import gensim
from gensim.models.doc2vec import TaggedDocument
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2

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

# TruncatedSVD useful for large sparse datasets which cannot be centered without making the memory usage explode.
class prepare_bow_data(BaseEstimator, TransformerMixin):
        
    def __init__(self, Params=None,n_components=100):
        self.Params = Params
        self.n_components=n_components
        self.decomposer = TruncatedSVD(n_components=n_components)
        #self.decomposer = NMF(n_components=n_components)
        self.tfidf = TfidfTransformer()
            
    def fit(self,X, y=None):
        if self.Params['n_custom_features']==0:
            dat = self.tfidf.fit_transform(X)
        else:
            dat = self.tfidf.fit_transform(X[:,0:-self.Params['n_custom_features']])
        if self.Params['Compress']==1:
            self.decomposer.fit(dat)
        return self    
            
    def transform(self,X):
        if self.Params['n_custom_features'] == 0:
            dat = self.tfidf.transform(X)
        else:
            dat = self.tfidf.transform(X[:,0:-self.Params['n_custom_features']])
        if self.Params['Compress']==1:
            dat = self.decomposer.transform(dat)
            if self.Params['n_custom_features'] > 0:
                dat = np.concatenate((dat,X[:,-self.Params['n_custom_features']:].todense()),axis=1)
        else:
            if self.Params['n_custom_features'] > 0:
                dat = scipy.sparse.hstack([dat,X[:,-self.Params['n_custom_features']:]])
        return dat
#
# class prepare_embedded_data(BaseEstimator, TransformerMixin):
#
#     def __init__(self, Params=None, Feat = None):
#         self.Params = Params
#         self.pipe = None
#         self.n_components = None
#         self.inds = None
#         self.Feat = Feat
#
#     def fit(self, X, y=None):
#         self.inds = np.where(np.sum(X[:,0:-self.Params['n_custom_features']],axis=0)>0)[0]
#         X = self.Feat['X_transformer'][self.inds, :]
#         self.inds = self.inds.tolist()
#         self.n_components = X.shape[1]-5
#         self.pipe = Pipeline([('scaler', StandardScaler()), ('pca',PCA(n_components=self.n_components))])
#         self.pipe.fit(X)
#         return self
#
#     def transform(self, X):
#         XX = np.zeros((X.shape[0],self.n_components),dtype=np.float32)
#         for row in range(XX.shape[0]):
#             ind = X[row,0:-self.Params['n_custom_features']].nonzero()[0]
#             x = 0
#             k = 0
#             for elem in ind:
#                 if elem in self.inds:
#                     xx = self.Feat['X_transformer'][elem,:].reshape(1, -1)
#                     x += xx*X[row,elem]
#                     k += X[row,elem]
#
#             x /= k
#             x = self.pipe.transform(x)
#
#             XX[row,0:self.n_components] = x
#         XX = np.concatenate((XX, X[:, -self.Params['n_custom_features']:]), axis=1)
#         return XX


# class IdentityTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass
#
#     def fit(self, input_array, y=None):
#         return self
#
#     def transform(self, input_array, y=None):
#         return input_array * 1

class doc2vecTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, Params=None,dim=50):
        self.Params = Params
        self.model = None
        self.dim = dim

    def fit(self, X, y=None):
        corpus = []
        for i,x in enumerate(X):
            corpus.append(TaggedDocument(x,[i]))

        # # doc2vec parameters
        # vector_size = 300
        # window_size = 15
        # min_count = 1
        # sampling_threshold = 1e-5
        # negative_size = 5
        # train_epoch = 100
        # dm = 0  # 0 = dbow; 1 = dmpv

        self.model = gensim.models.doc2vec.Doc2Vec(size=self.dim,window = 12,negative=5,sample=1e-4,min_count=2,iter=200)
        self.model.build_vocab(corpus)
        for epoch in range(0,10):
            self.model.train(np.random.permutation(corpus), total_examples=self.model.corpus_count, epochs=self.model.iter)
        return self

    def transform(self, X, y=None):
        result = []
        for i, doc in enumerate(X):
            result.append(self.model.infer_vector(doc))
        return result

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"dim": self.dim}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

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

        if self.Params['WordEmbedding'] is 'word2vec_pca':
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
            if self.Params['WordEmbedding'] is 'word2vec_pca':
                res[i,:] = self.pca.transform(x.reshape(1, -1) / k)
            else:
                res[i,:] = x/k
        return res

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"dim": self.dim}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def main(FEAT=None,Params=None):
                   
    if Params['TargetType'] == 'regression':
        if Params['Algorithm'] == 'SGD':
            model = SGDClassifier(penalty='l2',loss='squared_loss')     
            parameters = {'penalty':['l1','l2']}
    else:

        if Params['Algorithm'] == 'SVC':
            model = svm.SVC(kernel = 'rbf',C = 1.0)
            parameters = {'MODEL__C': [0.1,0.5,1.0]}
        if Params['Algorithm'] == 'NaiveBayes':
             model = MultinomialNB(alpha = 1.0)
             parameters = {'MODEL__alpha':[0.5,1.0]}
        if Params['Algorithm'] == 'RandomForest':
             model = RandomForestClassifier(n_estimators=100)
             parameters = {'MODEL__n_estimators':[50,100,150]}
        if Params['Algorithm'] == 'Neighbors':
            model = KNeighborsClassifier(n_neighbors=5,weights="uniform",algorithm ='auto', leaf_size = 30, p = 2, metric ='minkowski')
            parameters = {'MODEL__n_neighbors':[3,5],'MODEL__leaf_size':[30]}
        if Params['Algorithm'] == 'SGD':
            model = SGDClassifier(loss='log',penalty='l2')
            parameters = {'MODEL__penalty':['l2'],'MODEL__alpha':[0.00001,0.0001,0.001],'MODEL__n_iter':[10]}
        if Params['Algorithm'] == 'Logistic':
            model = LogisticRegression()
            parameters = {'MODEL__penalty': ['l1','l2'],'MODEL__C':[0.5,0.75,1.0]}
        if Params['Algorithm'] == 'ExtraTrees':   
            model = ExtraTreesClassifier(n_estimators=100)   
            parameters = {'MODEL__n_estimators':[100,200]}
        if Params['Algorithm'] == 'Ensemble':
             raise('not present!')
    #
    if Params['Compress']==1 and Params['WordEmbedding'] is 'none':
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

    if Params['WordEmbedding'] is not 'none':
        if Params['WordEmbedding'] is not 'doc2vec':
            PIPE = Pipeline([("PREPARATOR", TfidfEmbeddingVectorizer(Params,FEAT)),("MODEL", model)])
        else:
            PIPE = Pipeline([("PREPARATOR", doc2vecTransformer(Params,100)), ("MODEL", model)])
            parameters['PREPARATOR__dim']=[130]
    else:
        PIPE = Pipeline([("PREPARATOR", prepare_bow_data(Params)),("MODEL", model)])

    incorrect_samples = []
    best_parameters=[]
    k=0        
    print('\n---- Starting first loops ----')
    for train_indices, test_indices in k_fold.split(Y,Y):
        k+=1
        print('...fold %i of %i' % (k,Params['CV-folds']))

        if Params['WordEmbedding'] is not 'none':
            XX = [X[i] for i in train_indices]
        else:
            XX = X[train_indices,:]
        YY = Y[train_indices]
        
        t0 = time()                
        
        #grid_search = GridSearchCV(PIPE, parameters, n_jobs=-1, verbose=1,cv=10,refit=True)
        #grid_search.fit(X=XX,y=YY)
        #best_parameters.append(grid_search.best_estimator_.get_params())
        PIPE.fit(X=XX,y=YY)
        
        #best_params = MODEL.best_params_
        
        print('.....training done in %0.3fs' % (time() - t0))

        if Params['WordEmbedding'] is not 'none':
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

        print('Confusion matrix so far:')
        print(confusion)
        print('best parameters:')
        print(print(best_parameters[-1]))
        
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
