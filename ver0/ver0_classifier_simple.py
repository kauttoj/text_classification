# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:23:33 2017

http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html
"""

import os
import numpy
from pandas import DataFrame
from time import time
import pickle
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.decomposition import NMF,TruncatedSVD

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
#from vowpalwabbit import pyvw
#from vowpalwabbit.sklearn_vw import VWClassifier

import get_my_data


def show_most_informative_features(model, text=None, n=20):
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
    
    a = numpy.round(metrics.accuracy_score(true_labels,predicted_labels),3)
    print('Accuracy:',a)
    
    p = numpy.round(metrics.precision_score(true_labels,predicted_labels,average='weighted'),3) 
    print('Precision:',p)
    
    r = numpy.round(metrics.recall_score(true_labels,predicted_labels,average='weighted'),3)    
    print('Recall:', r)
    
    s = numpy.round(metrics.f1_score(true_labels,predicted_labels,average='weighted'),3)
    print('F1 Score:', s)
    
    cm = metrics.confusion_matrix(true_labels, predicted_labels)
    print(cm)
    
    return a,p,r,s,cm

#if __name__ == '__main__':
    
print('\n--- Parsing data ---')
data=get_my_data.getdata()
    
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

vect = CountVectorizer(max_df=0.75,max_features = 20000,ngram_range=(1,3))    
tfidf = TfidfTransformer();
model = LogisticRegression()
#decomposer = TruncatedSVD(2)
decomposer = NMF(n_components=50, random_state=1)
    
"""
START LOOP
"""
k_fold = KFold(n_splits=5,shuffle=True, random_state=666)
    
accuracys=[]
precisions=[]
recalls=[]
scores=[]

confusion = numpy.array([[0, 0], [0, 0]])    

best_parameters=[]
best_score=(-1,-1)

k=0        
print('\n---- Starting first loops ----')
for train_indices, test_indices in k_fold.split(data):
    k+=1
    print('...fold',k)
    
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['mylabel'].values.astype(str)

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['mylabel'].values.astype(str)        
    
    t0 = time()
    
    dat = vect.fit_transform(train_text)
    dat = tfidf.fit_transform(dat)
    
    print('raw data size ',dat.shape)         
    dat = decomposer.fit_transform(dat)  
    
    print('final data size ',dat.shape)      
    model.fit(X=dat,y=train_y)
    
    print('.....training done in %0.3fs' % (time() - t0))
    
    dat = vect.transform(test_text)
    dat = tfidf.transform(dat)        
    dat = decomposer.transform(dat)      
    
    predictions = model.predict(X=dat)
        
    accuracy,precision,recall,score,confusion1 = get_metrics(test_y, predictions)
    
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

print('\nTotal texts classified:', len(data))
print('Score:', sum(scores_old)/len(scores_old))
print('Confusion matrix:')
print(confusion_old)                          
"""
median_parameters = average_params(best_parameters,best_score[1])

grid_search = GridSearchCV( pipeline, median_parameters, n_jobs=3, verbose=1,cv=2)
    
accuracys=[]
precisions=[]
recalls=[]
scores=[]    
confusion = numpy.array([[0, 0], [0, 0]])

k=0        
print('\n\n---- Starting FINAL loops with optimal parameters ----')
for train_indices, test_indices in k_fold.split(data):
    k+=1
    print('...fold',k)
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['mylabel'].values.astype(str)

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['mylabel'].values.astype(str)        
    
    t0 = time()
    grid_search.fit(train_text,train_y)
    print('.....training done in %0.3fs' % (time() - t0))        
    
    predictions = grid_search.predict(test_text)

    accuracy,precision,recall,score,confusion1 = get_metrics(test_y, predictions)
    
    accuracys.append(accuracy)
    scores.append(score)
    precisions.append(precision)
    recalls.append(recall)
    
    confusion += confusion1
    
    print('')        

print('\nTotal texts classified:', len(data))
print('Score:', sum(scores)/len(scores))
print('Precision:', sum(precisions)/len(precisions))
print('Recall:', sum(recalls)/len(recalls))
print('Confusion matrix:')
print(confusion)
"""