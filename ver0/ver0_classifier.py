# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:23:33 2017

http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html
"""

import os
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from time import time

def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                past_header, lines = False, []
                f = open(file_path, encoding="latin-1")
                for line in f:
                    if past_header and len(line)>0 and line is not '\n':
                        line=line.rstrip()
                        lines.append(line)
                    else:
                        past_header = True                        
                f.close()
                content = ' '.join(lines)
                yield file_path, content


def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

def getdata():
    """
    DATA
    """
    SOURCES=[
        ('C:/Users\Jannek/Documents/git_repos/text_classification/data/bbc/business','BUSINESS'),
        ('C:/Users\Jannek/Documents/git_repos/text_classification/data/bbc/politics','POLITICS')
    ]
    
    data = DataFrame({'text': [], 'class': []})
    for path, classification in SOURCES:
        data = data.append(build_data_frame(path, classification))
    
    data = data.reindex(numpy.random.permutation(data.index))
    
    return data

if __name__ == '__main__':
    
    data=getdata()
        
    """
    MODEL
    """
    SVM_model = svm.SVC(kernel = 'linear')
    Forest_model = RandomForestClassifier()
    SGD_model = SGDClassifier(penalty='l2')
    Bayes_model=MultinomialNB()
    
    """
    PARAMETERS
    """
    parameters = {
        'vect__max_df': [0.75],
        'vect__max_features': [None],
        'vect__ngram_range': [(1,2),(1,3),(1,4),(1,5)],  # unigrams or bigrams
        'vect__stop_words': ['english'],
        'tfidf__use_idf': [True],
        'tfidf__norm': ['l2'],
        #'SGD__alpha': (0.0001,0.00001, 0.000001),
        #'SGD__penalty': ['l2'],
        #'SGD__n_iter': [50],
        #'svm__C': numpy.logspace(-2,2,8).tolist(),
        #'bayes__alpha': [0.75]
        'forest__n_estimators':[10,15,20,25]
    }
    
    
    """
    PIPELINE
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer()),    
        ('tfidf', TfidfTransformer()),
        ('forest',Forest_model)
    ])      
        
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1)
    """
    START LOOP
    """
    k_fold = KFold(n_splits=10,shuffle=True, random_state=666)
    
    scores = []
    confusion = numpy.array([[0, 0], [0, 0]])    
    best_parameters=[]
    k=0
        
    for train_indices, test_indices in k_fold.split(data):
        k+=1
        print('...split',k)
        train_text = data.iloc[train_indices]['text'].values
        train_y = data.iloc[train_indices]['class'].values.astype(str)
    
        test_text = data.iloc[test_indices]['text'].values
        test_y = data.iloc[test_indices]['class'].values.astype(str)        
        
        t0 = time()
        grid_search.fit(train_text,train_y)
        print('.....training done in %0.3fs' % (time() - t0))
        
        best_parameters.append(grid_search.best_estimator_.get_params())
        
        predictions = grid_search.predict(test_text)
    
        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label=data.iloc[0][0])
        scores.append(score)
        
        print('.....score was ',score)
        print('')        
    
    print('Total texts classified:', len(data))
    print('Score:', sum(scores)/len(scores))
    print('Confusion matrix:')
    print(confusion)