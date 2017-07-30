# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 12:15:22 2017

@author: Jannek
"""

import os.path
import pickle
from gensim.models.wrappers import FastText
import numpy as np
from functools import partial

USE_PARALLEL = False

def processInput(document,model,document_max_num_words,num_features):
    words = document[1].split(' ')
    #Y[idx,yvals[document[0]]] = 1
    #print('...processing document',idx)
    mat=np.zeros((document_max_num_words, num_features),dtype=np.float32)
    k=0
    for jdx, word in enumerate(words):

        ind = document_max_num_words-jdx+k-1
        if jdx == document_max_num_words:
            break
        else:
            success=False
            if 0:#word in oldwords:
                if oldwords!=None:
                    mat[ind, :] = oldwords[word]
                    success=True
            else:
                if word in model:
                    mat[ind,:] = model[word]
                    #oldwords[word]=X[idx,ind,:]
                    success=True
                #else:
                    #oldwords[word]=None
            if not success:
                k+=1
    return mat

def convert(filename,binfile,data,document_max_num_words=200,num_categories=2):

    #filename = "laurea_LSTM_classifier_ver1_typeA.p"
    #binfile = r'C:/Users/Jannek/Documents/git_repos/text_classification/data/wiki.fi'

    if not os.path.isfile(filename):

        # Load Skipgram or CBOW model
        print('Loading Fasttext model')
        model = FastText.load_fasttext_format(binfile)
        #model={'testi1':np.zeros((300)),'testi2':np.zeros((300))}

        num_features=model.wv.vector_size
        number_of_documents=data.shape[0]

        X = np.zeros((number_of_documents, document_max_num_words, num_features),dtype=np.float32)
        Y = np.zeros(shape=(number_of_documents, num_categories),dtype=np.float32)

        yvals={}
        k=0
        for a in data['mylabel'].values:
            if a not in yvals:
                yvals[a]=k
                k+=1
                
        for idx, document in enumerate(data.values):
            Y[idx,yvals[document[0]]] = 1                

        #p = mp.Pool(initializer=init, initargs=(a,anim))

        print('Vectorizing text')

        func = partial(processInput, model=model,document_max_num_words=document_max_num_words,num_features=num_features)                   

        if USE_PARALLEL:            
            import multiprocessing                     
            pool = multiprocessing.Pool(2)        
            results = pool.map_async(func, data.values)  
            for i,mat in enumerate(results):
                X[i,:,:]=mat
        else:            
            for i,doc in enumerate(data.values):
                print('..processing document',i+1)
                X[i,:,:]=func(doc)
            
        with open(filename, "wb") as f:
            pickle.dump((X,Y),f)

    else:

        with open(filename, "rb") as f:
            X,Y = pickle.load(f)
            
    if Y.shape[1]>1:
        Y_vec = np.zeros(Y.shape[0])
        for i in range(Y.shape[0]):
            Y_vec[i] = np.where(Y[i,:])[0]
    else:
        Y_vec=Y

    return X,Y,Y_vec
    
    