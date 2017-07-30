# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 14:04:20 2017

@author: Jannek
"""
import numpy as np
    
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
