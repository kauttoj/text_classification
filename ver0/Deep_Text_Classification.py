
# coding: utf-8

# # Text classification with Reuters-21578 datasets
#https://github.com/giuseppebonaccorso/Reuters-21578-Classification/blob/master/Text%20Classification.ipynb
# ### See: https://kdd.ics.uci.edu/databases/reuters21578/README.txt for more information

# In[ ]:




# In[ ]:


import re

#from gensim.models.word2vec import Word2Vec



import multiprocessing as mp

#from nltk.corpus import stopwords
#from nltk.tokenize import RegexpTokenizer, sent_tokenize
#from nltk.stem import WordNetLemmatizer

from pandas import DataFrame

from sklearn.cross_validation import train_test_split

import get_my_data

import numpy as np

import random

from gensim.models.wrappers import FastText

def update_frequencies(categories):
    for category in categories:
        idx = news_categories[news_categories.Name == category].index[0]
        f = news_categories.get_value(idx, 'Newslines')
        news_categories.set_value(idx, 'Newslines', f+1)
    
def to_category_vector(categories, target_categories):
    vector = zeros(len(target_categories)).astype(float32)
    
    for i in range(len(target_categories)):
        if target_categories[i] in categories:
            vector[i] = 1.0
    
    return vector

def init(a,b):
    global shared_array,shared_object
    shared_array = a
    shared_object = b
    
def worker(i):
    shared_array[i] = i

if __name__ == '__main__':

    print('\n--- Parsing data ---')
    data=get_my_data.getdata()
    
    random.seed(666)
    # ## General constants (modify them according to you environment)
    
    # In[ ]:
    
    
    # Set Numpy random seed
    
    
    
    # Selected categories
    
    
    # ## Prepare documents and categories
    
    # In[ ]:
    
    
    # Create category dataframe
    
    # In[ ]:
    
    
    
    number_of_documents = data.shape[0]
    num_categories = 2
    # Word2Vec number of features
    num_features = 300
    # Limit each newsline to a fixed number of words
    document_max_num_words = 100
    
    import pickle
    import os.path
    
    filename = "savedXY.p"
    
    if not os.path.isfile(filename): 
    
        X = np.zeros((number_of_documents, document_max_num_words, num_features),dtype=np.float32)
        Y = np.zeros(shape=(number_of_documents, num_categories),dtype=np.float32)
        
        # Load Skipgram or CBOW model
        model = FastText.load_fasttext_format('D:/JanneK/Documents/text_classification/data/wiki.fi')
        k=0
        yvals={}
        for a in data['mylabel'].values:
            if a not in yvals:
                yvals[a]=k
                k+=1
        
        oldwords={}
        #p = mp.Pool(initializer=init, initargs=(a,anim))
            
        for idx, document in enumerate(data.values):
            words = document[1].split(' ')
            
            Y[idx,yvals[document[0]]] = 1 
            
            print('processing document',idx)
            
            for jdx, word in enumerate(words):
        
                if jdx == document_max_num_words:
                    break
                else:
                    if word in oldwords:
                        if oldwords!=None:
                            X[idx, jdx, :] = oldwords[word] 
                    else:
                        if word in model:
                            X[idx, jdx, :] = model[word] 
                            oldwords[word]=X[idx, jdx, :]
                        else:
                            oldwords[word]=None
        with open(filename, "wb") as f:
            pickle.dump((X,Y),f)
        del model,oldwords
    
    else:
    
        with open(filename, "rb") as f:
            X,Y = pickle.load(f)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, LSTM
    
    model = Sequential()
    
    model.add(LSTM(int(document_max_num_words*1.5), input_shape=(document_max_num_words, num_features)))
    model.add(Dropout(0.3))
    model.add(Dense(num_categories))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train, Y_train, batch_size=128, nb_epoch=5, validation_data=(X_test, Y_test))
    
    # Evaluate model
    score, acc = model.evaluate(X_test, Y_test, batch_size=128)
        
    print('Score: %1.4f' % score)
    print('Accuracy: %1.4f' % acc)