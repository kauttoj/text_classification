
# coding: utf-8

# # Text classification with Reuters-21578 datasets
#https://github.com/giuseppebonaccorso/Reuters-21578-Classification/blob/master/Text%20Classification.ipynb
# ### See: https://kdd.ics.uci.edu/databases/reuters21578/README.txt for more information

# In[ ]:




# In[ ]:



#from gensim.models.word2vec import Word2Vec

import multiprocessing as mp

#from nltk.corpus import stopwords
#from nltk.tokenize import RegexpTokenizer, sent_tokenize
#from nltk.stem import WordNetLemmatizer

from pandas import DataFrame

from sklearn.cross_validation import train_test_split

import numpy as np

import random

import os
import sys

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


def get_metrics(true_labels, predicted_labels):
    
    from sklearn import metrics
    
    if true_labels.shape[1]>1:
        temp = []
        for i in range(true_labels.shape[0]):
            temp.append(np.where(true_labels[i,:])[0])
        true_labels=temp
    
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

def init(a,b):
    global shared_array,shared_object
    shared_array = a
    shared_object = b
    
def worker(i):
    shared_array[i] = i

if __name__ == '__main__':

    HOME=os.getcwd()
    os.chdir(HOME+os.sep+'..')
    sys.path.insert(0,os.getcwd())
    os.chdir(HOME)
    import get_my_data    
    
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
    document_max_num_words = 400
    
    filename = "laurea_LSTM_classifier_ver1_typeA.pickle"
    binfile = r'C:/Users/Jannek/Documents/git_repos/text_classification/data/wiki.fi'    
    
    import text_to_embedded_vectors
    X,Y,Y_vec = text_to_embedded_vectors.convert(filename,binfile,data,document_max_num_words,2)
    
    if 0:
        p=np.random.permutation(len(Y_vec))
        Y=Y[p,:]
        Y_vec=Y_vec[p]    
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, LSTM,Bidirectional
    from keras import backend as K
    K.set_learning_phase(1) #set learning phase
    
    batch_size=128
    def getmodel():
        model = Sequential()
        #model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        #model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        #model.add(MaxPooling1D(pool_size=2))
        rate_drop_lstm = 0.15
#        model.add(Bidirectional(LSTM(100,dropout=rate_drop_lstm,recurrent_dropout=rate_drop_lstm),input_shape=(document_max_num_words, num_features)))
        model.add(LSTM(100,dropout=rate_drop_lstm,recurrent_dropout=rate_drop_lstm,input_shape=(document_max_num_words, num_features)))
        # ORIGINAL:
        #model.add(LSTM(int, input_shape=(document_max_num_words, num_features)))
        #model.add(Dropout(0.3))
        model.add(Dense(num_categories))
        model.add(Activation('sigmoid')) # sigmoid
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
        return model
    
    from sklearn.model_selection import StratifiedKFold
    
    k_fold10 = StratifiedKFold(n_splits=10,shuffle=True,random_state=666)
#    k_fold2 = StratifiedKFold(n_splits=2,shuffle=True,random_state=666)
#    indices=[];
#    for train_index, test_index in k_fold5.split(X,Y_vec):
#        train_index1,_ = k_fold2.split(X[test_index,:],Y_vec[test_index])
#        test1 = test_index[train_index1[0]]
#        test2 = test_index[train_index1[1]]
#        indices.append((train_index,test1,test2,np.concatenate(test1,test2)))        
               
    accuracys=[]
    precisions=[]
    recalls=[]
    scores=[]    
    confusion = np.array([[0, 0], [0, 0]])         
    k=0        
    
    print('\n---- Starting first loops ----')
    #for train_indices, validate_indices, test_indices in indices:
    for train_index, test_index in k_fold10.split(X,Y_vec):
        k+=1

        print('...fold',k)
        
#        for flip in [False,True]:
#            if flip:
#                validate_indices, test_indices = test_indices , validate_indices
    
        # Train model                      
        model=getmodel()            
        
        model.fit(X[train_index,:], Y[train_index,:],
                  batch_size=batch_size, epochs=8, validation_split=0.1,shuffle=True)
                  #validation_data=(X[test_index,:],Y[test_index,:]))            
                    
        predictions = model.predict_classes(X[test_index,:], batch_size=batch_size)
        
        accuracy,precision,recall,score,confusion1 = get_metrics(Y[test_index,:],predictions)        
        
        # Evaluate model
        #score, acc = model.evaluate(X_final, Y_final, batch_size=batch_size)
        
        accuracys.append(accuracy)
        scores.append(score)
        precisions.append(precision)
        recalls.append(recall)
    
        confusion += confusion1               
        
    print('\n\n\nFINAL RESULTS:')
    print('Score: %1.4f' % np.mean(scores))
    print('Accuracy: %1.4f' % np.mean(accuracys))
    print(confusion)
    
    from keras.utils import plot_model
    import pydot
    import graphviz
    plot_model(model, to_file='laurea_LSTM_classifier_ver1_typeA_model.png')    
    