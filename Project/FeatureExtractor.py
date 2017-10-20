# -*- coding: utf-8 -*-
import numpy as np
from itertools import tee, islice

def count_numbers(data):
    c=0
    for tok in data['tokens']:
        try:
            _ = np.float(tok)
            c+=1
        except:       
            pass
    return c
    
def count_customtags(data,tags):
    c=np.zeros(len(tags))
    for tok in data['tokens']:
        for i,tag in enumerate(tags): 
            if tag == tok:
                c[i]+=1        
    return c
    
def count_POStags(data,tags):
    c=np.zeros(len(tags))
    for tok in data['POS_tags']:
        for i,tag in enumerate(tags): 
            if tag == tok:
                c[i]+=1        
    return c    
        
    
def construct_sentence(doc,Params):
    
    all_sent = []
    sent = []
    old_id = doc['sentence_id'][0]
    for i in range(0,len(doc['tokens'])):       
        id = doc['sentence_id'][i]
        if id>old_id:
            all_sent.append(sent)
            old_id = id
            sent=[]
                  
        if doc['is_char'][i]==1 and doc['is_number'][i]==0:            
            if Params['Lemmatize']==1:
                sent.append(doc['tokens'][i])
            else:
                sent.append(doc['tokens_raw'][i])                              
        else:
            sent.append('')
    
    return all_sent
    
def custom_analyzer(doc,Params):
    #print('')        
    
    for terms in construct_sentence(doc,Params):
        for ngramLength in range(1,Params['n-gram']+1):
            # find and return all ngrams
            # for ngram in zip(*[terms[i:] for i in range(3)]): <-- solution without a generator (works the same but has higher memory usage)
            for ngram in zip(*[islice(seq, i, len(terms)) for i, seq in enumerate(tee(terms, ngramLength))]): # <-- solution using a generator
                if Params['RemoveStops']==1:
                    ngram=[x for x in ngram if x not in Params['stopword_list']]
                    if len(ngram)<ngramLength:
                        continue                                                          
                ngram = ' '.join(ngram)            
                ngram=ngram.strip()
                
                yield ngram    
    
def cut_empty_features(X,X_labels):

    count = np.sum(X,axis=0)
    bad = count==0
    X = np.delete(X,bad,axis=1)
    X_labels = [x[0] for x in zip(X_labels,bad) if x[1]==False]                
    return X,X_labels
                
def main(data_in,Params):
            
#    Params['UseCustomFeatures'] = 1
#    Params['n-gram'] = 2
#    Params['WordSmoothing'] = 1
#    Params['WordEmbedding'] = 1

    feat = {}        
        
    if Params['WordEmbedding'] == 1:
        pass                
    else:        
        from sklearn.feature_extraction.text import CountVectorizer
#        vect = CountVectorizer(max_df=0.85,min_df=0,max_features = 20000,ngram_range=(1,2))    
        obj = CountVectorizer(analyzer=lambda x: custom_analyzer(x,Params),max_df=0.70,min_df=0.0025,max_features = 25000,ngram_range=(1,Params['n-gram']))    
        # test if it works
        X = obj.fit_transform(data_in)
        
        feat['X']=X
        feat['X_labels'] = obj.get_feature_names()  
        feat['X_transformer'] = obj
        
    print('Total %i main features created' % (len(feat['X_labels'])))        
    
    
    Y= np.zeros((X.shape[0],1))
    for i,data in enumerate(data_in):
        Y[i] = data['target']        
    feat['Y'] = Y
                 
    if Params['UseCustomFeatures'] == 1:

        X = np.zeros((len(data_in),200))
        X_labels = []
            
        for i,data in enumerate(data_in):
            
            L = len(data['tokens'])
                
            k=-1   
                    
            k+=1
            X[i,k] = sum(data['is_number'])/L
            if i==0:
                X_labels+=['Numbers count']
                
            k+=1            
            X[i,k]=L
            if i==0:
                X_labels+=['Word count']                
                
            tags = [*Params['POS_TAGS']]
            x = count_POStags(data,tags)/L   
            X[i,k:(k+len(x))] = x
            k+=len(x)
            if i==0:
                X_labels+=tags                               
            
            k+=1            
            X[i,k]=np.sum(np.array(data['in_dictionary'])*np.array(data['is_char']))/L
            if i==0:
                X_labels+=['Dictionary word count']
            
            if Params['UseCustomFeatures']==1:
                tags = [*Params['CUSTOM_TAGS']]
                x = count_customtags(data,tags)/L            
                X[i,k:(k+len(x))] = x
                k+=len(x)
                if i==0:
                    X_labels+=tags
                    
        X = X[:,0:k]
        
        N_old = len(X_labels)
        X,X_labels = cut_empty_features(X,X_labels)        
        N_new = len(X_labels)
            
        feat['X_custom']=X
        feat['X_custom_labels'] = X_labels
        
        print('Total %i custom features created (was %i) with %i features omitted (empty columns)' % (len(X_labels),N_old,N_old-N_new))
        
    
    return feat
    
