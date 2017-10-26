# -*- coding: utf-8 -*-
import numpy as np
from itertools import tee, islice
import os.path
from gensim.models.wrappers import FastText
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

def processInput(document,model,Params,word_matrix):

    words = []
    for terms in construct_sentence(document, Params):
        if Params['RemoveStops'] == 1:
            terms = [x for x in terms if x not in Params['stopword_list']]
        words = words + terms

    #document_max_num_words = len(words)
    # Y[idx,yvals[document[0]]] = 1
    # print('...processing document',idx)
    #mat = np.zeros((document_max_num_words, num_features), dtype=np.float32)
    ind = []
    for word in words:
        if word in word_matrix['word']:
            ind.append(word_matrix['word'].index(word))
        elif word in model:
            word_matrix['word'].append(word)
            a=np.array(model[word],dtype=np.float32)
            word_matrix['vector']=np.vstack((word_matrix['vector'],a)) if word_matrix['vector'].size else a
        else:
            ind.append(-1)

    return ind,word_matrix


def convert_to_embedded(Params,data):
    # filename = "laurea_LSTM_classifier_ver1_typeA.p"
    # binfile = r'C:/Users/Jannek/Documents/git_repos/text_classification/data/wiki.fi'

    # Load Skipgram or CBOW model
    print('Loading Fasttext model (this takes couple of minutes)')
    model = FastText.load_fasttext_format(Params['FastTextBin'])
    # model={'testi1':np.zeros((300)),'testi2':np.zeros((300))}
    num_features = model.vector_size
       # p = mp.Pool(initializer=init, initargs=(a,anim))
    word_matrix = {'vector': np.array([]), 'word': []}
    X=[]

    print('Starting text vectorization')
    if 0:
        import multiprocessing

        number_of_documents = len(data)
        func = partial(processInput, model=model, Params=Params, num_features=num_features, word_matrix=word_matrix)
        pool = multiprocessing.Pool(2)
        results = pool.map_async(func, data.values)
        for i, mat in enumerate(results):
            X.append(mat)
    else:
        for i,doc in enumerate(data):
            print('..processing document', i + 1)
            x,word_matrix = processInput(doc, model, Params, word_matrix)
            X.append(x)

    print('Done! Text vectorized')

    return X,word_matrix

def get_word_embeddings(Params,docs):
    print('Loading Fasttext model (this takes couple of minutes)')
    model = FastText.load_fasttext_format(Params['FastTextBin'])
    # model={'testi1':np.zeros((300)),'testi2':np.zeros((300))}
    num_features = model.vector_size
       # p = mp.Pool(initializer=init, initargs=(a,anim))
    X = {}
    for doc in docs:
        for word in doc:
            if word in X:
                pass
            elif word in model:
                a=np.array(model[word],dtype=np.float32)
                X[word] = a
            else:
                X[word] = None

    return X

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
        #else:
        #    sent.append('')
    
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
    good = count>0
    X = X[:,good]
    X_labels = [x[0] for x in zip(X_labels,good) if x[1]]
    assert(len(X_labels)==X.shape[1])
    return X,X_labels

def main(data_in,Params):
            
#    Params['UseCustomFeatures'] = 1
#    Params['n-gram'] = 2
#    Params['WordSmoothing'] = 1
#    Params['WordEmbedding'] = 1

    feat = {}

    if Params['WordEmbedding'] is not 'none':

        from sklearn.feature_extraction.text import CountVectorizer
        #        vect = CountVectorizer(max_df=0.85,min_df=0,max_features = 20000,ngram_range=(1,2))
        Params['n-gram']=1
        X=[]
        for i in data_in:
            X.append([x for x in custom_analyzer(i, Params)])

        if Params['WordEmbedding'] is not 'doc2vec':
            datafile = Params['OUTPUT-folder'] + '/embedded_data_temp.pickle'
            if not os.path.isfile(datafile):
                word2vec = get_word_embeddings(Params, X)
                pickle.dump(word2vec,open(datafile,"wb"))
            else:
                word2vec = pickle.load(open(datafile, "rb"))
            X_labels = list(word2vec.keys())
            Params['embedding_dimension'] = 300

            #X,X_labels = pool_embedded(doc_inds,Params,word_matrix)
            feat['word2vec'] = word2vec
        else:
            X_labels = 'doc2vec_vector'

        Y = np.zeros((len(X),1))
    else:

        from sklearn.feature_extraction.text import CountVectorizer
#        vect = CountVectorizer(max_df=0.85,min_df=0,max_features = 20000,ngram_range=(1,2))    
        obj = CountVectorizer(analyzer=lambda x: custom_analyzer(x,Params),max_df=0.70,min_df=0.0025,max_features = 50000,ngram_range=(1,Params['n-gram']))
        # test if it works
        X = obj.fit_transform(data_in)
        X_labels = obj.get_feature_names()
        feat['X_transformer'] = obj
        Y = np.zeros((X.shape[0], 1))

    feat['X'] = X
    feat['X_labels'] = X_labels

    print('Total %i main features created' % (len(feat['X_labels'])))
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

            k+=1
            tags = [*Params['POS_TAGS']]
            x = count_POStags(data,tags)/L   
            X[i,k:(k+len(x))] = x
            k+=len(x)-1
            if i==0:
                X_labels+=tags                               
            
            k+=1            
            X[i,k]=np.sum(np.array(data['in_dictionary'])*np.array(data['is_char']))/L
            if i==0:
                X_labels+=['Dictionary word count']
            
            if Params['CUSTOMTagging']==1:
                k+=1
                tags = [*Params['CUSTOM_TAGS']]
                x = count_customtags(data,tags)/L            
                X[i,k:(k+len(x))] = x
                k+=len(x)-1
                if i==0:
                    X_labels+=tags
                    
        X = X[:,0:(k+1)]
        
        N_old = len(X_labels)
        X,X_labels = cut_empty_features(X,X_labels)        
        N_new = len(X_labels)
            
        feat['X_custom']=X
        feat['X_custom_labels'] = X_labels
        
        print('Total %i custom features created (was %i) with %i features omitted (empty columns)' % (len(X_labels),N_old,N_old-N_new))
        
    
    return feat
    
