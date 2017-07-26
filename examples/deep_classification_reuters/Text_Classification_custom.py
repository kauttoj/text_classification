
# coding: utf-8

# # Text classification with Reuters-21578 datasets

# ### See: https://kdd.ics.uci.edu/databases/reuters21578/README.txt for more information

# In[ ]:


# In[ ]:

import re
import xml.sax.saxutils as saxutils

from numpy import zeros
import numpy as np

import random

from gensim.models.word2vec import Word2Vec
from gensim.models.wrappers import FastText

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD

from multiprocessing import cpu_count

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer

from pandas import DataFrame

from sklearn.cross_validation import train_test_split

import pickle
import os.path
# ## General constants (modify them according to you environment)

# In[ ]:

# Set Numpy random seed
random.seed(1000)

#sgml_number_of_files = 22
#sgml_file_name_template = 'reut2-NNN.sgm'
#
## Category files
#category_files = {
#    'to_': ('Topics', 'all-topics-strings.lc.txt'),
#    'pl_': ('Places', 'all-places-strings.lc.txt'),
#    'pe_': ('People', 'all-people-strings.lc.txt'),
#    'or_': ('Organizations', 'all-orgs-strings.lc.txt'),
#    'ex_': ('Exchanges', 'all-exchanges-strings.lc.txt')
#}
SOURCES=[
    ('D:/JanneK/Documents/text_classification/data/bbc/business','business'),
    ('D:/JanneK/Documents/text_classification/data/bbc/entertainment','entertainment'),
    ('D:/JanneK/Documents/text_classification/data/bbc/politics','politics'),
    ('D:/JanneK/Documents/text_classification/data/bbc/sport','sport')
]
# Word2Vec number of features
num_features = 200
# Limit each newsline to a fixed number of words
document_max_num_words = 150
# Selected categories

def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            print(file_name)
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                past_header, lines = False, []
                f = open(file_path, encoding='utf-8',errors='ignore')
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
        if len(text)>=500:
            rows.append({'text': text, 'mylabel': classification})
            index.append(file_name)
    if len(rows)<50:
        raise('ERROR: less than 50 samples!')
    data_frame = DataFrame(rows, index=index)
    return data_frame

data = DataFrame({'text': [], 'mylabel': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))

# In[ ]:


# ## Top 20 categories (by number of newslines)

# In[ ]:


# ## Tokenize newsline documents

# In[ ]:

# Load stop-words
stop_words = set(stopwords.words('english'))

# Initialize tokenizer
# It's also possible to try with a stemmer or to mix a stemmer and a lemmatizer
tokenizer = RegexpTokenizer('[\'a-zA-Z]+')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Tokenized document collection

# ## Vectorize each document

number_of_documents = data.shape[0]
num_categories = len(SOURCES)

filename = "savedXY.p"

newsline_documents = []
def tokenize(document):
    words = []

    for sentence in sent_tokenize(document):
        tokens = [lemmatizer.lemmatize(t.lower()) for t in tokenizer.tokenize(sentence) if t.lower() not in stop_words]
        words += tokens

    return words

if 1: #os.path.isfile(filename): 

    X = np.zeros((number_of_documents, document_max_num_words, num_features),dtype=np.float32)
    Y = np.zeros(shape=(number_of_documents, num_categories),dtype=np.float32)
    
    k=0
    yvals={}
    for a in data['mylabel'].values:
        if a not in yvals:
            yvals[a]=k
            k+=1     
    # Tokenize
    import codecs
    corpus='tempfile.txt'
    corpus1='tempfile_full.txt'
    
    f = codecs.open(corpus, "w", "utf-8")
    f1 = codecs.open(corpus1, "w", "utf-8")
    for k,doc in enumerate(data.iterrows()):
        print(k+1,'of',len(data))
        Y[k,yvals[doc[1]['mylabel']]] = 1
        aa=tokenize(doc[1]['text'])
        newsline_documents.append(aa)
        f.write(' '.join(aa))
        f1.write(str(yvals[doc[1]['mylabel']]) + '\t' +' '.join(aa) + '\n')
    f.close()
    f1.close()
    
#    w2v_model = Word2Vec(size=num_features, min_count=1, window=15, workers=cpu_count())
#    w2v_model.init_sims(replace=True)
#    w2v_model.save('D:/JanneK/Documents/text_classification/data/bbc/word2vec_model')
    outfile=r'D:/JanneK/Documents/text_classification/data/bbc/fasttext_model'
 
    w2v_model = FastText.train(corpus_file=corpus,output_file=outfile,ft_path=r'D:/JanneK/Documents/text_classification/fastText', model='cbow', size=num_features, alpha=0.025, window=10, min_count=2,loss='ns', sample=1e-3, negative=5, iter=5, min_n=3, max_n=6, sorted_vocab=1, threads=3)
   
    empty_word = zeros(num_features,dtype=np.float32)
    
    corpus2='tempfile_full_vector.txt'
    f = codecs.open(corpus2, "w", "utf-8")
    for idx, document in enumerate(newsline_documents):
        count=0
        a=empty_word
        for jdx, word in enumerate(document):
            if word in w2v_model:
                a+=w2v_model[word]
                count+=1
          
        a1 = ','.join(['%.3f' % num for num in Y[idx,:]])    
        a2 = ','.join(['%.3f' % num for num in a/count])        
        f.write(a1 + '|' +a2 + '\n')    
    f.close()    
    
    for idx, document in enumerate(newsline_documents):
        for jdx, word in enumerate(document):
            if jdx == document_max_num_words:
                break
                
            else:
                if word in w2v_model:
                    X[idx, jdx, :] = w2v_model[word]
                else:
                    X[idx, jdx, :] = empty_word

    with open(filename, "wb") as f:
        pickle.dump((X,Y),f)
    del w2v_model
    
    

else:

    with open(filename, "rb") as f:
        X,Y = pickle.load(f)

# ## Split training and test sets

# In[ ]:

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# ## Create Keras model

# In[ ]:

model = Sequential()

model.add(LSTM(int(document_max_num_words*1.5), input_shape=(document_max_num_words, num_features)))
model.add(Dropout(0.3))
model.add(Dense(num_categories))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam',#SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
          loss='binary_crossentropy',
          metrics=['accuracy'])


# ## Train and evaluate model

# In[ ]:

# Train model
model.fit(X_train, Y_train, batch_size=70, epochs=10, validation_data=(X_test, Y_test),shuffle=True)

# Evaluate model
score, acc = model.evaluate(X_test, Y_test, batch_size=70)
    
print('Score: %1.4f' % score)
print('Accuracy: %1.4f' % acc)

