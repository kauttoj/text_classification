
# coding: utf-8

# # Text classification with Reuters-21578 datasets

# ### See: https://kdd.ics.uci.edu/databases/reuters21578/README.txt for more information

# In[ ]:


# In[ ]:

import re
import xml.sax.saxutils as saxutils

from bs4 import BeautifulSoup

from numpy import zeros
import numpy

import random

from gensim.models.word2vec import Word2Vec
from gensim.models.wrappers import FastText

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM

from multiprocessing import cpu_count

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer

from pandas import DataFrame

from sklearn.cross_validation import train_test_split


# ## General constants (modify them according to you environment)

# In[ ]:

# Set Numpy random seed
random.seed(1000)

# Newsline folder and format
data_folder = r'D:/JanneK/Documents/text_classification/examples/deep_classification_reuters/reuters21578/'

sgml_number_of_files = 22
sgml_file_name_template = 'reut2-NNN.sgm'

# Category files
category_files = {
    'to_': ('Topics', 'all-topics-strings.lc.txt'),
    'pl_': ('Places', 'all-places-strings.lc.txt'),
    'pe_': ('People', 'all-people-strings.lc.txt'),
    'or_': ('Organizations', 'all-orgs-strings.lc.txt'),
    'ex_': ('Exchanges', 'all-exchanges-strings.lc.txt')
}

# Word2Vec number of features
num_features = 500
# Limit each newsline to a fixed number of words
document_max_num_words = 100
# Selected categories
selected_categories = ['pl_usa']


# ## Prepare documents and categories

# In[ ]:

# Create category dataframe

# Read all categories
category_data = []

for category_prefix in category_files.keys():
    with open(data_folder + category_files[category_prefix][1], 'r') as file:
        for category in file.readlines():
            category_data.append([category_prefix + category.strip().lower(), 
                                  category_files[category_prefix][0], 
                                  0])

# Create category dataframe
news_categories = DataFrame(data=category_data, columns=['Name', 'Type', 'Newslines'])


# In[ ]:

def update_frequencies(categories):
    for category in categories:
        idx = news_categories[news_categories.Name == category].index[0]
        f = news_categories.get_value(idx, 'Newslines')
        news_categories.set_value(idx, 'Newslines', f+1)
    
def to_category_vector(categories, target_categories):
    vector = zeros(len(target_categories),dtype=numpy.float32)
    
    for i in range(len(target_categories)):
        if target_categories[i] in categories:
            vector[i] = 1.0
    
    return vector


# In[ ]:

# Parse SGML files
document_X = {}
document_Y = {}

def strip_tags(text):
    return re.sub('<[^<]+?>', '', text).strip()

def unescape(text):
    return saxutils.unescape(text)

# Iterate all files
for i in range(sgml_number_of_files):
    if i < 10:
        seq = '00' + str(i)
    else:
        seq = '0' + str(i)
        
    file_name = sgml_file_name_template.replace('NNN', seq)
    print('Reading file: %s' % file_name)
    
    with open(data_folder + file_name, 'r') as file:
        content = BeautifulSoup(file.read().lower())
        
        for newsline in content('reuters'):
            document_categories = []
            
            # News-line Id
            document_id = newsline['newid']
            
            # News-line text
            document_body = strip_tags(str(newsline('text')[0].body)).replace('reuter\n&#3;', '')
            document_body = unescape(document_body)
            
            # News-line categories
            topics = newsline.topics.contents
            places = newsline.places.contents
            people = newsline.people.contents
            orgs = newsline.orgs.contents
            exchanges = newsline.exchanges.contents
            
            for topic in topics:
                document_categories.append('to_' + strip_tags(str(topic)))
                
            for place in places:
                document_categories.append('pl_' + strip_tags(str(place)))
                
            for person in people:
                document_categories.append('pe_' + strip_tags(str(person)))
                
            for org in orgs:
                document_categories.append('or_' + strip_tags(str(org)))
                
            for exchange in exchanges:
                document_categories.append('ex_' + strip_tags(str(exchange)))
                
            # Create new document    
            update_frequencies(document_categories)
            
            document_X[document_id] = document_body
            document_Y[document_id] = to_category_vector(document_categories, selected_categories)


# ## Top 20 categories (by number of newslines)

# In[ ]:

news_categories.sort_values(by='Newslines', ascending=False, inplace=True)
news_categories.head(20)


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
newsline_documents = []


# In[ ]:

def tokenize(document):
    words = []

    for sentence in sent_tokenize(document):
        tokens = [lemmatizer.lemmatize(t.lower()) for t in tokenizer.tokenize(sentence) if t.lower() not in stop_words]
        words += tokens

    return words

# Tokenize
for key in document_X.keys():
    newsline_documents.append(tokenize(document_X[key]))

number_of_documents = len(document_X)


# ## Word2Vec Model
# ### See: https://radimrehurek.com/gensim/models/word2vec.html and https://code.google.com/p/word2vec/ for more information

# In[ ]:

# Load an existing Word2Vec model
#w2v_model = Word2Vec.load('D:/Downloads/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin')


# In[ ]:

# Create new Gensim Word2Vec model
#w2v_model = Word2Vec(newsline_documents, size=num_features, min_count=1, window=10, workers=cpu_count())
#w2v_model.init_sims(replace=True)
#w2v_model.save('D:/Downloads/GoogleNews-vectors-negative300/Reuters_300')

#w2v_model = FastText(newsline_documents,size=num_features, alpha=0.025, window=10, min_count=1, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)
#w2v_model.init_sims(replace=True)
#w2v_model.save('D:/Downloads/GoogleNews-vectors-negative300/Reuters_300_fasttext')

# ## Vectorize each document

# In[ ]:

num_categories = len(selected_categories)
X = zeros(shape=(number_of_documents, document_max_num_words, num_features),dtype=numpy.float32)
Y = zeros(shape=(number_of_documents, num_categories),dtype=numpy.float32)

empty_word = zeros(num_features,dtype=numpy.float32)

for idx, document in enumerate(newsline_documents):
    for jdx, word in enumerate(document):
        if jdx == document_max_num_words:
            break
            
        else:
            if word in w2v_model:
                X[idx, jdx, :] = w2v_model[word]
            else:
                X[idx, jdx, :] = empty_word

for idx, key in enumerate(document_Y.keys()):
    Y[idx, :] = document_Y[key]


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

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# ## Train and evaluate model

# In[ ]:

# Train model
model.fit(X_train, Y_train, batch_size=128, nb_epoch=5, validation_data=(X_test, Y_test))

# Evaluate model
score, acc = model.evaluate(X_test, Y_test, batch_size=128)
    
print('Score: %1.4f' % score)
print('Accuracy: %1.4f' % acc)

