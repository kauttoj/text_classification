# LSTM and CNN for sequence classification in the IMDB dataset
# http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from itertools import tee, islice
from nltk import tokenize

import os
import re
import sys
HOME=os.getcwd()
os.chdir(HOME+'\\..')
sys.path.insert(0,os.getcwd())
os.chdir(HOME)
import get_my_data
print('\n--- Parsing data ---')
data=get_my_data.getdata()

MAX_NGRAM=1

def custom_analyzer(doc):
    #print('')
    for ln in tokenize.sent_tokenize(doc):
        terms = re.findall(r'\w{2,}', ln)
        for ngramLength in range(1,MAX_NGRAM+1):
            # find and return all ngrams
            # for ngram in zip(*[terms[i:] for i in range(3)]): <-- solution without a generator (works the same but has higher memory usage)
            for ngram in zip(*[islice(seq, i, len(terms)) for i, seq in enumerate(tee(terms, ngramLength))]): # <-- solution using a generator
                ngram = ' '.join(ngram)
                yield ngram
#vect = CountVectorizer(max_df=0.85,min_df=0,max_features = 5000,ngram_range=(1,3))    
vect = CountVectorizer(analyzer=custom_analyzer,max_df=0.85,min_df=0,max_features=5000,ngram_range=(1,3))    

# test if it works
data = vect.fit_transform(data.iloc[:]['text'].values)

"""X_train = []
X_test = []
y_train = []
y_test = []
for i in range(0,data.shape[0]-):
    X_train[]
"""

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))