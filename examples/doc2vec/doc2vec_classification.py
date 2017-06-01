
# coding: utf-8

# In this notebook I benchmark a few text categorization models to test whehter word embeddings like word2vec can improve text classification accuracy. The the notebook proceeds as follows:
# 1. downloading the datasets
# 2. construction of the training set
# 3. definitions of models
# 4. benchmarking models
# 5. plotting results

# Downloading datasets and pretrained wector embeddings. Especially the embeddings can take a while to download. You might want run these in the terminal instead to see wget's progress bar. If you're on Windows (and not in cygwin) %%bash cell magic won't work and you'll have to do all this manually (or with %%cmd magic I guess). 

# In[ ]:

#get_ipython().run_cell_magic('bash', '', "# download Reuters' text categorization benchmarks\nwget http://www.cs.umb.edu/~smimarog/textmining/datasets/r8-train-no-stop.txt\nwget http://www.cs.umb.edu/~smimarog/textmining/datasets/r8-test-no-stop.txt\n# concatenate train and test files, we'll make our own train-test splits\ncat r8-*-no-stop.txt > r8-no-stop.txt\nwget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-train-no-stop.txt\nwget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-test-no-stop.txt\ncat r52-*-no-stop.txt > r52-no-stop.txt\nwget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-test-no-stop.txt\nwget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-train-no-stop.txt\ncat 20ng-*-no-stop.txt > 20ng-no-stop.txt")


# In[ ]:

#get_ipython().run_cell_magic('bash', '', '# download GloVe word vector representations\n# bunch of small embeddings - trained on 6B tokens - 822 MB download, 2GB unzipped\nwget http://nlp.stanford.edu/data/glove.6B.zip\nunzip glove.6B.zip\n\n# and a single behemoth - trained on 840B tokens - 2GB compressed, 5GB unzipped\nwget http://nlp.stanford.edu/data/glove.840B.300d.zip\nunzip glove.840B.300d.zip')


# In[6]:

from tabulate import tabulate
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit

import os.path


# TRAIN_SET_PATH = "20ng-no-stop.txt"
# TRAIN_SET_PATH = "r52-all-terms.txt"
TRAIN_SET_PATH = "r8-no-stop.txt"

GLOVE_6B_50D_PATH = "glove.6B.50d.txt"
GLOVE_840B_300D_PATH = "glove.840B.300d.txt"


# In[2]:
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import LabeledSentence
    
os.path.isfile(TRAIN_SET_PATH) 

X, y,XX = [], [],[]
with open(TRAIN_SET_PATH, "r") as infile:
    for line in infile:
        label, text = line.split("\t")
        # texts are already tokenized, just split on space
        # in a real case we would use e.g. spaCy for tokenization
        # and maybe remove stopwords etc.
        words = text.split()
        X.append(words)
        
        sentence = LabeledSentence(words=words,tags=label)    
        XX.append(sentence)
        
        y.append(label)
X, y = np.array(X), np.array(y)
#print("total examples %s" % len(y))


# Prepare word embeddings - both the downloaded pretrained ones and train a new one from scratch

    
    # In[7]:


# train word2vec on all the texts - both training and test set
# we're not using test labels, just texts so this is fine
model = Doc2Vec(XX, size=100, window=5, min_count=5, workers=2)
model.wv.index2word
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
print('gensim completed!')
    

# In[]


import numpy as np
with open(GLOVE_6B_50D_PATH, "rb") as lines:
    word2vec = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}


# In[4]:

# reading glove files, this may take a while
# we're reading line by line and only saving vectors
# that correspond to words from our training set
# if you wan't to play around with the vectors and have 
# enough RAM - remove the 'if' line and load everything

glove_small = {}
dim = 50
all_words = set(w for words in X for w in words)
with open(GLOVE_6B_50D_PATH, "r",encoding='utf8') as infile:
    k=0
    for line in infile:
        k+=1       
        if k%5000 == 0:
            print('line ',k)
        parts = line.split()
        word = ''.join(parts[0:(len(parts)-dim)])
        nums = list(map(float, parts[-dim:]))     
        if word in all_words:
            try:                
                glove_small[word] = np.array(nums)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise
            
glove_big = {}
dim = 300
with open(GLOVE_840B_300D_PATH, "r",encoding='utf8') as infile:
    k=0
    for line in infile:
        k+=1
        if k%5000 == 0:
            print('line ',k)        
        parts = line.split()        
        word = ''.join(parts[0:(len(parts)-dim)])
        nums = list(map(float, parts[-dim:]))
        if word in all_words:
            glove_big[word] = np.array(nums)

assert(len(glove_small)>1)
assert(len(glove_big)>1)



# Time for model definitions

# In[8]:

# start with the classics - naive bayes of the multinomial and bernoulli varieties
# with either pure counts or tfidf features
mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
# SVM - which is supposed to be more or less state of the art 
# http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])


# Now the meat - classifiers using vector embeddings. We will implement an embedding vectorizer - a counterpart of CountVectorizer and TfidfVectorizer - that is given a word -> vector mapping and vectorizes texts by taking the mean of all the vectors corresponding to individual words.

# In[9]:

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # this line is different from python2 version - no more itervalues
        self.dim = model.vector_size
        
    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])   
# and a tf-idf version of the same

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = model.vector_size
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])  


# In[10]:

# Extra Trees classifier is almost universally great, let's stack it with our embeddings
etree_glove_small = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_small)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_small_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_small)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_big = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_big)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_big_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_big)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])

etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])


# benchmark all the things!

# In[25]:

all_models = [
    ("mult_nb", mult_nb),
    ("mult_nb_tfidf", mult_nb_tfidf),
    ("bern_nb", bern_nb),
    ("bern_nb_tfidf", bern_nb_tfidf),
    ("svc", svc),
    ("svc_tfidf", svc_tfidf),
    ("glove_small", etree_glove_small), 
    ("glove_small_tfidf", etree_glove_small_tfidf),
    ("glove_big", etree_glove_big), 
    ("glove_big_tfidf", etree_glove_big),
    ("w2v", etree_w2v),
    ("w2v_tfidf", etree_w2v_tfidf),
]
#scores = sorted(,key=lambda (_, x): -x )
scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in all_models]
print(tabulate(scores, floatfmt=".4f", headers=("model", 'score')))


# In[26]:

plt.figure(figsize=(15, 6))
sns.barplot(x=[name for name, _ in scores], y=[score for _, score in scores])


# ok, this is how it is. Let's see how the ranking depends on the amount of training data. Word embedding models which are semi-supervised should shine when there is very little labeled training data

# In[27]:

def benchmark(model, X, y, n):
    test_size = 1 - (n / float(len(y)))
    scores = []
    for train, test in StratifiedShuffleSplit(y, n_iter=5, test_size=test_size):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        scores.append(accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test))
    return np.mean(scores)


# In[28]:

train_sizes = [10, 40, 160, 640, 3200, 6400]
table = []
for name, model in all_models:
    for n in train_sizes:
        table.append({'model': name, 
                      'accuracy': benchmark(model, X, y, n), 
                      'train_size': n})
df = pd.DataFrame(table)


# In[29]:

plt.figure(figsize=(15, 6))
fig = sns.pointplot(x='train_size', y='accuracy', hue='model', 
                    data=df[df.model.map(lambda x: x in ["mult_nb", "svc_tfidf", "w2v_tfidf", 
                                                         "glove_small_tfidf", "glove_big_tfidf", 
                                                        ])])
sns.set_context("notebook", font_scale=1.5)
fig.set(ylabel="accuracy")
fig.set(xlabel="labeled training examples")
fig.set(title="R8 benchmark")
fig.set(ylabel="accuracy")


