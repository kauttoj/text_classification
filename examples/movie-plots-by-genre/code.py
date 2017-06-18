
# coding: utf-8

# ## Word Embeddings for Fun and Profit
# ### Document classification with Gensim
# 
# In this tutorial we'll classify movie plots by genre using word embeddings techniques in [gensim](http://radimrehurek.com/gensim/) . 
# 
# See accompanying slides in this repo.
# 
# We will show how to get a __'hello-world'__ first untuned run using 7 techniques:
# 
# - Bag of words
# 
# - Character n-grams
# 
# - TF-IDF 
# 
# - Averaging word2vec vectors
# 
# - doc2vec
# 
# - Deep IR 
# 
# - Word Mover's Distance
# 
# The goal of this tutorial is to show the API so you can start tuning them yourself. Model tuning of the models is out of scope of this tutorial.
# 
# We will also compare the accuracy of this first 'no tuning'/out of the box run of these techniques. It is in no way an indication of their best peformance that can be achieved with proper tuning. The benefit of the comparison is to manage the expectations.

# ## Requirements
# - Python 3
# - [Google News pre-trained word2vec (1.5 GB)](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
# - gensim
# - sklearn
# - pandas
# - matplotlib
# - nltk with English stopwords
# - pyemd
# - 4 GB RAM
# - 8 GB disk space for WMD

# ## Dataset
# We will use MovieLens dataset linked with plots from OMDB. Thanks to [Sujit Pal](http://sujitpal.blogspot.de/2016/04/predicting-movie-tags-from-plots-using.html) for this linking idea. The prepared csv is in this repository. If you wish to link the datasets yourself - see the code in the [blog]((http://sujitpal.blogspot.de/2016/04/predicting-movie-tags-from-plots-using.html).

# In[126]:

import logging
logging.root.handlers = []  # Jupyter messes up logging so needs a reset
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from smart_open import smart_open
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from nltk.corpus import stopwords

from IPython import get_ipython
ipython = get_ipython()


# ## Exploring the data
# 
# 

# In[25]:

df = pd.read_csv('data/tagged_plots_movielens.csv')
df = df.dropna()
df['plot'].apply(lambda x: len(x.split(' '))).sum()


# The dataset is only __170k__ words. It is quite small but makes sure we don't have to wait a long time for the code to complete.

# In[26]:

my_tags = ['sci-fi' , 'action', 'comedy', 'fantasy', 'animation', 'romance']
df.tag.value_counts().plot(kind="bar", rot=0)


# The data is very unbalanced. We have Comedy as majority class. 
# 
# A naive classifier that predicts everything to be comedy already achieves __40%__ accuracy.

# The language in sci-fi plots differs a lot from action plots, so there should be some signal here.

# In[27]:

df


# In[28]:

def print_plot(index):
    example = df[df.index == index][['plot', 'tag']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Genre:', example[1])


# In[29]:


# Train/test split of 90/10

# In[31]:

train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)


# In[32]:

len(test_data)


# In[33]:

test_data.tag.value_counts().plot(kind="bar", rot=0)


# ## Model evaluation approach
# We will use confusion matrices to evaluate all classifiers

# In[34]:

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(my_tags))
    target_names = my_tags
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[35]:

def evaluate_prediction(predictions, target, title="Confusion matrix"):
    print('accuracy %s' % accuracy_score(target, predictions))
    cm = confusion_matrix(target, predictions)
    print('confusion matrix\n %s' % cm)
    print('(row=expected, col=predicted)')
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, title + ' Normalized')


# In[36]:

def predict(vectorizer, classifier, data):
    data_features = vectorizer.transform(data['plot'])
    predictions = classifier.predict(data_features)
    target = data['tag']
    evaluate_prediction(predictions, target)


# ## Baseline: bag of words, n-grams, tf-idf
# Let's start with some simple baselines before diving into more advanced methods.

# ### Bag of words

# The simplest document feature is just a count of each word occurrence in a document.
# 
# We remove stop-words and use NLTK tokenizer then limit our vocabulary to 3k most frequent words.

# In[37]:

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


# In[38]:

count_vectorizer = CountVectorizer(analyzer="word", tokenizer=nltk.word_tokenize,
                                   preprocessor=None, stop_words='english', max_features=3000) 
train_data_features = count_vectorizer.fit_transform(train_data['plot'])


# Multi-modal logistic regression is a simple white-box classifier. We will use either logistic regression or KNN throughout this tutorial.

# In[41]:

logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
logreg = logreg.fit(train_data_features, train_data['tag'])


# In[42]:

count_vectorizer.get_feature_names()[80:90]


# Nothing impressive - only 2% better better than the classifier that thinks that everything is a comedy.

# In[43]:

predict(count_vectorizer, logreg, test_data)


# White box vectorizer and classifier are great! We can see what are the most important words for sci-fi. This makes it very easy to tune and debug.

# In[44]:

def most_influential_words(vectorizer, genre_index=0, num_words=10):
    features = vectorizer.get_feature_names()
    max_coef = sorted(enumerate(logreg.coef_[genre_index]), key=lambda x:x[1], reverse=True)
    return [features[x[0]] for x in max_coef[:num_words]]    


# In[45]:

# words for the fantasy genre
genre_tag_id = 1
print(my_tags[genre_tag_id])
most_influential_words(count_vectorizer, genre_tag_id)


# In[46]:

train_data_features[0]


# ### Character N-grams

# A character _n-gram_ is a chunk of a document of length _n_. It is a poor man's tokenizer but sometimes works well. The parameter _n_ depends on language and the corpus. We choose length between 3 and 6 characters and to only focus on 3k most popular ones.

# In[43]:

n_gram_vectorizer = CountVectorizer(analyzer="char",ngram_range=([2,5]),tokenizer=None,preprocessor=None,max_features=3000) 
logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
train_data_features = n_gram_vectorizer.fit_transform(train_data['plot'])
logreg = logreg.fit(train_data_features, train_data['tag'])


# In[44]:

n_gram_vectorizer.get_feature_names()[50:60]


# The results are worse than using a tokenizer and bag of words. Probably due to not removing the stop words.

# In[45]:

predict(n_gram_vectorizer, logreg, test_data)


# ### TF-IDF
# 
# [Term Frequency - Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is a little more advanced way to count words in a document.
# It adjusts for document length, word frequency and most importantly for frequency of a particular word in a particular document.
# 

# In[53]:

tf_vect = TfidfVectorizer(min_df=2, tokenizer=nltk.word_tokenize,preprocessor=None, stop_words='english')
train_data_features = tf_vect.fit_transform(train_data['plot'])
logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
logreg = logreg.fit(train_data_features, train_data['tag'])


# In[54]:

tf_vect.get_feature_names()[1000:1010]


# In[48]:

predict(tf_vect, logreg, test_data)


# White box vectorizer and classifier are great! We can see what are the most important words for sci-fi. This makes it very easy to tune and debug.

# In[60]:

most_influential_words(tf_vect, 1)


# ### Things to try with bag of words
# 
# 10 mins for exercises.
# 
# For more insight into the model print out the most influential words for a particular plot.
# 
# Try n-grams with TF-IDF.
# 
# 

# # Averaging word vectors

# Now let's use more complex features rather than just counting words.
# 
# A great recent achievement of NLP is the [word2vec embedding](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf). See Chris Moody's [video](https://www.youtube.com/watch?v=vkfXBGnDplQ) for a great introduction to word2vec. 
# 

# First we load a word2vec model. It has been pre-trained by Google on a 100 billon word Google News corpus. You can play with this model using a fun [web-app](http://rare-technologies.com/word2vec-tutorial/#app).
# 
# Link to the web-app: http://rare-technologies.com/word2vec-tutorial/#app
# 
# Vocabulary size: 3 mln words. 
# 
# __Warning__: 3 mins to load, takes 4 GB of RAM.

# In[47]:

wv = KeyedVectors.load_word2vec_format("/data/w2v_googlenews/GoogleNews-vectors-negative300.bin.gz",binary=True)
v.init_sims(replace=True)


# Example vocabulary

# In[64]:

from itertools import islice
list(islice(wv.vocab, 13000, 13020))


# Now we have a vector for each word. How do we get a vector for a sequence of words (aka a document)?
# 
# 
# 

# The most naive way is just to take an average. [Mike Tamir](https://www.youtube.com/watch?v=7gTjYwiaJiU) has suggested that the resulting vector points to a single word summarising the whole document. For example all words in a book
#  ‘A tale of two cities’ should add up to 'class-struggle’

# <img src="images/naivedoc2vec.png">

# In[53]:


def word_averaging(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.layer1_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, review) for review in text_list ])


# For word2vec we apply a different tokenization. We want to preserve case as the vocabulary distingushes lower and upper case.

# In[88]:

def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


# In[50]:

test_tokenized = test_data.apply(lambda r: w2v_tokenize_text(r['plot']), axis=1).values
train_tokenized = train_data.apply(lambda r: w2v_tokenize_text(r['plot']), axis=1).values


# In[54]:

X_train_word_average = word_averaging_list(wv,train_tokenized)
X_test_word_average = word_averaging_list(wv,test_tokenized)


# Let's see how KNN and logistic regression classifiers perform on these word-averaging document features.

# In[55]:

knn_naive_dv = KNeighborsClassifier(n_neighbors=3, n_jobs=1, algorithm='brute', metric='cosine' )
knn_naive_dv.fit(X_train_word_average, train_data['tag'])


# In[56]:

predicted = knn_naive_dv.predict(X_test_word_average)


# In[57]:

evaluate_prediction(predicted, test_data['tag'])


# KNN is even worse than the naive 'everything is comedy' baseline! Let's see if logistic regression is better.

# In[94]:

logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
logreg = logreg.fit(X_train_word_average, train_data['tag'])
predicted = logreg.predict(X_test_word_average)


# Great! It gives __54%__ accuracy. Best that we have seen so far.

# In[95]:

evaluate_prediction(predicted, test_data['tag'])


# Now just for fun let's see if text summarisation works on our data. Let's pick a plot and see which words it averages to.

# In[96]:

test_data.iloc()[56]


# Hmm... The summarisation doesn't work here. Any ideas why? Hint: look at the area where the average ends up.

# In[97]:

wv.most_similar(positive=[X_test_word_average[56]], restrict_vocab=100000, topn=30)[0:20]


# ### Word2vec things to try

# 10 mins exercise
# 
# Remove stop-words. 
# 
# 
# 

# In[51]:



def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            if word in stopwords.words('english'):
                continue
            tokens.append(word)
    return tokens


# ### What accuracy do you get?

# In[ ]:

### More word2vec things to try


# # Doc2Vec

# A [paper](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) by Google suggests a model for document classification called Paragraph Vectors Doc2Vec or Doc2vec in short. It is very similar to word2vec. 
# 
# It introduces 'a tag' - a word that is in every context in the document.
# 
# For our first try we tag every plot with its genre. This makes it 'semi-supervised' learning - the genre labels is just one objective among many.

# In[59]:

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


# In[60]:

train_tagged = train_data.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['plot']), tags=[r.tag]), axis=1)


# In[61]:

test_tagged = test_data.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['plot']), tags=[r.tag]), axis=1)


# This is what a training entry looks like - an example plot tagged by 'sci-fi'.

# In[62]:

test_tagged.values[50]


# In[141]:

trainsent = train_tagged.values
testsent = test_tagged.values # simple gensim doc2vec api\ndoc2vec_model = Doc2Vec(trainsent, workers=1, size=5, iter=20, dm=1)\n\ntrain_targets, train_regressors = zip(\n    *[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=20)) for doc in trainsent])')


# Interesting thing about doc2vec is that we need to run gradient descent during prediction to infer the vector for an unseen document. An unseen document is initially assigned a random vector and then this vector fit by gradient descent. Because of this randomness we get different vectors on re-runs of the next cell.
# 
# Consequently, the accuracy of logistic regression changes when the test set vectors change.

# In[150]:

test_targets, test_regressors = zip(*[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=20)) for doc in testsent])


# In[151]:

logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
logreg = logreg.fit(train_regressors, train_targets)
evaluate_prediction(logreg.predict(test_regressors), test_targets, title=str(doc2vec_model))


# KNN gives a lower accuracy than logistic regression.

# In[140]:

knn_test_predictions = [doc2vec_model.docvecs.most_similar([pred_vec], topn=1)[0][0] for pred_vec in test_regressors]
evaluate_prediction(knn_test_predictions, test_targets, str(doc2vec_model))


# Doc2vec gives us a vector for each genre so we can see which genres are close together.

# In[67]:

doc2vec_model.docvecs.most_similar('action')


# Words surrounding the 'sci-fi' tag describe it pretty accurately!

# In[68]:

doc2vec_model.most_similar([doc2vec_model.docvecs['sci-fi']])


# ### Doc2vec exercise
# 
# 10 mins
# 
# Find the random seed that gives the best prediction. :)
# 
# 

# In[137]:

seed = 20

doc2vec_model.seed = seed
doc2vec_model.random = random.RandomState(seed)


test_targets, test_regressors = zip(
    *[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=20)) for doc in testsent])


logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5, random_state=42)
logreg = logreg.fit(train_regressors, train_targets)
evaluate_prediction(logreg.predict(test_regressors), test_targets, title=str(doc2vec_model))
print(doc2vec_model.seed)


# ## Doc2vec things to try
# Try tagging every sentence with a unique tag 'SENT_123' and then apply KNN. 
# 
# Try multiple tags per plot as in this repo published __today__ : https://github.com/sindbach/doc2vec_pymongo
# 
# 

# # Deep IR

# 'Deep IR' is a technique developed by  [“Document Classification by Inversion of Distributed Language Representations”, Matt Taddy](http://arxiv.org/pdf/1504.07295v3.pdf). Matt has contributed a gensim [tutorial](https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/deepir.ipynb) - great source of more in depth information.
# 
# In short the algorithm is:
# 
# 1. Train a word2vec model only on comedy plots.
# 
# 2. Trian another model only on sci-fi, another on romance etc. Get 6 models - one for each genre.
# 
# 3. Take a plot and see which model fits it best using Bayes' Theorem
# 
# 

# The tokenization is different from other methods. The reason for this is that we are following an original approach in the paper. The purpose of this tutorial is to see how the models behave out of the box.
# 
# We just clean non-alphanumeric characters and split by sentences.

# In[152]:

import re
contractions = re.compile(r"'|-|\"")
# all non alphanumeric
symbols = re.compile(r'(\W+)', re.U)
# single character removal
singles = re.compile(r'(\s\S\s)', re.I|re.U)
# separators (any whitespace)
seps = re.compile(r'\s+')

# cleaner (order matters)
def clean(text): 
    text = text.lower()
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = singles.sub(' ', text)
    text = seps.sub(' ', text)
    return text

# sentence splitter
alteos = re.compile(r'([!\?])')
def sentences(l):
    l = alteos.sub(r' \1 .', l).rstrip("(\.)*\n")
    return l.split(".")


# In[154]:

def plots(label):
    my_df = None
    if label=='training':
        my_df = train_data
    else:
        my_df = test_data
    for i, row in my_df.iterrows():
        yield {'y':row['tag'],        'x':[clean(s).split() for s in sentences(row['plot'])]}


# In[155]:

revtrain = list(plots("training"))
revtest = list(plots("test"))


# In[156]:

# shuffle training set for unbiased word2vec training
np.random.shuffle(revtrain)


# In[157]:

def tag_sentences(reviews, stars=my_tags):  
    for r in reviews:
        if r['y'] in stars:
            for s in r['x']:
                yield s


# An example `sci-fi` sentence:

# In[158]:

next(tag_sentences(revtrain, my_tags[0]))


# We train our own 6 word2vec models from scratch. 

# In[135]:

from gensim.models import Word2Vec
import multiprocessing
basemodel = Word2Vec(workers=multiprocessing.cpu_count(),iter=100, hs=1, negative=0)
basemodel.build_vocab(tag_sentences(revtrain)) 
from copy import deepcopy

genremodels = [deepcopy(basemodel) for i in range(len(my_tags))]
for i in range(len(my_tags)):
    slist = list(tag_sentences(revtrain, my_tags[i]))
    print(my_tags[i], "genre (", len(slist), ")")
    genremodels[i].train(  slist, total_examples=len(slist) )
    
    # get the probs (note we give docprob a list of lists of words, plus the models)')


# Now we will compute most likely class for a plot using Bayes' Theorem formula.

# <img src='images/deep_ir_bayes.png' width=600>

# For any new sentence we can obtain its _likelihood_ (lhd; actually, the composite likelihood approximation; see the paper) using the [score](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.score) function in the `word2vec` class.  We get the likelihood for each sentence in the first test review, then convert to a probability over star ratings. Every sentence in the review is evaluated separately and the final star rating of the review is an average vote of all the sentences. This is all in the following handy wrapper. (from the original [tutorial](https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/deepir.ipynb) by Matt Taddy.

# In[136]:

"""
docprob takes two lists
* docs: a list of documents, each of which is a list of sentences
* models: the candidate word2vec models (each potential class)

it returns the array of class probabilities.  Everything is done in-memory.
"""


def docprob(docs, mods):
    # score() takes a list [s] of sentences here; could also be a sentence generator
    sentlist = [s for d in docs for s in d]
    # the log likelihood of each sentence in this review under each w2v representation
    llhd = np.array( [ m.score(sentlist, len(sentlist)) for m in mods ] )
    # now exponentiate to get likelihoods, 
    lhd = np.exp(llhd - llhd.max(axis=0)) # subtract row max to avoid numeric overload
    # normalize across models (stars) to get sentence-star probabilities
    prob = pd.DataFrame( (lhd/lhd.sum(axis=0)).transpose() )
    # and finally average the sentence probabilities to get the review probability
    prob["doc"] = [i for i,d in enumerate(docs) for s in d]
    prob = prob.groupby("doc").mean()
    return prob


# In[137]:


    probs = docprob( [r['x'] for r in revtest], genremodels )  
    predictions = probs.idxmax(axis=1).apply(lambda x: my_tags[x])


# In[138]:

tag_index = 0
col_name = "out-of-sample prob positive for " + my_tags[tag_index]
probpos = pd.DataFrame({col_name:probs[[tag_index]].sum(axis=1), 
                        "true genres": [r['y'] for r in revtest]})
probpos.boxplot(col_name,by="true genres", figsize=(12,5))


# In[94]:

target = [r['y'] for r in revtest]


# In[95]:

evaluate_prediction(predictions, target, "Deep IR with word2vec")


# Performance is worse than for a naive predictor that says that everything is `comedy`.
# 
# ### Why?

# 
# 
# It is because we train each word2vec model from scratch on a very small sample of about 30k words.
# 
# This model needs more data.

# # Word Mover's Distance

# <img src='images/wmd_gelato.png'>
# 
# Image from 
# http://tech.opentable.com/2015/08/11/navigating-themes-in-restaurant-reviews-with-word-movers-distance/

# Word Mover's Distance is a new algorithm developed in by [Matt Kusner](http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf). There is Matt's code on [github](https://github.com/mkusner/wmd) and also Gensim can compute WMD similarity in this [PR](https://github.com/piskvorky/gensim/pull/659).

# For KNN the best code is from [Vlad Niculae's blog](http://vene.ro/blog/word-movers-distance-in-python.html). He is a contributor to sklearn and did great integration of WMD with sklearn KNN.

# __Warning__: Write 7 GB file on disk to use memory mapping.

# ## __This part requires Python 3__

# In[159]:

data_folder = '/data'
fp = np.memmap(data_folder + "embed.dat", dtype=np.double, mode='w+', shape=wv.syn0norm.shape)
fp[:] = wv.syn0norm[:]

with smart_open(data_folder + "embed.vocab", "w") as f:
    for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
        print(w.encode('utf8'), file=f)
        del fp, wv


# In[97]:

W = np.memmap(data_folder + "embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))
with smart_open(data_folder + "embed.vocab", mode="rb") as f:
    vocab_list = [line.strip() for line in f]


# In[98]:

vocab_dict = {w: k for k, w in enumerate(vocab_list)}


# ### sklearn KNN integration with WMD

# In[99]:

"""%%file word_movers_knn.py"""

# Authors: Vlad Niculae, Matt Kusner
# License: Simplified BSD

import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.externals.joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_array
from sklearn.cross_validation import check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.preprocessing import normalize

from pyemd import emd


class WordMoversKNN(KNeighborsClassifier):
    """K nearest neighbors classifier using the Word Mover's Distance.

    Parameters
    ----------
    
    W_embed : array, shape: (vocab_size, embed_size)
        Precomputed word embeddings between vocabulary items.
        Row indices should correspond to the columns in the bag-of-words input.

    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`k_neighbors` queries.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for Word Mover's Distance computation.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
    
    verbose : int, optional
        Controls the verbosity; the higher, the more messages. Defaults to 0.
        
    References
    ----------
    
    Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger
    From Word Embeddings To Document Distances
    The International Conference on Machine Learning (ICML), 2015
    http://mkusner.github.io/publications/WMD.pdf
    
    """
    _pairwise = False

    def __init__(self, W_embed, n_neighbors=1, n_jobs=1, verbose=False):
        self.W_embed = W_embed
        self.verbose = verbose
        super(WordMoversKNN, self).__init__(n_neighbors=n_neighbors, n_jobs=n_jobs,
                                            metric='precomputed', algorithm='brute')

    def _wmd(self, i, row, X_train):
        """Compute the WMD between training sample i and given test row.
        
        Assumes that `row` and train samples are sparse BOW vectors summing to 1.
        """
        union_idx = np.union1d(X_train[i].indices, row.indices)
        W_minimal = self.W_embed[union_idx]
        W_dist = euclidean_distances(W_minimal)
        bow_i = X_train[i, union_idx].A.ravel()
        bow_j = row[:, union_idx].A.ravel()
        return emd(bow_i, bow_j, W_dist)
    
    def _wmd_row(self, row, X_train):
        """Wrapper to compute the WMD of a row with all training samples.
        
        Assumes that `row` and train samples are sparse BOW vectors summing to 1.
        Useful for parallelization.
        """
        n_samples_train = X_train.shape[0]
        return [self._wmd(i, row, X_train) for i in range(n_samples_train)]

    def _pairwise_wmd(self, X_test, X_train=None):
        """Computes the word mover's distance between all train and test points.
        
        Parallelized over rows of X_test.
        
        Assumes that train and test samples are sparse BOW vectors summing to 1.
        
        Parameters
        ----------
        X_test: scipy.sparse matrix, shape: (n_test_samples, vocab_size)
            Test samples.
        
        X_train: scipy.sparse matrix, shape: (n_train_samples, vocab_size)
            Training samples. If `None`, uses the samples the estimator was fit with.
        
        Returns
        -------
        dist : array, shape: (n_test_samples, n_train_samples)
            Distances between all test samples and all train samples.
        
        """
        n_samples_test = X_test.shape[0]
        
        if X_train is None:
            X_train = self._fit_X

        dist = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._wmd_row)(test_sample, X_train)
            for test_sample in X_test)

        return np.array(dist)

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : scipy sparse matrix, shape: (n_samples, n_features)
            Training data. 

        y : {array-like, sparse matrix}
            Target values of shape = [n_samples] or [n_samples, n_outputs]

        """
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        return super(WordMoversKNN, self).fit(X, y)

    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X : scipy.sparse matrix, shape (n_test_samples, vocab_size)
            Test samples.

        Returns
        -------
        y : array of shape [n_samples]
            Class labels for each data sample.
        """
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        dist = self._pairwise_wmd(X)
        return super(WordMoversKNN, self).predict(dist)
    
    
class WordMoversKNNCV(WordMoversKNN):
    """Cross-validated KNN classifier using the Word Mover's Distance.

    Parameters
    ----------
    W_embed : array, shape: (vocab_size, embed_size)
        Precomputed word embeddings between vocabulary items.
        Row indices should correspond to the columns in the bag-of-words input.

    n_neighbors_try : sequence, optional
        List of ``n_neighbors`` values to try.
        If None, tries 1-5 neighbors.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, StratifiedKFold is used.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for Word Mover's Distance computation.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    verbose : int, optional
        Controls the verbosity; the higher, the more messages. Defaults to 0.

    Attributes
    ----------
    cv_scores_ : array, shape (n_folds, len(n_neighbors_try))
        Test set scores for each fold.

    n_neighbors_ : int,
        The best `n_neighbors` value found.

    References
    ----------

    Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger
    From Word Embeddings To Document Distances
    The International Conference on Machine Learning (ICML), 2015
    http://mkusner.github.io/publications/WMD.pdf
    
    """
    def __init__(self, W_embed, n_neighbors_try=None, scoring=None, cv=3,
                 n_jobs=1, verbose=False):
        self.cv = cv
        self.n_neighbors_try = n_neighbors_try
        self.scoring = scoring
        super(WordMoversKNNCV, self).__init__(W_embed,
                                              n_neighbors=None,
                                              n_jobs=n_jobs,
                                              verbose=verbose)

    def fit(self, X, y):
        """Fit KNN model by choosing the best `n_neighbors`.
        
        Parameters
        -----------
        X : scipy.sparse matrix, (n_samples, vocab_size)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target
        """
        if self.n_neighbors_try is None:
            n_neighbors_try = range(1, 6)
        else:
            n_neighbors_try = self.n_neighbors_try

        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)

        cv = check_cv(self.cv, X, y)
        knn = KNeighborsClassifier(metric='precomputed', algorithm='brute')
        scorer = check_scoring(knn, scoring=self.scoring)

        scores = []
        for train_ix, test_ix in cv:
            dist = self._pairwise_wmd(X[test_ix], X[train_ix])
            knn.fit(X[train_ix], y[train_ix])
            scores.append([
                scorer(knn.set_params(n_neighbors=k), dist, y[test_ix])
                for k in n_neighbors_try
            ])
        scores = np.array(scores)
        self.cv_scores_ = scores

        best_k_ix = np.argmax(np.mean(scores, axis=0))
        best_k = n_neighbors_try[best_k_ix]
        self.n_neighbors = self.n_neighbors_ = best_k

        return super(WordMoversKNNCV, self).fit(X, y)


# Let's see how well it performs.

# In[100]:

test_tokenized = test_data.apply(lambda r: w2v_tokenize_text(r['plot']), axis=1).values
train_tokenized = train_data.apply(lambda r: w2v_tokenize_text(r['plot']), axis=1).values

flat_train_tokenized = [item for sublist in train_tokenized for item in sublist]
flat_test_tokenized = [item for sublist in test_tokenized for item in sublist]


# To speed up performance we focus only on the words that are both in Google News model and in our dataset.

# In[ ]:

# the word2vec model was loaded with strings as byte-arrays so need to convert
def convert_to_vocab_bytes(s):
     return bytes("b'" + s + "'", encoding='utf-8')    


# In[ ]:

vect = CountVectorizer(stop_words="english").fit(flat_train_tokenized)
common = [word for word in vect.get_feature_names() if convert_to_vocab_bytes(word) in vocab_dict]
W_common = W[[vocab_dict[ convert_to_vocab_bytes(w)] for w in common]]


# The mass of the pile of mud in the Word Mover's Distance is determined by how many times a word appears in a document. So we need a CountVectorizer.

# In[ ]:

test_tokenized[0]


# In[ ]:

vect = CountVectorizer(vocabulary=common, dtype=np.double)
X_train = vect.fit_transform(train_tokenized)
X_test = vect.transform(test_tokenized)


# Let's train the model.

# In[ ]:

knn = WordMoversKNN(n_neighbors=1,W_embed=W_common, verbose=5, n_jobs=7)


# In[ ]:

knn.fit(X_train, train_data['tag'])


# __Warning__: 10 minutes runtime on 7 cores

# In[ ]:

predicted = knn.predict(X_test)


# Only 2% above the naive baseline unfortunately. WMD achieves good results on sentiment analysis in the published paper. Maybe it works better for sentiment than for topic classification that we use it here. Or maybe preprocessing can be tuned here. It is hard to debug a black box method!

# In[ ]:

evaluate_prediction(predicted, test_data['tag'])


# # Conclusion

# Above we shown how to run 'hello-world' in 7 different document classification techniques. It is just a beginning of exploration of their features... There are a lot of parameters that can be tuned to get the best possible results out of them. The 'hello-world' run is in no way an indication of their best peformance. The goal of this tutorial is to show the API so you can start tuning them yourself.

# Out of the box "no tuning" accuracy of bag of words is not far behind more advanced techniques. 
# Tune them and the pre-processing for them well first and only then reach for more advanced methods if more accuracy is absolutely needed.

