
import os
import collections
import smart_open
import random
import pickle
import gensim
from gensim.models.doc2vec import TaggedDocument

def ranking(model,train_corpus):
    ranks = []
    second_ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])

    print(collections.Counter(ranks))

# Set file names for train and test data
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

train_corpus = train_corpus[0:200]

model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=200)

model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

ranking(model,train_corpus)


datafile = r'D:/JanneK/Documents/git_repos/text_classification/Project/results' + r'/preprocessed_data.pickle'

(FEAT,Params)=pickle.load(open( datafile, "rb" ))

corpus = []
tokens=[]
for k,doc in enumerate(FEAT):
    X=[]
    for i,x in enumerate(doc['tokens']):
        if doc['is_number'][i]==0 and doc['is_char'][i]==1:
            X.append(x)
    tokens.append(X)
    X =  TaggedDocument(X,[k])
    corpus.append(X)

train_corpus = corpus[:-5]
test_corpus = corpus[-5:]

model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=500)

model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

ranking(model,train_corpus)
