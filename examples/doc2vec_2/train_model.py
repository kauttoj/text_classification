#python example to train doc2vec model (with or without pre-trained word embeddings)

train_corpus = "C:/Users/Jannek/Documents/git_repos/text_classification/examples/doc2vec_2/toy_data/train_docs.txt"
from gensim.models.wrappers import FastText
fT = FastText();
model=fT.train('C:/Users/Jannek/Documents/git_repos/text_classification/examples/doc2vec_2/fastText/bin/fastText',corpus_file=train_corpus, output_file=None, model='cbow', size=100, alpha=0.025, window=5, min_count=5,loss='ns', sample=1e-3, negative=5, iter=5, min_n=3, max_n=6, sorted_vocab=1, threads=2)
print(model['forests'])

import gensim.models as g
import logging

#doc2vec parameters
vector_size = 300
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0 #0 = dbow; 1 = dmpv
worker_count = 1 #number of parallel processes

#pretrained word embeddings
pretrained_emb = None #"toy_data/pretrained_word_embeddings.txt" #None if use without pretrained embeddings

#input corpus
train_corpus = "toy_data/train_docs.txt"

#output model
saved_path = "toy_data/model.bin"

#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#train doc2vec model
docs = g.doc2vec.TaggedLineDocument(train_corpus)
model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, iter=train_epoch)

#save model
model.save(saved_path)

#
#from subprocess import call
#import os
#a=os.system('C:/Users/Jannek/Documents/git_repos/text_classification/examples/doc2vec_2/fastText/bin/fasttext.exe')

