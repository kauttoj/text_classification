'''
Read the training dataset and get thr training data for classification
Name: Wei Xu
Uniquename: weixu '''
import os
import sys
#print sys.path
#currentPath = os.getcwd()
#sys.path.append(currentPath + "/sklearn")
#print sys.path


import glob
import numpy as np
import vectorization
import stratified_split
#import sklearn
#from sklearn.cross_validation import StratifiedKFold 
#from scikit_learn.feature_extraction.text import TfidfVectorizer
#from scikit_learn.svm.classes import SVC

# Get training data and store in the list,
# each element of the list will be the training document string
file_corpus = []
file_folder = "kaggle.training/"
filenames = os.listdir(file_folder)
#filenames = filenames[1:100] + filenames[24000:24100]
doc_num = len(filenames)

# Read the label of the file using its filename
# Convert the string label into numeric value: joke 1 and mix -1
# Read in all the documents into a lis of strings
file_label = np.zeros((doc_num), dtype = np.int)
for ind in range(0, doc_num):
    filename = filenames[ind]
    if "joke" in filename:
        file_label[ind] = 1
    elif "mix" in filename:
	file_label[ind] = -1
    f_file = open(file_folder + filename)
    file_string = f_file.read()
    file_corpus.append(file_string)
    f_file.close()
# Split the dataset into training and test dataset based on stratified splitting method
train_indices, test_indices = stratified_split.stratified_split(file_label, 0.8)
train_corpus = test_corpus = []
train_label = test_label = []
for i in range(0, len(train_indices)):
    if train_indices[i] == True:
        train_corpus.append(file_corpus[i])
	train_label.append(file_label[i])
    else:
        test_corpus.append(file_corpus[i])
	test_label.append(file_label[i])
# Vectorize the training documents into numeric vectors
# Return necessary results about training docs' tokens and the corresponding idf value
train_tfidf_mat, train_token, idf_token = vectorization.tfidfVectorize(train_corpus)
train_label = np.array(train_label)
#print train_tfidf_mat[0:3,:]
#print "\n\n"

np.savetxt("train_data.out", train_tfidf_mat)
np.savetxt("train_label.out", train_label)


# Vectorize the test doc using results from training dataset
test_tfidf_mat = np.zeros((len(test_label), len(train_token)))
for i in range(0, len(test_label)):
    test_doc = test_corpus[i]
    test_token = vectorization.processDoc(test_doc)
    test_doc_vector = np.zeros(len(train_token))
    for j in range(0, len(train_token)):
	token = train_token[j]
        if token in test_token:
            tf = test_token.count(token)
	    test_doc_vector[j] = tf*idf_token[j] 
        else:
            test_doc_vector[j] = 0.0

#print test_tfidf_mat[0:3,:]
test_label = np.array(test_label)
np.savetxt("test_data.out", test_tfidf_mat)
np.savetxt("test_label.out", test_label) 
