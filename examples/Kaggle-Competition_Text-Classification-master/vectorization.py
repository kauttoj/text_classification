
'''
Implement function to vectorize the given document using VectorSpace Model
Name: Wei Xu
Uniquename: weixu '''


import preprocess
import math
import numpy as np

def processDoc(doc_string):
    token_doc = preprocess.stemWords(preprocess.removeStopwords(preprocess.tokenizeText(doc_string)))
    return token_doc

''' Function to count the occurance frequency of the tokens in each document
    This function will be called by function tfIdfVectorize
    Return a document-term matrix, in which each element indicates the occurence count of 
    the column token in the row document '''
def countTokenFreq(raw_documents):
    # preprocess the doc content to get tokens
    token_document = []
    token_voc = []
    for contentDoc in raw_documents:
        token_Doc = processDoc(contentDoc)
        token_document.append(token_Doc)
        token_voc.extend(list(set(token_Doc)))
    token_voc = list(set(token_voc))
    doc_num = len(token_document)
    token_num = len(token_voc)
    doc_term_mat = np.zeros((doc_num, token_num))
    for i in range(0, doc_num):
        for j in range(0, token_num):
            doc_term_mat[i][j] = token_document[i].count(token_voc[j])
            #print doc_term_mat[i][j]
    return token_voc, doc_term_mat

''' Function to compute the document_term matrix based on tf-idf weighting
    Each row is one document
    Each column is one token
    Return the wieghting document-term matrix '''
def tfidfVectorize(raw_documents):
    token_voc, doc_term_mat = countTokenFreq(raw_documents)
    doc_num = doc_term_mat.shape[0]
    token_num = doc_term_mat.shape[1]
    idf_arr = np.zeros(token_num)
    for i in range(0, token_num ):
	df = 0
	for j in range(0,doc_num):
            if doc_term_mat[j,i] != 0:
                df += 1
	idf_arr[i] = math.log10(doc_num*1.0/df)

    weight_doc_term = np.zeros((doc_num, token_num))
    for m in range(0, token_num):
	for n in range(0, doc_num):
	    tf = doc_term_mat[j, i]
	    weight_doc_term[j, i] = tf*idf_arr[m]
    return weight_doc_term, token_voc, idf_arr	                 

