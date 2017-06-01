
'''
First method

Implements the Naive Bayes text classifier for Kaggle Classification Challenge

Note: I use all 26000 documents in the "kaggle.training" folder as training dataset, thus the code may require several minutes to run to get the result. It produce a csv file containing the test file and the corresponding classified label

Name: Wei Xu
Uniquename: weixu '''

import os
import sys
import math
import csv
import numpy as np
import preprocess

def processDoc(doc_string):
    # Preprocess the doc strings
    doc_token = preprocess.tokenizeText(doc_string)
    #doc_token = preprocess.removeStopwords(doc_token)
    #doc_token = preprocess.stemWords(doc_token)
    return doc_token

''' Apply Naive Bayes algorithm to train 
    Train step '''
def trainNaiveBayes(train_corpus, train_label):
    train_data = []
    joke_data = []
    mix_data = []
    for i in range(0, len(train_corpus)):
        # Preprocess the doc strings
        doc_train = train_corpus[i]
        token_train = processDoc(doc_train)
        if train_label[i] == 1:
            joke_data.extend(token_train)
        else:
            mix_data.extend(token_train)
        train_data.extend(token_train)
    
    num_total = len(train_label)
    prob_joke = train_label.count(1) * 1.0/num_total
    prob_mix = 1-prob_joke
    class_prob = [prob_joke, prob_mix]
    
    train_data_voc = list(set(train_data))
    num_voc = len(train_data_voc)
    num_word_joke = len(joke_data)
    num_word_mix = len(mix_data)
    word_prob_dic = {}
    for word in train_data_voc:
        probJoke = (joke_data.count(word)+1)*1.0/(num_word_joke + num_voc)
        probMix = (mix_data.count(word)+1)*1.0/(num_word_mix + num_voc)
        word_prob_dic.update({word: [probJoke, probMix]})
    num_paras= [num_word_joke, num_word_mix, num_voc]
    return class_prob, word_prob_dic, num_paras

''' Test step '''
def testNaiveBayes(test_doc, class_prob, word_prob_dic, num_paras):
    test_token = processDoc(test_doc)
    probBeJoke = 0.0
    probBeMix = 0.0
    for word in test_token:
        if word not in word_prob_dic.keys():
            probJoke = 1.0/(num_paras[0]+num_paras[2])
            probMix = 1.0/(num_paras[1]+num_paras[2])
        else:
            probJoke = word_prob_dic[word][0]
            probMix = word_prob_dic[word][1]
        probBeJoke += math.log(probJoke)
        probBeMix += math.log(probMix)
    probBeJoke = math.log(class_prob[0]) + probBeJoke
    probBeMix = math.log(class_prob[1]) + probBeMix
    if probBeJoke > probBeMix:
        predict_label = 1
    else:
        predict_label = -1
    return predict_label

'''The main program '''
''' Get training data and store in the list,
    each element of the list will be the training document string '''
train_corpus = []
train_folder = "kaggle.training/"
trainNames = os.listdir(train_folder)
doc_num = len(trainNames)

''' Read the label of the file using its filename
    Convert the string label into numeric value: joke 1 and mix -1
    Read in all the documents into a lis of strings '''
train_label = np.zeros((doc_num), dtype = np.int)
for ind in range(0, doc_num):
    filename = trainNames[ind]
    if "joke" in filename:
        train_label[ind] = 1
    elif "mix" in filename:
	train_label[ind] = -1
    f_file = open(train_folder + filename)
    file_string = f_file.read()
    train_corpus.append(file_string)
    f_file.close()

train_label = list(train_label)
class_prob, word_prob_dic, num_paras = trainNaiveBayes(train_corpus, train_label)

''' Read in the test dataset '''
test_folder = str(sys.argv[1])
#test_folder = "kaggle.test/"
testNames = os.listdir(test_folder)
predict_result = {}
for j in range(0, len(testNames)):
    test_name = testNames[j]
    test_num = int(test_name[8:])
    
    f_test = open(test_folder + test_name)
    test_doc = f_test.read()
    predict_label = testNaiveBayes(test_doc, class_prob, word_prob_dic, num_paras)
    if predict_label == 1:
        label = "joke"
    else:
        label = "mix"
    predict_result.update({test_num: [test_name, label]})
sort_predict_result = sorted(predict_result.iteritems(), key = lambda d:d[0], reverse=False)
result = []
for item in sort_predict_result:
    result.append(item[1])

''' Write the prediction result to the csv file '''
with open("classify_result_naiveBayes.csv", 'w') as fp:
    csv_writer = csv.writer(fp, delimiter = ',')
    csv_writer.writerow(["File", "Class"])
    for row in result:
        csv_writer.writerow(row)
