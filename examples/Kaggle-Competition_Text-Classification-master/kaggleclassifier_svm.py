'''
Final method

Note: Before running this code, please run "module load python" in the Terminal so that scikit learn package can be used, or you will get errors

Implements the SVM text classifier for Kaggle Classification Challenge
Name: Wei Xu
Uniquename: weixu

Note: I use all 26,000 documents for training, and I use linear kernel for svm to classify

 '''

import os
import sys
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn import svm
from sklearn.grid_search import GridSearchCV

''' Read the label of the file using its filename
    Convert the string label into numeric value: joke 1 and mix -1
    Read in all the documents into a lis of strings '''
def get_data_label(filenames, train_folder):
    doc_num = len(filenames)
    file_corpus = []
    train_corpus = []
    file_label = np.zeros((doc_num), dtype = np.int)
    for ind in range(0, doc_num):
        filename = filenames[ind]
        f_file = open(train_folder + filename)
        file_string = f_file.read()
        if "joke" in filename:
            file_label[ind] = 1
        elif "mix" in filename:
        file_label[ind] = -1
        file_corpus.append(file_string)
        f_file.close()

    return file_corpus, file_label

'''The main Program '''
file_folder = "kaggle.training/"
filenames = os.listdir(file_folder)
train_corpus, train_label = get_data_label(filenames, file_folder)

''' Vectorize the training documents usinf tf-idf weighting '''    
vectorize = TfidfVectorizer(encoding='ISO-8859-1', sublinear_tf = True, max_df = 0.5, stop_words = 'english')
train_data = vectorize.fit_transform(train_corpus)

''' Get test documents and perform the same vectorizer to get test data '''
test_folder = str(sys.argv[1])
#test_folder = "kaggle.test/"
testNames = os.listdir(test_folder)
test_corpus = []
test_num = []
for j in range(0, len(testNames)):
    test_name = testNames[j]
    test_num.append(int(test_name[8:]))
    
    f_test = open(test_folder + test_name)
    test_doc = f_test.read()
    test_corpus.append(test_doc)

test_data = vectorize.transform(test_corpus)

''' Using SVM with Linear kernel to classify the test data
    I also use Grid search to find the best parameters, which takes around 20 minutes when training on 26,000
    documents, thus I only use default parameters here. Release the annotation if you want to run grid search.

    Note: I got better result when using grid search to find best parameters for SVM '''
cPara_range = [1.0]
#cPara_range = list(np.logspace(-2,2,10)) # release this annotation and kill the previous sentence to run grid search
parameters = {'C':cPara_range}
clf = svm.SVC(kernel = 'linear')
model_tunning = GridSearchCV(clf, param_grid = parameters)

model_tunning.fit(train_data, train_label)
predict_labels = model_tunning.predict(test_data)

''' Write prediction result into csv file '''
predict_result = {}
for ind in range(0, len(list(predict_labels))):
    predict_label = predict_labels[ind]
    if predict_label == 1:
        label = "joke"
    else:
        label = "mix"
    predict_result.update({test_num[ind]: [testNames[ind], label]})
sort_predict_result = sorted(predict_result.iteritems(), key = lambda d:d[0], reverse=False)
result = []
for item in sort_predict_result:
    result.append(item[1])

with open("classify_result_svm.csv", 'w') as fp:
    csv_writer = csv.writer(fp, delimiter = ',')
    csv_writer.writerow(["File", "Class"])
    for row in result:
        csv_writer.writerow(row)

