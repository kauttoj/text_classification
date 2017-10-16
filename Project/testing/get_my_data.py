# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:52:01 2017

@author: Jannek
"""
from pandas import DataFrame
import os
import numpy

def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                past_header, lines = False, []
                f = open(file_path, encoding="latin-1")
                for line in f:
                    if past_header and len(line)>0 and line is not '\n':
                        line=line.rstrip()
                        lines.append(line)
                    else:
                        past_header = True                        
                f.close()
                content = ' '.join(lines)
                yield file_path, content


def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        if len(text)>=1000:
            rows.append({'text': text, 'mylabel': classification})
            index.append(file_name)
    if len(rows)<50:
        raise('ERROR: less than 50 samples!')
    data_frame = DataFrame(rows, index=index)
    return data_frame

def shuffle(df, n=1, axis=0):     
    print('\n\n!!!!! WARNING: shuffling data for testing purposes !!!!!\n')
    df = df.copy()
    for _ in range(n):
        df.apply(numpy.random.shuffle, axis=axis)
    return df

def getdata():
    """
    DATA
    """
    SOURCES=[
        #('C:/Users/Jannek/Documents/git_repos/text_classification/data/bbs_sport/football','FOOTBALL'),
        #('C:/Users/Jannek/Documents/git_repos/text_classification/data/bbs_sport/rugby','RUGBY')                    
        #('C:/Users/Jannek/Documents/git_repos/text_classification/data/bbc/business','BUSINESS'),
        ('C:/Users/Jannek/Documents/git_repos/text_classification/data/bbc/politics','POLITICS'),
        ('C:/Users/Jannek/Documents/git_repos/text_classification/data/bbc/tech','TECH')        
    ]
    
    data = DataFrame({'text': [], 'mylabel': []})
    for path, classification in SOURCES:
        data = data.append(build_data_frame(path, classification))
    
    data = data.reindex(numpy.random.permutation(data.index))
    
    labels = data.mylabel.unique()
    counts=[-1]*len(labels)
    for i in range(len(counts)):
        counts[i]=len(data[data.mylabel==labels[i]])
        
    M=min(counts)
    for i in range(len(counts)):
        ind=data[data.mylabel==labels[i]].index
        data=data.drop(ind[M:])
    
    print('\n-- Total',len(counts),'labels with',M,'samples each')
    
    #data = shuffle(data)
    
    return data