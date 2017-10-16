# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:52:01 2017

@author: Jannek
"""
from pandas import DataFrame
import os
import numpy
import re
import nltk
from nltk.tokenize import WhitespaceTokenizer
import string
#from nltk.corpus import wordnet as wn
#from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
#from nltk.stem import WordNetLemmatizer

translate_table = dict((ord(char), ' ') for char in string.punctuation)   

#wordnet_lemmatizer = WordNetLemmatizer('finnish')
#snowball_stemmer = SnowballStemmer('finnish')
stopword_list = nltk.corpus.stopwords.words('finnish')

def remove_special_characters(text):
    tokens = tokenize_text(text)
    punc = string.punctuation
    #punc=punc.replace('.','')
    #punc=punc.replace('?','')
    #punc=punc.replace('!','')
    pattern = re.compile('[{}]'.format(re.escape(punc)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
        
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text   

def normalize_corpus(corpus):
    
    normalized_corpus = corpus.copy()    
    for i,val in normalized_corpus.iterrows():
        text = val['text']
        text = remove_special_characters(text)
        #text = remove_stopwords(text)
        #text = stemmer(text)        
        normalized_corpus.set_value(i,'text',text)
            
    return normalized_corpus

def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                past_header, lines = False, []
                f = open(file_path, encoding='utf-8')
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

def main(Params):
    """
    DATA
    """
    SOURCES=[
        #('C:/Users/Jannek/Documents/git_repos/text_classification/data/bbs_sport/football','FOOTBALL'),
        #('C:/Users/Jannek/Documents/git_repos/text_classification/data/bbs_sport/rugby','RUGBY')                    
        #('C:/Users/Jannek/Documents/git_repos/text_classification/data/bbc/business','BUSINESS'),
        #('C:/Users/Jannek/Documents/git_repos/text_classification/data/bbc/politics','POLITICS'),
        #('C:/Users/Jannek/Documents/git_repos/text_classification/data/bbc/tech','TECH')        
        #(r'/media/jannek/Data/JanneK/Documents/git_repos/text_classification/data/TALOUS','TALOUS'), 
        #(r'/media/jannek/Data/JanneK/Documents/git_repos/text_classification/data/TERVEYS','TERVEYS')    
		(r'D:/JanneK/Documents/git_repos/text_classification/data/TALOUS','TALOUS'), 
		(r'D:/JanneK/Documents/git_repos/text_classification/data/TERVEYS','TERVEYS')  
    ]
    
    data = DataFrame({'text': [], 'mylabel': []})
    for path, classification in SOURCES:
        data = data.append(build_data_frame(path, classification))
        
    data = normalize_corpus(data)
    
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


# Function to lemmatize finnish text using online tool http://demo.seco.tkk.fi/las/
# useful for short texts, not for very large corpuses!




# orig_strings = corpus with words separated by space OR list of strings
# NOTE: text should contain nothing else than words, no punctuations!
def lemmatize(orig_strings):
    
    from urllib.request import urlopen
    from urllib.parse import quote
    from urllib.parse import unquote
    import time    
    
    if len(orig_strings)==0:
        orig_strings = 'Tämä kysymys nousi esille juuri ennen Venäjän ja Valko-Venäjän yhteistä Zapad 2017- sotaharjoitusta kun Grybauskaite antoi erikoishaastattelun Delfi-uutissivustolle Ilta-Sanomat on parhaillaan Vilnassa seuraamassa Liettuan varautumista Zapadiin ja Grybauskaiten haastattelu on yksi maan tuoreimmista virallisista viesteistä Venäjän suuntaan Grybauskaiten mukaan Putin antoi hänelle reilut seitsemän vuotta sitten suorasukaisen listan vaatimuksia jotka Liettuan pitäisi täyttää'
        orig_strings = orig_strings*50
        print('!!!! Empty input given, using test string !!!!\n')

    if type(orig_strings) is not list:                
        strings = orig_strings.split(' ')
    else:
        strings = orig_strings
        
    N_orig = len(strings)
    
    start = time.time()
    
    strings = [quote(a) for a in strings]
    strings = '+'.join(strings)        
    MAX_LENGTH = 4000
    res=''
    k=0
    while len(strings)>0:
        inds = np.array([m.start() for m in re.finditer('\+',strings)])
        if inds[-1] > MAX_LENGTH:
            ind = inds[np.where(inds<MAX_LENGTH)[0][-1]]
        else:
            ind = len(strings)
        s1 = strings[0:ind]
        strings = strings[(ind+1):]
        s2 = 'http://demo.seco.tkk.fi/las/baseform?text='+s1+'&locale=fi'        
        f = urlopen(s2)
        s2 = f.read()    
        s2 = s2.decode('utf-8')    
        s2 = s2[1:-1]        
        res = res + ' ' + s2
        k+=1
    
    res=res[1:]
    
    N_new = len(res.split(' '))
    
    assert(N_orig == N_new)

    elapsed = time.time() - start
    print('Lemmatizer: Document with %i words lemmatized in %0.1f seconds (using %i parts)' % (N_orig,elapsed,k))
    
    return res

