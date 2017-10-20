# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:52:01 2017

@author: Jannek
"""

import os
import numpy as np
import re
from nltk import wordpunct_tokenize
from nltk import sent_tokenize
import string
import nltk
import math

stopword_list = nltk.corpus.stopwords.words('finnish')

import sys

OMORFI_PATH = r'/home/jannek/Downloads/omorfi/src/python'
sys.path.append(OMORFI_PATH)

from omorfi.omorfi import Omorfi

omorfi = Omorfi()
omorfi.load_from_dir()

USE_SECO_LEMMER = 1

def tokenize(document,Params):
    
    sentence_id = []
    is_char = []
    is_number = []
    tokens = []
    skip_space = []
        
    for id in Params['CUSTOM_TAGS'].keys():
        l=len(id)
        m = len(document.split())/2
        k=0
        while 1:
            k+=1
            if k>m:
                raise('Too many custom tags - BUG!!!')
            s1s = document.find('<%s>' % id)
            s1e = s1s+l+2
            if s1s>-1:
                s2s = document.find('</%s>' % id)
                if s2s==-1:
                    raise('ID %s does not have ending!!' % id)
                s2e = s2s+l+3                                            
                if Params['CUSTOMTagging']==1:
                    document = document[0:s1s] + Params['CUSTOM_TAGS'][id] + document[s2e:] 
                else:
                    document = document[0:s1s] + document[s1e:s2s] + document[s2e:]                                     
            else:
                break
    
    k=0
    n_tokens=0
    # Break the document into sentences
    for sent in sent_tokenize(document):
       
        n_tokens_sent=0
        k+=1
        # Break the sentence into part of speech tagged tokens
        sent_tokens = sent.split(' ')                
        for i,token_piece in enumerate(sent_tokens):
            token_piece = wordpunct_tokenize(token_piece)
            
            for j,token in enumerate(token_piece):
                
                if j>0:
                    skip_space.append(n_tokens)
                    
                n_tokens+=1
                n_tokens_sent+=1
                # Apply preprocessing to the token
                
                # token = token.lower()
                sentence_id.append(k)            
                tokens.append(token)
    
                # If punctuation, ignore token and continue            
                if all(char in string.punctuation for char in token) or all(not(char.isalpha()) for char in token) or len(token)<2:
                    is_char.append(0)
                else:
                    is_char.append(1)
                    
                try:
                    _ = np.float(token)   
                    is_number.append(1)
                except:       
                    is_number.append(0)                      
                    
    tokens_raw = tokens.copy()                    
    tokens_raw = [t.lower() for t in tokens_raw]
                    
    document_rec=''
    if len(skip_space)>5:
        k=0
        for i,tok in enumerate(tokens):
            if i==0 or (k<len(skip_space) and skip_space[k]==i):
                document_rec+=tok
                if i>0:
                    k+=1
            else:
                document_rec+=' ' + tok    
                
    if len(document_rec)!=len(document):
        print('    !!! White space reconstruction FAILED! Original length %i, reco length %i !!!' % (len(document_rec),len(document)))
            # Lemmatize the token and yield
    return {'tokens':tokens,'sentence_id':sentence_id,'is_char':is_char,'is_number':is_number,'skip_space':skip_space,'tokens_raw':tokens_raw}

def remove_text_inside_brackets(text, brackets="{}()[]"):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)        
    
def lemmatize_omorfi(tokens_in,dictionary_lookup=0):

    is_word = tokens_in['is_char']
    tokens = tokens_in['tokens']    
    in_dictionary = [0]*len(tokens)

    for i,token in enumerate(tokens):        
        if is_word[i]==1:
            res = omorfi.lemmatise(token)
            if dictionary_lookup==0:
                tokens[i]=remove_text_inside_brackets(res[-1][0])
            if float(res[-1][1]) != math.inf:
                in_dictionary[i]=1
                
    if dictionary_lookup==1:
        return in_dictionary
    
# Function to lemmatize finnish text using online tool http://demo.seco.tkk.fi/las/
# useful for short texts, not for very large corpuses!

# orig_strings = corpus with words separated by space OR list of strings
# NOTE: text should contain nothing else than words, no punctuations!
def lemmatize_seco(tokens_in):
    
    tokens = tokens_in['tokens']
    is_word = tokens_in['is_char']
    
    from urllib.request import urlopen
    from urllib.parse import quote
    from urllib.parse import unquote
    import time    
    
#    if len(tokens)==0:
#        orig_strings = 'Tämä kysymys nousi esille juuri ennen Venäjän ja Valko-Venäjän yhteistä Zapad 2017- sotaharjoitusta kun Grybauskaite antoi erikoishaastattelun Delfi-uutissivustolle Ilta-Sanomat on parhaillaan Vilnassa seuraamassa Liettuan varautumista Zapadiin ja Grybauskaiten haastattelu on yksi maan tuoreimmista virallisista viesteistä Venäjän suuntaan Grybauskaiten mukaan Putin antoi hänelle reilut seitsemän vuotta sitten suorasukaisen listan vaatimuksia jotka Liettuan pitäisi täyttää'
#        orig_strings = orig_strings*50
#        print('!!!! Empty input given, using test string !!!!\n')
#
#    if type(tokens) is not list:                
#        strings = tokens.split(' ')
#    else:
#        strings = tokens
        
    N_orig = len(tokens)

    word_inds = []
    strings = []
    for i in range(0,N_orig):
        if is_word[i]==1:                
            strings.append(quote(tokens[i]))
            word_inds.append(i)               
    
    start = time.time()
               
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
    res=res.split()
    
    new_tokens = tokens.copy()
    for i_ind,i_real in enumerate(word_inds):
        new_tokens[i_real]=res[i_ind]
        
    N_new = len(new_tokens)
    
    assert(N_orig == N_new)

    elapsed = time.time() - start
    print('SeCo lemmatizer: Document with %i words lemmatized in %0.1f seconds (using %i parts)' % (N_orig,elapsed,k))
    
    tokens_in['tokens']=new_tokens
    tokens_in['in_dictionary'] = lemmatize_omorfi(tokens_in,dictionary_lookup=1)
    
    assert(len(tokens_in['tokens'])==len(tokens_in['is_char'])==len(tokens_in['in_dictionary']))
        
def mark_stopwords(tokens):        
    is_stopword=[0]*len(tokens['tokens'])
    for i in range(0,len(tokens['tokens'])):
        if tokens['tokens'][i] in stopword_list:
            is_stopword[i]=1
    tokens['is_stopword']=is_stopword
    
def pos_tagging(tokens_in):        
    tags = []
    for i,tok in enumerate(tokens_in['tokens']):
        if tokens_in['is_char'][i]==0:
            tags.append('') 
            continue        
        res = omorfi.analyse(tok)
        s1=-1
        for elem in res:
            if elem[0][9:(9+len(tok))]==tok:
                res = elem[0]
                s1=res.find('UPOS=')   
                break                             
        if s1==-1:
            tags.append('')       
        else:
            s2=res[s1:].find(']')
            tags.append(res[(s1+5):(s1+s2)])
            
    tokens_in['POS_tags']=tags

def main(Params):
                
    text=[]
    text_meta=[]
    
    Y = []
    if len(Params['INPUT-folder'])==0:
        text = ['kirjailija kirjoitti hienon kirjan. Hän on erinomainen kirjailija. Maalari maalasi hienoa taloa, kunnes kuoli pois. muut maalarit jatkoivat työtä. Kela pyytää realisoimaan rahastot, koska tulkitsee lapsille säästetyn noin tuhannen euron summan lasten eli perheen tuloiksi. Kelan etuusjohtajan Anne Neimalan mukaan ratkaisu kuulostaa oikealta. Hän ei tunne tapauksen yksityiskohtia vaan kommentoi Kelan ratkaisuja yleisellä tasolla. – Kyseessä on viimesijainen tuki, ja siinä otetaan huomioon kaikki tulot ja omaisuus, joka on helposti realisoitavissa, Neimala sanoo IS:lle.','Neimalan mukaan summalle ei ole ylä- tai alarajaa. Päätöksissä huomioidaan kaikki helposti realisoitava omaisuus, kuten tilillä ja rahastoissa olevat varat.']
        text_meta = ['teksti A','teksti B']
        k=2
    else:
        #root, dir_names, file_names = os.walk(Params['INPUT-folder'])
        root = Params['INPUT-folder'] + '/'
        filenames = os.listdir(root)
        filenames = [(root+f) for f in filenames if os.path.isfile(root+f)]
        k=0
        for file in filenames:
            drive, path = os.path.splitdrive(file)
            path, filename = os.path.split(path)
            if filename[0:4]=='text':
                k+=1
                with open(file, encoding='utf-8') as f:
                    text.append(f.read())
                    text_meta.append(filename)
            elif filename[0:6]=='target':
                with open(file, encoding='utf-8') as f:
                    Y =f.read()
                    Y=Y.split()
                    Y = [float(y) for y in Y]                

    assert(len(Y)==len(text))
                         
    POS_tags = []
    
    data = []
    for i in range(0,len(text)):    
        
        print('---processing document %i' % (i+1))
        
        tokens = tokenize(text[i],Params)
        
        tokens['raw_text'] = text[i]
        tokens['meta_information'] = text_meta[i]
        tokens['target'] = Y[i]
        tokens['tokens_raw'] = tokens['tokens']    
        
        print('tokenization done')
        mark_stopwords(tokens)
        print('Stopwords removal done')
        if USE_SECO_LEMMER==1:
            lemmatize_seco(tokens)
        else:
            lemmatize_omorfi(tokens)
        print('Lemmatization done')
        pos_tagging(tokens)
        print('POS tagging done')        
        
        # as a final step, lower case all words
        for j in range(0,len(tokens['tokens'])):
            tokens['tokens'][j] = tokens['tokens'][j].lower()
            val = tokens['POS_tags'][j]
            if len(val)>0 and (val not in POS_tags):
                    POS_tags.append(val)
        
        data.append(tokens)
    
    Params['POS_TAGS'] = POS_tags
    Params['stopword_list'] = stopword_list
        
    print('\n-- Data summary: %i text processed\n' % len(data))
    
    
    #data = shuffle(data)
    
    return data
    

#Params = {}
#Params['CUSTOM_TAGS'] = {'MIES':'NIMI_MIES',
#               'NAINEN':'NIMI_NAINEN',
#               'NETTISIVU':'VIITE_NETTI',
#               'JULKAISU':'VIITE_TIEDEJULKAISU',
#               'YLIOPISTO':'NIMI_YLIOPISTO',
#               'YRITYS':'NIMI_YRITYS',
#               'TAULUKKO':'TAULUKKO'}
#Params['INPUT-folder'] = r'/media/jannek/Data/JanneK/Documents/git_repos/text_classification/data/pikkudata'
#data = main(Params)

#
#def read_files(path):
#    for root, dir_names, file_names in os.walk(path):
#        for path in dir_names:
#            read_files(os.path.join(root, path))
#        for file_name in file_names:
#            file_path = os.path.join(root, file_name)
#            if os.path.isfile(file_path):
#                past_header, lines = False, []
#                f = open(file_path, encoding='utf-8')
#                for line in f:
#                    if past_header and len(line)>0 and line is not '\n':
#                        line=line.rstrip()
#                        lines.append(line)
#                    else:
#                        past_header = True                        
#                f.close()
#                content = ' '.join(lines)
#                yield file_path, content
