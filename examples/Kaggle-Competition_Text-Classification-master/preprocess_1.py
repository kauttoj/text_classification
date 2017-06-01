
''' Preprocess the documents and query to get tokens '''

import re

''' Function that removes the SGML tags'''
def removeSGML(content):
    newContent = re.sub(r'<.*?>','',content)
    return newContent

''' Function that tokenizes the text'''
def tokenizeText(line):
    line = line.lower() # convert all words into lowercase
    list_line = re.split('\s+',line)
            
    for ind in range(0, len(list_line)):
        remove_punct = [")", "(", "?", ":", "/", ".", ";"]
        for punct in remove_punct:
            if  punct in list_line[ind]:
                list_line[ind] = list_line[ind].replace(punct, "")
        
        item = list_line[ind]        
        if "," in list_line[ind]:  # Deal with comma in numbers
            idx = list_line[ind].find(",", 0, len(list_line[ind]))
            if idx==0 or idx==len(list_line[ind])-1:
                list_line[ind] = list_line[ind].replace(",", "")
            elif (item[idx-1] in "0123456789") and (item[idx+1] in "0123456789"):
                list_line[ind] = list_line[ind].replace(",", ",")
            else:
                list_line[ind] = list_line[ind].replace(",", " ") 
        # Assume that the comma in numbers will not appear at the fist or last element of the string
                       
        if "i'm" in list_line[ind]:
            list_line[ind] = list_line[ind].replace("i'm", "i am")
        
        if "we've" in list_line[ind]:
            list_line[ind] = list_line[ind].replace("'ve", " have")
        
        if "n't" in list_line[ind]:
            list_line[ind] = list_line[ind].replace("'n't", " not")
            
            
        if "'s" in list_line[ind]:
            if "s"==item[-1]:
                list_line[ind] = list_line[ind].replace("'s", " 's")
        elif "s'" in list_line[ind]:
            if "'"==item[-1]:
                list_line[ind] = list_line[ind].replace("s'", "s 's")
            
    list_line = ' '.join(list_line)
    list_line = re.split('\s+',list_line)

    for ind in range(0, len(list_line)):
        if "-" in list_line[ind]:
            list_line[ind] = list_line[ind].replace("-", " ")
    
    for word in list_line:
        if word=='.' or word=='..' or word=='' or word=='':
            list_line.remove(word)        
    
    return list_line

''' Function that removes stopwords'''
def removeStopwords(list_tokens):
    # read stopwords from the file "stopwords.txt" into a list
    f_stopwords = open("stopwords.txt",'r')
    stopwords=f_stopwords.read()
    list_stopwords = re.split('\s+',stopwords)
    f_stopwords.close()
    
    new_list_tokens = []
    for token in list_tokens:
        if token not in list_stopwords:
            new_list_tokens.append(token)
    return new_list_tokens 

''' Functions that stems the words'''
def stemWords(list_tokens):
    stem_list_tokens = []
    
    import porterStemmer
    stemObj = porterStemmer.PorterStemmer()
    
    for idx in range(0, len(list_tokens)-1):
        list_tokens[idx] = list_tokens[idx].lower()
        stem_tokens = stemObj.stem(list_tokens[idx], 0, len(list_tokens[idx])-1)
        stem_list_tokens.append(stem_tokens) 
    return stem_list_tokens 

''' The end '''

