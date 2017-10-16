# -*- coding: utf-8 -*-

# Function to lemmatize finnish text using online tool http://demo.seco.tkk.fi/las/
# useful for short texts, not for very large corpuses!

from urllib.request import urlopen
from urllib.parse import quote
from urllib.parse import unquote
import numpy as np
import re
import time


# orig_strings = corpus with words separated by space OR list of strings
# NOTE: text should contain nothing else than words, no punctuations!
def lemmatize(orig_strings):
    
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
    '''
    f = open('LEMMA_TEMPFILE.txt', 'w', encoding='utf8')
    for i in strings:
        f.write(i + '\n')  # python will convert \n to os.linesep
    f.close()  # you can omit in most cases as the destructor will call it
    
    subprocess.call('hfst-lookup fi-analysis.hfst.ol -n 1 -I LEMMA_TEMPFILE.txt', shell=True)
    
    p = subprocess.Popen('hfst-lookup fi-analysis.hfst.ol -n 1 -I LEMMA_TEMPFILE.txt', shell=False,bufsize=1,
                     stdin=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     stdout=subprocess.PIPE)       
    
    kk=0
    for k,line in enumerate(iter(p.stdout.readline, b'')):
        if k%2==0:            
            ss = line.split()
            strings[kk] = ss[1].decode('utf-8', 'ignore')
            strings[kk].replace('#','')
            kk += 1
    '''        

'''
mytest = 'Tämä kysymys nousi esille juuri ennen Venäjän ja Valko-Venäjän yhteistä Zapad 2017- sotaharjoitusta kun Grybauskaite antoi erikoishaastattelun Delfi-uutissivustolle Ilta-Sanomat on parhaillaan Vilnassa seuraamassa Liettuan varautumista Zapadiin ja Grybauskaiten haastattelu on yksi maan tuoreimmista virallisista viesteistä Venäjän suuntaan Grybauskaiten mukaan Putin antoi hänelle reilut seitsemän vuotta sitten suorasukaisen listan vaatimuksia jotka Liettuan pitäisi täyttää'
mytest = mytest*20
result = parse_text(mytest)
'''


'''
import subprocess
import json
import pprint
import websocket
from websocket import create_connection

websocket.enableTrace(True)
ws = create_connection('http://demo.seco.tkk.fi/las/baseformWS')

result = ws.recv()
print('Result: {}'.format(result))

result = ws.recv()
print('Result: {}'.format(result))

ws.send(json.dumps([json.dumps({'msg': 'connect', 'version': '1', 'support': ['1', 'pre2', 'pre1']})]))
result = ws.recv()
print('Result: {}'.format(result))

#import json

#PATH = 'D:/JanneK/Documents/git_repos/text_classification/hfst/'
'''    

