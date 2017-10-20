# -*- coding: utf-8 -*-

def main(Params):
    
    with open(Params['OUTPUT-folder'] + '/target.txt','r') as f:
        Y = f.read()
           
    Y = [float(y) for y in Y]
        
    return Y