# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:05:37 2017

@author: JanneK

Pipeline for fake text project
"""

import pickle
import os

if __name__ == "__main__":        
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    SKIP_PRESTORED = 0
    
    Params = {}
    Params['CUSTOM_TAGS'] = {
        'MIES':'NIMI_MIES',
        'NAINEN':'NIMI_NAINEN',
        'NETTISIVU':'VIITE_NETTI',
        'JULKAISU':'VIITE_TIEDEJULKAISU',
        'YLIOPISTO':'NIMI_YLIOPISTO',
        'YRITYS':'NIMI_YRITYS',
        'TAULUKKO':'TAULUKKO'
       }
    
    # how to preprocess text before classification
    # Note: this only takes effect after analyzing raw text
    Params['CUSTOMTagging'] = 1
    Params['Lemmatize'] = 1
    Params['RemoveStops'] = 0
    Params['POSTagging'] = 1
    Params['RemovePunctuation'] = 1
    
    Params['UseCustomFeatures'] = 1
    Params['TF-IDFScaling'] = 1
    Params['n-gram'] = 2
    Params['WordSmoothing'] = 1
    
    Params['WordEmbedding'] = 0 # 0 = use BOW
    Params['Compression'] = 0
    
    # main test/train split degree
    Params['CV-folds'] = 20
    
    # How to treat target vector
    #Params['TargetType'] = 'regression'
    Params['TargetType'] = 'classification_binary'
    #Params['TargetType'] = 'classification_multilabel'
    
    ## BOW type (n-grams + custom features)
    #Params['Algorithm'] = 'SVM'
    #Params['Algorithm'] = 'NaiveBayes'
    #Params['Algorithm'] = 'RandomForest'
    Params['Algorithm'] = 'Logistic'
    #Params['Algorithm'] = 'SGD'
    #Params['Algorithm'] = 'ExtraTrees'
    #Params['Algorithm'] = 'Ensemble'
    
    ## precompiled external tools
    #Params['Algorithm'] = 'Vowpal'
    #Params['Algorithm'] = 'Fasttext'
    
    ## NN type (embedded words + custom features)
    #Params['Algorithm'] = 'RNN'
    
    
    Params['INPUT-folder'] = r'D:/JanneK/Documents/git_repos/text_classification/data/pikkudata'
    Params['OUTPUT-folder'] = r'D:/JanneK/Documents/git_repos/text_classification/Project/results'
    
    #Params['INPUT-folder'] = r'/media/jannek/Data/JanneK/Documents/git_repos/text_classification/data/pikkudata'
    #Params['OUTPUT-folder'] = r'/media/jannek/Data/JanneK/Documents/git_repos/text_classification/Project/results'
    
    #%% Start pipeline
    
    if Params['Algorithm'] == 'RNN':
        Params['WordEmbedding'] = 1
    
    datafile = Params['OUTPUT-folder'] + '/preprocessed_data.pickle'
    
    if not os.path.isfile(datafile) or SKIP_PRESTORED==1:
        import Preprocessor
        data = Preprocessor.main(Params)
        pickle.dump((data,Params), open( datafile, "wb" ))
    else:
        data,Params_loaded = pickle.load(open( datafile, "rb" ))
        Params['POS_TAGS'] = Params_loaded['POS_TAGS']
        Params['stopword_list'] = Params_loaded['stopword_list']
    
    import FeatureExtractor
    FEAT = FeatureExtractor.main(data,Params)
    
    if Params['Algorithm'] is 'LSTM':
        
        raise('not yet implemented!')
        #import RNNModels
        #results = RNNModels.main(Params,data)
        
    elif Params['Algorithm'] is 'Vowpal':
        
        raise('not yet implemented!')
    
    elif Params['Algorithm'] is 'Fasttext':
        
        raise('not yet implemented!')
    
    else:
        
        import StandardModels
        StandardModels.main(FEAT,Params)
    
#import Resultplotter
#Resultplotter.main(Params,results)

