# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:05:37 2017

@author: JanneK

Pipeline for fake text project
"""
Params = {}

Params['DoLemmatize'] = 1
Params['DoRemoveStops'] = 1
Params['DoTagging'] = 1
Params['UseCustomFeatures'] = 1
Params['Do_TFIDF'] = 1
Params['CV_folds'] = 10
Params['UseEmbedding'] = 1 #

Params['TargetType'] = 'regression'
#Params['TargetType'] = 'classification_binary'
#Params['TargetType'] = 'classification_multilabel'

Params['UseEmbedding'] = 1 #

Params['Algorithm'] = 'LSTM'
#Params['Algorithm'] = 'SVM'
#Params['Algorithm'] = 'NaiveBayes'
#Params['Algorithm'] = 'Vowpal'
#Params['Algorithm'] = 'Fasttext'
#Params['Algorithm'] = 'RandomForest'
Params['Algorithm'] = 'Logistic'

Params['INPUT_folder'] = ''
Params['OUTPUT_folder'] = ''

import Preprocessor
import Surveydata

X = Preprocessor.main(Params)
Y = Surveydata.main(Params)

if Params['Algorithm'] is 'LSTM':

    import RNNModels
    results = RNNModels.main(Params,X)
    
elif Params['Algorithm'] is 'Vowpal':
    
    raise('not yet implemented!')

elif Params['Algorithm'] is 'Fasttext':
    
    raise('not yet implemented!')

else:
    
    import StandardModels
    results = StandardModels.main(Params,X)
    
import Resultplotter
Resultplotter.main(Params,results)

