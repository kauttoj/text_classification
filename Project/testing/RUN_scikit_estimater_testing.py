# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:03:48 2017

@author: JanneK
"""
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import scikit_estimater_testing
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":        
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    import numpy as np
    X=np.random.randn(500,300)
    Y=np.random.randint(1,3,500).ravel() 
    PIPE = Pipeline([("PREPARATOR",scikit_estimater_testing.data_preparator()),("CLASSIFIER",LogisticRegression())])
    grid_search = GridSearchCV(PIPE,{'PREPARATOR__n_components':[5,10,20,40,50,60,70,80,90,110]},verbose=1,n_jobs=1,cv=8)
    grid_search.fit(X,Y)