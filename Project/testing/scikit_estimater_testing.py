from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD

class data_preparator(BaseEstimator, TransformerMixin):  
    """An example of classifier"""

    def __init__(self, n_components=10):
        """
        Called when initializing the classifier
        """
        self.n_components=n_components
        self.decomposer = TruncatedSVD(n_components,n_iter=10)

    def fit(self, X, y=None):
        self.decomposer.fit(X)
        return self

    def transform(self, X):        
        return self.decomposer.transform(X)
        
    def fit_transform(self,X, y=None):   
        self.fit(X,y)        
        return self.transform(X)
    

