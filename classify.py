
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# ---------------------------------------------------------------------------
class BinnedFitter(BaseEstimator,ClassifierMixin):
    
    def __init__(self,bins,ranges=None):
        self.bins = bins
        self.ranges = ranges
        
    # ---------------------------------------------------------------------------
    def fit(self, X, y, sample_weight=None):
        self.classes_, indices = np.unique(y,return_inverse=True)
        
        self.bin_edges_, denom = np.histogramdd(X,bins=self.bins,weights=sample_weight,range=self.ranges)
        self.weights_ = map(lambda ind: np.histogramdd(X[ind],bins=self.bins,weights=sample_weight,range=self.ranges)[1]/denom, indices)
        

    # ---------------------------------------------------------------------------
    def predict(X):
        return np.argmax(self.predict_proba(X),axis=1)
        
    # ---------------------------------------------------------------------------
    def predict_proba(X):
        
        # this is too convoluted
        indexes = map(lambda y: tuple(y), 
                      np.array(map(lambda x: np.digitize(X[:,x],self.bin_edges_[x][1:-1]), xrange(X.shape[1]))).transpose()
                  )
        
        return np.array( map(lambda ind: np.array(map(lambda w: w[ind], self.weights_ ) ), indexes ) )
                
    # ---------------------------------------------------------------------------
    def get_params(self, deep=True):
        return { "bins" : self.bins, "ranges" : self.ranges }
    
    # ---------------------------------------------------------------------------
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self
