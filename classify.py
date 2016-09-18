
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
        
        masks = [ y == cl for cl in self.classes_ ]
        
        denom, self.bin_edges_  = np.histogramdd(X,weights=sample_weight.ravel(),bins=self.bins,range=self.ranges)
        self.weights_ = map(lambda mask: np.histogramdd(X[mask],bins=self.bins,weights=sample_weight[mask],range=self.ranges)[0]/denom, masks)
        
        replace = denom == 0.
        self.weights_[0][replace] = 1.
        for wei in self.weights_[1:]: wei[replace] = 0.
        self.dims_ = denom.shape
        self.weights_ = np.vstack( map(lambda w: w.ravel(), self.weights_) ).transpose()
        

    # ---------------------------------------------------------------------------
    def predict(self,X):
        return np.argmax(self.predict_proba(X),axis=1)
        
    # ---------------------------------------------------------------------------
    def predict_proba(self,X):
        
        # this is too convoluted
        indexes = np.array(map(lambda x: np.digitize(X[:,x],self.bin_edges_[x][1:-1]), xrange(X.shape[1]))).transpose()
        
        ret = np.apply_along_axis(lambda ind: self.weights_[np.ravel_multi_index(ind,self.dims_),:], 1, indexes)
        return ret
                
    # ---------------------------------------------------------------------------
    def get_params(self, deep=True):
        return { "bins" : self.bins, "ranges" : self.ranges }
    
    # ---------------------------------------------------------------------------
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self
