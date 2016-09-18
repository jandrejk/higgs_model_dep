import numpy as np
import root_numpy as rnp
import ROOT as RT

import root_pandas as rpd
import pandas as pd

import matplotlib.pyplot as plt
import os

import sklearn as skl
from sklearn import ensemble

from gzip import open as gopen
try:
   import cPickle as pickle
except:
   import pickle
# import pickle

import itertools

basedir=""

# ---------------------------------------------------------------------------
def setclass_and_weight(x):
    cls,df = x
    cls -= 1
    
    ## if proc:
    ##     df['proc'] = np.full(df.index.size,proc,dtype=np.int8)
    df['class'] = np.full(df.index.size,cls,dtype=np.int8)

    df['absweight'] = np.abs(df['weight'])
    df.weight_column = df['absweight'] 

# ---------------------------------------------------------------------------
def set_proc(x):
    proc,df = x
    
    df['proc'] = np.full(df.index.size,proc,dtype=np.int8)
    

# ---------------------------------------------------------------------------
def mk_grid_1d(x):
    nbins,xmin,xmax = x
    step = (xmax - xmin) / float(nbins)
    
    x0 = xmin + step*0.5
    return np.arange(x0,xmax,step)
    
# ---------------------------------------------------------------------------
def readRoot(fname,process,treepfx,ncat,genBranches,recoBranches):
    
    fname = os.path.join(basedir,fname)

    trees = map(lambda x: treepfx+"_SigmaMpTTag_%d" %x, xrange(ncat))
    gtree = treepfx+"_NoTag_0"
    
    dfs = [rpd.read_root(fname,gtree,columns=genBranches)]+map(lambda x: 
                                                               rpd.read_root(fname,x,columns=genBranches+recoBranches), trees )
    
    map(setclass_and_weight,enumerate(dfs))
    
    df = pd.concat(dfs)
    if process:
        set_proc((process,df))
    
    return df


# ---------------------------------------------------------------------------
class IO(object):

    # ---------------------------------------------------------------------------
    @staticmethod
    def reload(obj):
        new = type(obj)(obj.name)
        new.__dict__.update(obj.__dict__)
        return new

    # ---------------------------------------------------------------------------
    @staticmethod
    def saveData(obj):
        obj.df.to_root(os.path.join(obj.outdir,obj.name)+'.root',mode='w')
    
        
    # ---------------------------------------------------------------------------
    @staticmethod
    def loadData(path):
        return rpd.read_root(path)
        
    # ---------------------------------------------------------------------------
    @staticmethod
    def save(obj,nodata=False):
        fname = os.path.join(obj.outdir,obj.name)+'.pkl.gz'
        with gopen(fname,'w+') as fout:
            pickle.dump(obj,fout)
            fout.close()
        if not nodata:
            IO.saveData(obj)

    # ---------------------------------------------------------------------------
    @staticmethod
    def load(name,path='.',nodata=False):
        fname = os.path.join(path,name)+'.pkl.gz'
        with gopen(fname,'r') as fin:
            print("loading pickle %s" % fname)
            obj = pickle.load(fin)
            fin.close()
        
        if not nodata:
            dname = fname.replace('.pkl.gz','.root')
            print("loading data %s" % dname)
            obj.df = IO.loadData(dname)
        
        return IO.reload(obj)
    
    # ---------------------------------------------------------------------------
    @staticmethod
    def saveClf(obj,column='class'):
        with gopen(os.path.join(obj.outdir,obj.name)+'_'+column+'.pkl.gz','w+') as fout:
            pickle.dump(obj.clfs[column],fout)
            fout.close()

    # ---------------------------------------------------------------------------
    @staticmethod
    def loadClf(path,column):
        with gopen(path,'r') as fin:
            clf = pickle.load(fin)
            obj.clfs[column] = clf
            fout.close()
    
    
    
# ---------------------------------------------------------------------------
class EfficiencyFitter(object):
    
    # ---------------------------------------------------------------------------
    def __init__(self,name,outdir="."):
        
        self.name = name
        self.outdir = outdir

        self.df = None
        self.split_frac = 0.75
        self.best_params = {}

        self.recoBranches = []
        self.genBranches = []
        self.ncats = 0

        self.clfs = {}
        
    # ---------------------------------------------------------------------------
    def readData(self,ncats,genBranches,recoBranches,inputs):
        
        self.genBranches,self.recoBranches=genBranches,recoBranches
        self.ncats = ncats
        
        if not hasattr(self,"df"): 
            self.df = None
        if type(self.df) != type(None): 
            del self.df
            self.df = None
        map(lambda x: self.addData(*x), inputs)
        
        return self.df
        

    # ---------------------------------------------------------------------------
    def addData(self,fname,process,treepfx,merge=True):
        
        df = readRoot(fname,process,treepfx,self.ncats,self.genBranches,self.recoBranches) 
        
        if type(self.df) != type(None):
            self.df.append(df)
        else:
            self.df = df
            
        return self.df
        

    # ---------------------------------------------------------------------------
    def fitClass(self,**kwargs):

        if not 'absGenRapidity' in self.df.columns:
            self.df['absGenRapidity'] = np.abs(self.df['genRapidity'])
        
        Xbr = ['genPt','absGenRapidity']
        self.clfs['class'] = self.runFit(Xbr,'class','absweight',**kwargs)
        
        return self.clfs['class']

    # ---------------------------------------------------------------------------
    def effMap(self,column,grid):
        
        clf = self.clfs[column]
        inputs = clf.inputs
        
        if type(grid) == list:
            axes = map(mk_grid_1d,grid)
            grid = np.array(list(itertools.product(*axes)))
        
        # X = grid if conditional == None else np.hstack([grid,conditional])
        probs = clf.predict_proba(grid)
        return grid,probs
        
    # ---------------------------------------------------------------------------
    def runFit(self,Xbr,Ybr,wbr='absweight',
               cvoptimize=False,split=True,               
               classifier=ensemble.GradientBoostingClassifier,
               addprobs=True,addval=False,
               trainevts=-1,mask=None,
               **kwargs):
        
        if mask != None:
            self.split = None
            df = self.df[mask]
        else:
            df = self.df

        ## split_params = kwargs.get('split_params',self.split_params)
        if split:
            ### if not self.split:
            ###     self.split = skl.cross_validation.train_test_split(df,**split_params)
            ###     traindf,testdf = self.split
            split_frac = kwargs.get('split_frac',self.split_frac)
            first_train_evt = int(round(df.index.size*(1.-split_frac)))
            traindf = df[first_train_evt:]
        else:
            traindf = df
            
        X_train,y_train = traindf[Xbr][:trainevts].values,traindf[Ybr][:trainevts].values
        w_train = None if not wbr else traindf[wbr][:trainevts].values
        
        print "cvoptimize", cvoptimize
        if cvoptimize:
            cv_params_grid = kwargs.get('cv_params_grid')
            cv_nfolds      = kwargs.get('cv_nfolds')
            
            cvClf = skl.grid_search.RandomizedSearchCV(classifier(**kwargs),cv_params_grid,cv=cv_nfolds)
            
            cvClf.fit(X_train,y_train)
            self.best_params[Ybr] = cvClf.best_params_
            kwargs.update(cvClf.best_params_)
            
        clf = classifier(**kwargs)
        print(X_train.shape,X_train.size)
        print(y_train.shape,y_train.size)
        print(w_train.shape,w_train.size)
        
        clf.fit(X_train,y_train,sample_weight=w_train)
        clf.inputs = Xbr

        ### if mask != None:
        ###     self.split = None
        
        self.runPrediction(Ybr,clf,addprobs=addprobs,addval=addval)
            
        return clf
        

    # ---------------------------------------------------------------------------
    def defineBins(self,column,boundaries,overflow=True,underflow=False):
        
        binColumn,catColumn = self._binColName(column)
        
        colmin = self.df[column].min()
        colmax = self.df[column].max()
        
        if overflow:
            boundaries.append(colmax)
        if underflow:
            boundaries.insert(0,colmin)

        labels = xrange(len(boundaries)-1)
        
        if binColumn in self.df.columns: del self.df[binColumn]
        self.df[binColumn] = pd.cut(self.df[column],bins=boundaries,labels=labels)

        if catColumn in self.df.columns: del self.df[catColumn]
        self.df[catColumn] = self.df['class'] + float(self.ncats)*self.df[binColumn].astype(np.float)
        self.df[catColumn] = self.df[catColumn].fillna(-1)
        
        return binColumn,catColumn

    
    # ---------------------------------------------------------------------------
    def cleanClfs(self,keys):
        for key in filter(lambda x: x in self.clfs.keys(), keys):
            del self.clfs[key]
            for col in filter(lambda x: (key in x) and ("_prob_" in x), self.df.columns):
                del self.df[col]
    


    # ---------------------------------------------------------------------------
    def runPrediction(self,target,clf=None,addprobs=True,addval=False):
        
        if not clf:
            clf    = self.clfs[target]
        inputs = clf.inputs
        
        if addprobs:
            column_names = map(lambda x: "%s_prob_%d" % (target,x), xrange(len(clf.classes_)))
            column_probs = clf. predict_proba(self.df[inputs].values)
            
            for icol,name in enumerate(column_names):
                if name in self.df.columns: del self.df[name]
                self.df[name] = column_probs[:,icol]
                
        if addval:
            if "%s_predict" % target in df.columns: del df[df.columns]
            df["%s_predict" % target] = clf.predict(df[inputs].values)

        
    # ---------------------------------------------------------------------------
    def _binColName(self,column):
        return '%sBin' % column,'%sCat' % column
          
    # ---------------------------------------------------------------------------
    def fitBins(self,column,Xbr,factorized=False,includeClassProbs=True,boundaries=[],**kwargs):

        binColumn,catColumn = self._binColName(column)
        
        if not catColumn in self.df.columns:
            self.defineBins(column,boundaries)
        
        if factorized:
            clf = self.runFit(Xbr,binColumn,'absweight',mask=(self.df['class']>=0),**kwargs)
        else:
            if includeClassProbs:
                Xbr.extend( filter(lambda x: x.startswith("class_prob_"), self.df.columns  ) )
            
            clf = self.runFit(Xbr,catColumn,'absweight',**kwargs)
            
        self.clfs[column] = clf
        
        return binColumn,catColumn,clf
    
        
        
    # ---------------------------------------------------------------------------
    def __getstate__(self):
        skip = ["df","split"]
        return dict(filter(lambda x: not x[0] in skip, self.__dict__.items()))
        
    ###     
    ### # ---------------------------------------------------------------------------
    ### def save_data(self):
    ###     pass
    ### 
    ### # ---------------------------------------------------------------------------
    ### def load_data(self):
    ###     pass

    
