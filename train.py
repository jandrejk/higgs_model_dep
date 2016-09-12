import numpy as np
import root_numpy as rnp
import ROOT as RT

import root_pandas as rpd
import pandas as pd

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
def setclass_and_weight(x):
    cls,df = x
    cls -= 1
    
    if proc:
        df['proc'] = np.full(df.index.proc,cls,dtype=np.int8)
    df['class'] = np.full(df.index.size,cls,dtype=np.int8)

    df['absweight'] = np.abs(df['weight'])
    df.weight_column = df['absweight'] 

# ---------------------------------------------------------------------------
def set_proc(x):
    proc,df = x
    
    df['proc'] = np.full(df.index.size,proc,dtype=np.int8)
    

# ---------------------------------------------------------------------------
def readIn(fname,process,treepfx,ncatm,genBranches,recoBranches):
    
    trees = map(lambda x: treepfx+"SigmaMpTTag_%d" %x, xrange(ncat))
    gtree = treepfx+"_NoTag_0"
    
    dfs = [rpd.read_root(fname,gtree,columns=genBranches)]+map(lambda x: 
                                                               rpd.read_root(fname,x,columns=genBranches+recoBranches), trees )
    
    map(setclass_and_weight,enumerate(dfs))
    
    df = pd.concat(dfs)
    if process:
        set_proc((process,df))
    
    return df

# ---------------------------------------------------------------------------
def savePandas()
