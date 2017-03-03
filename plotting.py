
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import target_name

# ---------------------------------------------------------------------------
def rat_plus_remind(num,den):
    ret = num / den
    if num % den > 0: ret+=1
    return ret

# ---------------------------------------------------------------------------
def scatter_hist(df,columns,cmap=plt.cm.Blues,figsize=(14,8),colorbar=False,
                 log=False,
                 **kwargs):
    
    ncols = len(columns)
    fig, axarr = plt.subplots(ncols,ncols,figsize=figsize)
    
    for ix,xcol in enumerate(columns):        
        xargs = {}
        if type(xcol) == tuple: 
            xcol, xargs = xcol
        xlabel = "cat "+xargs.get('xlabel',xcol.split("_")[-1])
        for iy,ycol in enumerate(columns):
            histargs = { "bins" : 20 } #, "normed" : True, "log" : log }
            histargs.update(xargs)
            yargs = {}
            if type(ycol) == tuple: 
                ycol,yargs = ycol
            ylabel = "cat "+yargs.get('ylabel',ycol.split("_")[-1])
            if iy == ix:
                axarr[ix,iy].hist(df[xcol],weights=df['weight'],**histargs)
            else:
                axarr[iy,ix].hexbin(x=df[xcol],y=df[ycol],C=df['weight'],cmap=cmap)
                if colorbar: plt.colorbar(ax=axarr[iy,ix])
            if ix == 0:
                axarr[iy,ix].set_ylabel(ylabel)
            else:
                plt.setp(axarr[iy,ix].get_yticklabels(), visible=False)                
            if iy == ncols-1:
                axarr[iy,ix].set_xlabel(xlabel)
            else:
                plt.setp(axarr[iy,ix].get_xticklabels(), visible=False)
                
    plt.show()

# ---------------------------------------------------------------------------
def efficiency_map(x,y,z,cmap=plt.cm.viridis,layout=None,
                   xlabel=None,ylabel=None,**kwargs):
    
    fig = plt.figure(**kwargs)
    nplots = z[0].size
    if not layout:
        for ncols in xrange(1,nplots):
            nrows = rat_plus_remind(nplots,ncols)
            if abs(nrows-ncols) <= 1: break
    else:
        ncols,nrows = layout
        if not nrows:
            nrows = rat_plus_remind(nplots,ncols)
        if not ncols:
            ncols = rat_plus_remind(nplots,nrows)
    ## layout=(nrows,ncols)
    
    for icat in xrange(1,nplots):
        irow = (icat-1) / ncols
        icol = (icat-1) % ncols
        ## ax = axarr[irow,icol]
        ax = plt.subplot(nrows, ncols, icat)
        plt.hexbin(x=x,y=y,C=z[:,icat],cmap=cmap,vmin=0,vmax=1)
        if icol == 0: 
            if ylabel: plt.ylabel(ylabel)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if ((irow+1)*ncols + icol >= nplots):
            if xlabel: plt.xlabel(xlabel)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
            
        plt.title("efficiency cat %d" % icat)
        ## if icol == ncols - 1: plt.colorbar()
        
    plt.subplot(nrows, ncols, nplots)
    plt.hexbin(x=x,y=y,C=1.-z[:,0],cmap=cmap,vmin=0,vmax=1)
    if (nplots % (nrows-1) == 1): 
        plt.ylabel(ylabel)
    else: 
        plt.setp(ax.get_yticklabels(), visible=False)
    if xlabel: plt.xlabel(xlabel)
    plt.title("total efficiency")
    
    fig.subplots_adjust(top=0.9)
    cbar_ax = fig.add_axes([0.15, 0.97, 0.7, 0.02])
    plt.colorbar(cax=cbar_ax,orientation='horizontal')
    
    
# ---------------------------------------------------------------------------
def naive_closure(df,column,first=0,logy=False,title=None):
    target = target_name(column)
    nstats = np.unique(df[target]).size
    print(target,nstats)
    
    pred_cols = map(lambda x: ("%s_prob_%d" % (target, x)), range(nstats) ) 
    
    trueh = np.histogram(df[target],np.arange(-1.5,nstats-0.5))[0].ravel()
    predh = np.array(df[pred_cols].sum(axis=0)).ravel()
    
    print(trueh,predh)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ## true = ax.bar(np.arange(0,2*(nstats),2)[first:],trueh[first:],color='black')
    ## pred = ax.bar(np.arange(1,2*(nstats)+1,2)[first:],predh[first:],color='red')

    xp = np.arange(nstats)[first:]
    pred = ax.bar(xp-0.5,predh[first:],color='green',width=1.,alpha=0.5)
    true = ax.errorbar(xp,trueh[first:],ls='None',
                       xerr=np.ones_like(xp)*0.5,
                       yerr=np.sqrt(trueh[first:]),
                       ecolor='black')
    plt.xticks(xp,xp)
    plt.xlabel(column)
    if title:
        plt.title(title)
    
    if logy:
        ax.set_yscale('log')
        
    plt.legend((true,pred),("true","predicted"),loc='best')
    
    plt.show()

# ---------------------------------------------------------------------------
def control_plots(key,fitter):
    target = target_name(key)
    
    nclasses = len(fitter.clfs[key].classes_)
    columns = map(lambda x: "%s_prob_%d" % (target,x), xrange(nclasses) )
    
    columns = columns[1:]+columns[:1]
    
    df = fitter.df
    if fitter.split_frac > 0:
        first_train_evt = int(round(df.index.size*(1.-fitter.split_frac)))
        df = df[:first_train_evt]
    
    nrows = nclasses/3+1
    ncols = 3
    df.boxplot(by=target,column=columns,figsize=(7*ncols,7*nrows),layout=(nrows,ncols))
     
    scatter_hist(df,columns,figsize=(28,28))
    
    
    naive_closure(df,key,logy=True,title='All')
    naive_closure(df,key,first=1,logy=False,title='All')
    
    naive_closure(df[df['genPt'] > 50.],key,first=1,logy=False,title='pT > 50')

    naive_closure(df[df['genPt'] < 50.],key,first=1,logy=False,title='pT < 50')

    naive_closure(df[df['absGenRapidity'] > 1.],key,title='|y| > 1.',
                  first=1,logy=False)
    naive_closure(df[df['absGenRapidity'] < 1.],key,
                  title='|y| < 1.',
                  first=1,logy=False)
    naive_closure(df[(df['absGenRapidity'] > 1.) & (df['genPt'] > 50.)],key,
                  title='|y| > 1. & pT > 50',
                  first=1,logy=False)
    naive_closure(df[(df['absGenRapidity'] > 1.) & (df['genPt'] < 50.)],key,
                  title='|y| > 1. & pT < 50',
                  first=1,logy=False)
    naive_closure(df[(df['absGenRapidity'] > 0.5) & (df['absGenRapidity'] < 1.) ],key,
                  title='0.5 < |y| < 1.',
                  first=1,logy=False)
    naive_closure(df[df['absGenRapidity'] < 0.5],key,
                  title='|y| < 0.5',
                  first=1,logy=False)
    naive_closure(df[(df['absGenRapidity'] > 0.25) & (df['absGenRapidity'] < .5) ],key,
                  title='0.25 < |y| < 0.5',
                  first=1,logy=False)
    naive_closure(df[df['absGenRapidity'] < 0.25],key,
                  title='|y| < 0.25',
                  first=1,logy=False)
