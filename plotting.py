
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
def rat_plus_remind(num,den):
    ret = num / den
    if num % den > 0: ret+=1
    return ret

# ---------------------------------------------------------------------------
def scatter_hist(df,columns,cmap=plt.cm.Blues,figsize=(14,8),colorbar=False,
                 **kwargs):
    
    ncols = len(columns)
    fig, axarr = plt.subplots(ncols,ncols,figsize=figsize)
    
    for ix,xcol in enumerate(columns):        
        xargs = {}
        if type(xcol) == tuple: 
            xcol, xargs = xcol
        xlabel = xargs.get('xlabel',xcol.split("_")[-1])
        for iy,ycol in enumerate(columns):
            histargs = { "bins" : 20, "normed" : True, "log" }
            histargs.update(xargs)
            yargs = {}
            if type(ycol) == tuple: 
                ycol,yargs = ycol
            ylabel = yargs.get('ylabel',ycol.split("_")[-1])
            if iy == ix:
                axarr[ix,iy].hist(df[xcol],**histargs)
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
                plt.setp(axarr[iy,ix].get_yticklabels(), visible=False)

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
    
    
