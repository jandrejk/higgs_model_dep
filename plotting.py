
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import target_name

# ---------------------------------------------------------------------------
def rat_plus_remind(num,den):
    """
    This function returns the value of the ratio num : den rounded up
    to the next integer.
    :params 
            num : int - numerator
            den : int - denominator
    :retruns
            ret : int - num/den rounded up
    """
    ret = num / den
    if num % den > 0: ret+=1
    return ret

# ---------------------------------------------------------------------------
def scatter_hist(df,columns,cmap=plt.cm.Blues,figsize=(14,8),colorbar=False,
                 log=False, **kwargs):
    """
    This function produces a set of scatter plots.
    : params 
             df : pandas dataframe - includes the tabulated properties
                  of the classifier
        columns : list, string - specifies which columns in the dataframe
                  should be plotted against each other.  
           cmap : 
        figsize : tuple, int - specifies the figure size. (default: 
                  figsize=(14,8))
       colorbar : boolean - specifies whether to display the colorbar
                  (True) or not (False). (default=False)
            log : boolean - specifies whether to display the scatter
                  plot in logarithmic scale (True) or not (False). 
                  (default=False)
    : retruns 
                :
    """
    
    #extract the number of columns and initialize a figure with
    #sqaure size ncols x ncols
    ncols = len(columns)
    fig, axarr = plt.subplots(ncols,ncols,figsize=figsize)
    
    
    for ix,xcol in enumerate(columns):        
        xargs = {}
        if type(xcol) == tuple: 
            xcol, xargs = xcol
        xlabel = "prob "+"cat "+xargs.get('xlabel',xcol.split("_")[-1])
        for iy,ycol in enumerate(columns):
            histargs = { "bins" : 20, "edgecolor" : 'black', "color" : "red" } #, "normed" : True, "log" : log }
            histargs.update(xargs)
            yargs = {}
            if type(ycol) == tuple: 
                ycol,yargs = ycol
            ylabel = "prob "+"cat "+yargs.get('ylabel',ycol.split("_")[-1])
            if iy == ix:
                axarr[ix,iy].hist(df[xcol],weights=df['weight'], **histargs)
            else:
                axarr[iy,ix].hexbin(x=df[xcol],y=df[ycol],C=df['weight'],cmap=cmap)
                if colorbar: plt.colorbar(ax=axarr[iy,ix])
                    
            if ix == 0:
                axarr[iy,ix].set_ylabel(ylabel)
            else:
                plt.setp(axarr[iy,ix].get_yticklabels())#, visible=False)                
            if iy == ncols-1:
                axarr[iy,ix].set_xlabel(xlabel)
            else:
                plt.setp(axarr[iy,ix].get_xticklabels(), visible=False)
                
    plt.show()
    
    
    
    figsize = map(lambda x : x/3, figsize)
    fig2, ax = plt.subplots(ncols/2,ncols/2, figsize=figsize)
    
    for i,col in enumerate(columns):        
        args = {}
        if type(col) == tuple: 
            col, args = col
        xlabel = "prob "+"cat "+args.get('xlabel',col.split("_")[-1])
        ylabel = "weighted count" 
        histarg = { "bins" : 20, "edgecolor" : 'black', "color" : "red" }
        histarg.update(args)
        
        #indices in the subplots
        k = i/2
        l = i%2
        
        ax[k,l].hist(df[col],weights=df['weight'], **histarg)
        ax[k,l].set_xlabel(xlabel)
        if l == 0:
            ax[k,l].set_ylabel(ylabel)      
        
    fig2.tight_layout()        
    plt.show()
    
# ---------------------------------------------------------------------------
def efficiency_map(x,y,z,cmap=plt.cm.viridis,layout=None,
                   xlabel=None,ylabel=None,**kwargs):
    """
    This function produces efficiency plots for all the categories.
    : params 
            x : numpy.ndarry - specifies the x bins of the of the
                2d histogram.
            y : numpy.ndarry - specifies the corresponding y bins of the of 
                the 2d histogram.
            z : numpy.ndarray - specifiec the efficiency/probability of each
                category to be ploted.
         cmap : colormap style
       layout : tuple - specifies the number of rows and columns of the plot
                (default: layout=None)
       xlabel : string - specifies the label of the x-axis (default: 
                xlabel=None)
       ylabel : string - specifies the label of the y-axis (default: 
                ylabel=None)
            
    """
    #initilize a figure. kwargs are e.g. figsize
    fig = plt.figure(**kwargs)
    #extract the number of categories
    nplots = z[0].size
    
    #This block of code is used in order to find out the number of rows
    #and columns in case no specific layout is given.
    #-----------------------------------------------
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
    #-----------------------------------------------
    
    for icat in xrange(1,nplots):
        #row and column index
        irow = (icat-1) / ncols
        icol = (icat-1) % ncols

        ## ax = axarr[irow,icol]
        ax = plt.subplot(nrows, ncols, icat)
        plt.hexbin(x=x,y=y,C=z[:,icat],cmap=cmap,vmin=0,vmax=1)
        
        #editing the plots when to show which label
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
    
    #1. - not_reco = efficiency
    plt.hexbin(x=x,y=y,C=1.-z[:,0],cmap=cmap,vmin=0,vmax=1)
    if (nplots % (ncols) == 1): plt.ylabel(ylabel)
    else: 
        plt.setp(ax.get_yticklabels(), visible=False)
    if xlabel: plt.xlabel(xlabel)
    plt.title("total efficiency")
    
    fig.subplots_adjust(top=0.9)
    cbar_ax = fig.add_axes([0.15, 0.97, 0.7, 0.02])
    plt.colorbar(cax=cbar_ax,orientation='horizontal')
    
    
# ---------------------------------------------------------------------------
def naive_closure(df,column,first=0,logy=False,title=None):
    """
    This function produces 1d histograms ensuring the in this case the
    BDT gives comparable results to simply counting the events.
    :params 
             df : pandas dataframe - includes the tabulated properties
                  of the classifier
         column : list, string - specifies which columns in the dataframe
                  should be plotted against each other.  
          first : int - specifies from which category on the histogram
                  should be produced. This can be used to omit the first
                  class of not reconstructed events by setting first=1
                  (default: first=0)
           logy : boolean - specifies whether the histogram should have a 
                  logarithmic y-axis (True) or not (False).
                  (default: logy=False)
          title : string - specifies the title of the histogram (default: 
                  title = None)
    
    """
    
    #naive_closure(df,column=key,logy=True,title='All')
    
    
    target = target_name(column)
    #extract the number of features that belong to column
    nstats = np.unique(df[target]).size
    print("There are " + str(nstats) + " features of type " + str(target))
    
    pred_cols = map(lambda x: ("%s_prob_%d" % (target, x)), range(nstats) ) 
    
    print(pred_cols)
    
    #the outcome of trueh is an int
    trueh = np.histogram(df[target],np.arange(-1.5,nstats-0.5))[0].ravel()
    #the predicted hist sums over all events weighted by their probability
    #hence the result is a float
    predh = np.array(df[pred_cols].sum(axis=0)).ravel()
    
    #This print is not really needed
    #print(trueh,predh)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ## true = ax.bar(np.arange(0,2*(nstats),2)[first:],trueh[first:],color='black')
    ## pred = ax.bar(np.arange(1,2*(nstats)+1,2)[first:],predh[first:],color='red')

    #list of class features, e.g. [0,1,2,3]
    xp = np.arange(nstats)[first:]
    
    #pred = ax.bar(xp-0.5,predh[first:],color='green',width=1.,alpha=0.5)
    """
    ? Did a change here: removed the shift by 0.5 in the x-axis
    """
    pred = ax.bar(xp,predh[first:],color='green',width=1.,alpha=0.5, edgecolor='black')
    
    true = ax.errorbar(xp,trueh[first:],ls='None',
                       xerr=np.ones_like(xp)*0.5,
                       yerr=np.sqrt(trueh[first:]),
                       ecolor='black')
    plt.xticks(xp,xp)
    plt.xlabel(column)
    
    plt.ylabel("No. of events")
    
    if title:
        plt.title(title)
    
    if logy:
        ax.set_yscale('log')
        
        
    ax.legend((true,pred),("true","predicted"),bbox_to_anchor=(1.45, 1.))    
    #plt.legend((true,pred),("true","predicted"),loc='best')
    
    plt.show()

    
# ---------------------------------------------------------------------------
def control_plots(key,fitter):
    """
    This function produces a series of plots. First it performs a box plot.
    Then it produces a scatter plot and at the end some histograms with
    different selection cuts.
    
    : params   
              key : string - specifies what type of feature should be 
                    extracted, e.g. key='class'
           fitter : train.EfficiencyFitter - trained classifier
    """
    #goes to util to call target_name. If it is key=class then target=class
    target = target_name(key)
    
    #extract the number of classes. fitter.clfs[key].classes_ yields [-1,0,1,2 and datatype]
    nclasses = len(fitter.clfs[key].classes_)
    
    #map creates new list by applying the inside function to xrange(nclasses)
    #creates a list of [class_prob_0,...,class_prob_3]
    columns = map(lambda x: "%s_prob_%d" % (target,x), xrange(nclasses) ) 
    columns = columns[:1]+columns[1:]
    
    #create pandas data frame
    df = fitter.df
    #if data set was splitten in train and test set then take only the test set.
    #note that the test set is indexed from 0 to first_train_evt
    if fitter.split_frac > 0:
        first_train_evt = int(round(df.index.size*(1.-fitter.split_frac)))
        df = df[:first_train_evt]
    
    #needed for box plot
    nrows = nclasses/3+1 #would not work in python 3
    ncols = 3  
    #perform the box plot
    df.boxplot(by=target,column=columns,figsize=(7*ncols,7*nrows),layout=(nrows,ncols))
       
        
    #perform the scatter plots
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
  