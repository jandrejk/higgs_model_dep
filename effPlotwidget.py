from ipywidgets import interact, fixed, interactive, widgets, interact_manual
from IPython.display import display, clear_output, HTML



import train as tn
reload(tn)

import plotting
reload(plotting)

import util as ut

import matplotlib.pyplot as plt
from matplotlib import ticker

import numpy as np

from pprint import pprint

import itertools
import time

import os

from collections import OrderedDict

widgetparams={}

def defaultWidgets (**kwargs) :
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    proj_vars = ['absGenJet2p5Rapidity0','absGenJet2p5Rapidity1','absGenJet2p5Rapidity2',
    'absGenJet2p5Rapidity3','absGenRapidity','genJet2p5Pt0','genJet2p5Pt1','genJet2p5Pt2',
             'genJet2p5Pt3','genPt']
    proj_keys = ['|y| leading jet','|y| subleading jet','|y| 3rd leading jet','|y| 4th leading jet','|y| di-photon',
            'pt leading jet','pt subleading jet','pt 3rd leading jet','pt 4th leading jet','pt di-photon']

    proj_var_dict = OrderedDict(zip(proj_keys,proj_vars))
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    widgetparams['w_inputDir'] = widgets.Text(
        value='./classifiers',
        placeholder='directory to the classifiers',
        description='inputDir:',
        disabled=False,
        )

    widgetparams['w_dataDir'] = widgets.Text(
        value='./data',
        placeholder='directory to data files',
        description='dataDir:',
        disabled=False
        )
    
    #widgetparams['w_inputName'] = widgets.Dropdown(
    #    description='Classifier:',
    #    options=class_dict,
    #   )

    widgetparams['w_Load'] = widgets.Checkbox(
        value=False,
        description='Load classifier',
        disabled=False
        )

    widgetparams['w_varName_x'] = widgets.Dropdown(
        options=proj_var_dict,
        description='x-axis:'
        )

    widgetparams['w_varName_y'] = widgets.Dropdown(
        options=proj_var_dict,
        description='y-axis:',
        )

    widgetparams['w_mres_cat'] = widgets.ToggleButtons(
        options={'good':0, 'medium':1, 'bad':2},
        description='Di-photon mass resolution:',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Description',
        )

    widgetparams['w_noJets'] = widgets.SelectionSlider(
        options=OrderedDict(zip(['0','1','2','3','>3'],[0,1,2,3,4])),
        description='Number of jets:',
        )

    widgetparams['w_detector_eff'] = widgets.Checkbox(
        value=False,
        description='detector efficiency',
        disabled=False
        )
    
    widgetparams['w_prodProc'] = widgets.SelectMultiple(
    options={'ggF':0,'ttH':1,'VBF':2,'VH':3},
    #value=['ggF','ttH','VBF','VH'],
    description='production process',
    disabled=False
    )


def GetFitter ( dataDir, inputName, 
                inputDir,
                load,
               genBranch_params = None
              ) :
    
    #---------------------------------------------------------------------------- 
    if (load) :
        w_load_bar = widgets.IntProgress(
        min=0,
        max=3,
        description='Loading:',
        bar_style='success',
        orientation='horizontal' )   
        
        display(w_load_bar)
            
        ut.defaultParameters(dataDir=dataDir, inputName=inputName, inputDir=inputDir)
        w_load_bar.value += 1
        ut.setParams()
        w_load_bar.value += 1
        
        if (genBranch_params != None) :
            ut.params["genBranches"] = genBranch_params
        
        effFitter = ut.loadOrMake()
        w_load_bar.value += 1

        time.sleep(1)
        print('Done loading')
        #w_load_bar.close()

        return effFitter
    #----------------------------------------------------------------------------

    
def GetDictionary (inputDir) :
    #----------------------------------------------------------------------------
    classifier_names = []
    try :
        for names in os.listdir(inputDir) :
            if names.endswith(".pkl.gz") :
                classifier_names.append(names[:-7])
        if (classifier_names != []) :
            classifier_dictionary = dict(zip(classifier_names,classifier_names))
            print("classifiers found")
            return classifier_dictionary, inputDir;
        else : 
            print('No classifiers in directory '+inputDir)
            return 'Failure';
    except OSError :
        print("No such directory")
        return 'Failure';
    #----------------------------------------------------------------------------  

def GetBins (x_pt, y_pt, x_name, y_name) :
    #----------------------------------------------------------------------------   
    N_PtBins = 8
    N_RapidityBins = 8
    if (x_pt and y_pt) :
        return { x_name : dict(boundaries=np.linspace(0.,300.,N_PtBins)), 
                y_name : dict(boundaries=np.linspace(0.,300.,N_PtBins))
               }   
    else :
        if x_pt :
            return { x_name : dict(boundaries=np.linspace(0.,300.,N_PtBins)), 
                    y_name : dict(boundaries=np.linspace(0.,2.5,N_RapidityBins))
                   }
        if y_pt :
            return { x_name : dict(boundaries=np.linspace(0.,2.5,N_RapidityBins)), 
                    y_name : dict(boundaries=np.linspace(0.,300.,N_PtBins))
                   }
        return { x_name : dict(boundaries=np.linspace(0.,2.5,N_RapidityBins)), 
                y_name : dict(boundaries=np.linspace(0.,2.5,N_RapidityBins))
               }         
    #----------------------------------------------------------------------------   
    
    
def CheckforPt (name) :
    #----------------------------------------------------------------------------
    if 'Pt' in name :
        return True
    return False    
    #----------------------------------------------------------------------------

    
def DefineBinsforProjVar (effFitter, x_var,y_var) :
    #define bins
    defineBins = GetBins(x_pt=CheckforPt(x_var),y_pt=CheckforPt(y_var),x_name=x_var,y_name=y_var)
    ut.runDefineBins(effFitter,defineBins)
    return defineBins
    
    
def NjetsEffPlots (effFitter, x_var, y_var, prodProc=[], m_gamma_cat=0, Njets=0, effTag=False,
                  savepath=None) :
    #----------------------------------------------------------------------------  
    if (x_var == y_var) :
        print('Please choose 2 different variables for the projection plot')
    else :
        defineBins = DefineBinsforProjVar(effFitter, x_var,y_var)

        """
        df = effFitter.df
        first_train_evt = int(round(df.index.size*(1.-effFitter.split_frac)))
        #take the test sample 
        df_test = df[:first_train_evt]
        """
        df_initial = effFitter.df
        first_train_evt = int(round(df_initial.index.size*(1.-effFitter.split_frac)))
        #take the test sample 
        df_test_initial = df_initial[:first_train_evt]

        
        for proc in prodProc :
            print(proc)
            df = df_initial[df_initial['proc']==proc]
            df_test = df_test_initial[df_test_initial['proc']==proc]
        

            if effTag :
                NjetsCat = 0
            else :
                NjetsCat = 3*Njets +1 + m_gamma_cat

            i = NjetsCat

            column_proba_name = 'recoNjets2p5Cat_prob_'+str(i)





            gb_freq  = df.groupby([x_var+'Bin',y_var+'Bin']).apply(weight_freq,"recoNjets2p5Cat",(i-1),'weight')
            gb_proba = df_test.groupby([x_var+'Bin',y_var+'Bin']).apply(weighted_average, column_proba_name,'weight')


            title1 ='recoNjetsCat('+str(i-1)+') \n predicted by clf'
            title2 ='recoNjetsCat('+str(i-1)+') \n true from data'



            if (i==0) :
                #print('perform efficiency (1- reco-proba) plot')
                plot_imshow([gb_proba,gb_freq],binBoundaries=defineBins,x_lab=x_var,y_lab=y_var,titles=['predicted (clf) reco eff',
                                                                            'true data reco eff'],effTag=True,savepath=savepath)
            else :
                plot_imshow([gb_proba,gb_freq],binBoundaries=defineBins, x_lab=x_var,y_lab=y_var,titles=[title1,title2],
                            savepath=savepath)

            #----------------------------------------------------------------------------
    
def weighted_average(df_name, column_name, weight_name=None):
    """
        This function computes the weighted average of the quantity column_name
        stared in the pandas dataframe df_name. In case no weights are given
        or if they sum up to zero, the mean is returned instead.
        :params 
                df_name :
            column_name :
            weight_name :
        :retruns
                        :
        """
    #----------------------------------------------------------------------------
    d = df_name[column_name]
    w = df_name[weight_name]
    if (weight_name == None) :
        return float(d.mean())
    else :
        try:
            return (d * w).sum() / float(w.sum())
        except ZeroDivisionError:
            return float(d.mean())
    #----------------------------------------------------------------------------

def weight_freq (df_name, column_name, equal_to, weight_name) :
    #----------------------------------------------------------------------------

    df = df_name#[df_name[column_name]==equal_to]

    w_all = df[weight_name].sum()
    w_PartPhaseSpace = df[df[column_name] == equal_to][weight_name].sum()
    return w_PartPhaseSpace / w_all
    #----------------------------------------------------------------------------

    
            

def plot_imshow(groupby_objects, binBoundaries, 
                        titles=[],
                        x_lab = None,
                        y_lab = None,
                        cmap=plt.cm.Blues,
                        effTag=False, effPlot=False,
               savepath=None):
    #----------------------------------------------------------------------------
    if effTag :
        plot_imshow(groupby_objects, binBoundaries=binBoundaries, titles=['predicted (clf) reco eff', 'true (data) reco eff'],
                    x_lab = x_lab, y_lab = y_lab, cmap=plt.cm.Reds, effTag=False, effPlot=True, savepath=savepath)
    else :
        cm_list = []

        for k, r in enumerate(groupby_objects) :
            x,y = r.index.levels
            cm = r.values.reshape(len(x),len(y))
            if effPlot :
                cm = 1. - cm

            #check this
            cm_list.append(cm.T)    


        fig, axarr = plt.subplots(1,2,figsize=(12,12))


        minimum = np.amin(cm_list)
        maximum = np.amax(cm_list)

        for l, cm in enumerate(cm_list) :

            plt.subplot('22'+str(l+1))


            plt.imshow(cm_list[l], interpolation='nearest', cmap=cmap,origin='lower',
                       vmin=minimum,vmax=maximum)

            plt.title(titles[l])

            xtick_marks = np.arange(len(x)+1)-0.5
            xtick_names = binBoundaries[x_lab]['boundaries']
            ytick_marks = np.arange(len(y))-0.5
            ytick_names = binBoundaries[y_lab]['boundaries']

            xtick_names = np.round(xtick_names,2)
            ytick_names = np.round(ytick_names,2)
            plt.xticks(xtick_marks,xtick_names,rotation='90')

            plt.xlabel(x_lab)
            if l == 0 :
                plt.ylabel(y_lab)
                plt.yticks(ytick_marks, ytick_names)
            else :
                plt.yticks(visible=False)

                    #plt.tight_layout()

                # text
            thresh = (maximum + minimum) / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, '%.2f' % cm[i, j], horizontalalignment="center",
                                 color="white" if cm[i, j] > thresh else "black")
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.95, wspace=0.1, hspace=None)
        fig.subplots_adjust(top=0.85)   
        cbar_ax = fig.add_axes([0.1, 0.98, 0.8, 0.02])
        plt.colorbar(cax=cbar_ax,orientation='horizontal')#,norm=plt.colors.Normalize(vmin=min_col,vmax=max_col))
        
        #plt.tight_layout()
        
        """
        if (savepath == None) :
            print('no savepath')
            plt.show()
        else :
            print(savepath)
            plt.savefig(savepath,bbox_inches='tight')
        """
        
        if l == 0 :
            plotRelDiff_imshow(groupby_objects=groupby_objects,binBoundaries=binBoundaries,
                                       x_lab=x_lab,y_lab=y_lab,cmap=plt.cm.Oranges,effPlot=True,
                              savepath=savepath)
        else :
            plotRelDiff_imshow(groupby_objects=groupby_objects,binBoundaries=binBoundaries,
                                       x_lab=x_lab,y_lab=y_lab,cmap=plt.cm.Oranges,
                              savepath=savepath)
            
    #----------------------------------------------------------------------------


def plotRelDiff_imshow(groupby_objects,binBoundaries,
                       true_err=[],
                       title = '',
                        x_lab = None,
                        y_lab = None,
                          cmap=plt.cm.Blues,
                      effPlot=False,
                      savepath=None) :
    #----------------------------------------------------------------------------   
        
    fig = plt.figure(figsize=(7,7))

    r_pred, r_true = groupby_objects

    x_pred,y_pred = r_pred.index.levels
    x_true,y_true = r_true.index.levels

    cm_pred = r_pred.values.reshape(len(x_pred),len(y_pred))
    cm_true = r_true.values.reshape(len(x_true),len(y_true))

    
    if (len(true_err) != 0) :
        x_true_err, y_true_err = true_err.index.levels
        cm_err = true_err.values.reshape(len(x_true_err),len(y_true_err))
        cm_err = cm_err.T
    else :
        cm_err = (cm_true-cm_true).T
    

    if effPlot :
        cm_true = 1. - cm_true
        cm_pred = 1. - cm_pred

    total_sum_true_permille = np.sum(np.sum(cm_true))/1000.
    #print(total_sum_true_permille)

    cm = np.divide((cm_pred-cm_true),cm_true, out=np.zeros_like(cm_true-cm_pred), 
                   where=(cm_true>total_sum_true_permille) ) 

    cm = cm.T
    
    cm_abs_diff = abs(cm_pred-cm_true).T
    
    #plt.plot(np.arange(len(cm_true.T.ravel())),cm_true.T.ravel(),'bo')
    #plt.errorbar(np.arange(len(cm_true.T.ravel())),cm_true.T.ravel(), yerr=cm_err.ravel())
    #plt.yscale('log')
    #plt.ylim(0,1)
    #plt.show()
    
    
    
    
    
    
    
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap,origin='lower')
    plt.title(title+r'$\frac{\mathrm{pred} - \mathrm{true}}{\mathrm{true}}$'+ '\n'+str(cm.sum()))



    xtick_marks = np.arange(len(x_pred)+1)-0.5
    xtick_names = binBoundaries[x_lab]['boundaries']
    ytick_marks = np.arange(len(y_pred))-0.5
    ytick_names = binBoundaries[y_lab]['boundaries']

    xtick_names = np.round(xtick_names,2)
    ytick_names = np.round(ytick_names,2)
    plt.xticks(xtick_marks,xtick_names,rotation='90')

    plt.xticks(xtick_marks, xtick_names)
    plt.yticks(ytick_marks, ytick_names)


    thresh = (cm.max()+cm.min()) / 2.
    
    #print(cm_abs_diff)
    
    #print(cm_true.T)
    
    #print(cm_err)
    
    
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        
        """
        if (cm_err[i,j] == 'NoEv') :
            print(cm_err[i,j])
            plt.text(j, i, cm_err[i, j], horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        if (abs(cm[i,j]) > 0.2 ) :
            plt.text(j, i, np.round(cm[i, j],2), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        """
        if (cm_abs_diff[i,j] > 3.*cm_err[i,j] ) :
            plt.text(j, i, np.round(cm[i, j],2), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.tight_layout()

    fig.subplots_adjust(top=0.78)   
    cbar_ax = fig.add_axes([0.15, 0.97, 0.7, 0.02])
    cb = plt.colorbar(cax=cbar_ax,orientation='horizontal')
    
    
    if (savepath == None) :
        plt.show()
    else :
        print(savepath+'_relDif')
        plt.savefig(savepath+'_relDif')
    
    #----------------------------------------------------------------------------   

    

    
    