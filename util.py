import train as tn
from pprint import pprint
import importlib
import os

import numpy as np

from copy import copy

params = {}

# -------------------------------------------------------------------------------------
def defaultParameters(**kwargs):
    """
    This function is used to set up parameters such as the directory where the root 
    files are located, the branches which will be read from the root trees and so 
    forth.
    : params      
            dataDir : string - specifies the directory of the data. (default =   
                      root://t3dcachedb03.psi.ch//pnfs/psi.ch/cms/trivcat/store
                      /user/musella/mod_dep_005")
          dataFname : string - specifies the file name of the root file inside
                      the directory dataDir. (default = "output_InsideAcceptance
                      _125.root")
                pfx : string - specifies the prefix going from the root data file
                      name dataFname down to the branches of the different classes.
                      The classes are NoTag_0, SigmaMpTTag_0, SigmaMpTTag_1 and 
                      SigmaMpTTag_2. (default =  "genDiphotonDumper/trees
                      /InsideAcceptance_125_13TeV")
           inputDir : string - specifis the directory of the classifier in case
                      one is going to load a classifier. (default = ".")
          inputName : string - specifies the name of the classifier fitted.
                      (default = "effFitter")
             outDir : string - specifis the directory where the classifier is
                      stored. (default = ".")
            outName : string - specifies the name of the classifier when saved in
                      outDir. (default = "effFitter_out")
              ncats : int - specifies the number of categories. (default = 3)
        genBranches : list - specifies the branches of generated events.
                      (default = ["genPt","genRapidity",
                             "genJet2p5Pt0","genJet2p5Rapidity0",
                             "genJet2p5Pt1","genJet2p5Rapidity1",
                             "genJet2p5Pt2","genJet2p5Rapidity2",
                             "genJet2p5Pt3","genJet2p5Rapidity3",
                             "weight",
                             "genNjets2p5"
                         ])
       recoBranches : list - specifies the branches of reconstructed events.
                      (default = ['recoPt','recoRapidity',"recoNjets2p5"])
            rndseed : int =  9347865 - specifies the starting point of the random 
                      number seed. Needed for shuffling. 
          rndseed 2 : int = 2315645 - see rndseed for explanantion.
         split_frac : float - specifies the amount of data that will be used for
                      the training. (default = 0.75 i.e. 75% of the data is used
                      for training)
               load : boolean - specifies whether to load an already trained 
                      classifier (True) or generate a new one (False).
                      (default = True)
          forceMake : boolean - specifies whether to force to generate a new 
                      classifier instance.
              clean : 
        classifiers : string - specifies the type of machine learning technique
                      you want to apply, e.g. class for classification.
                      (default = [])
         defineBins : 
    : retruns 
                    :
    """
    global params
    
    params["dataDir"]="root://t3dcachedb03.psi.ch//pnfs/psi.ch/cms/trivcat/store/user/musella/mod_dep_005"
    params["dataFname"] = "output_InsideAcceptance_125.root"
    params["pfx"] = "genDiphotonDumper/trees/InsideAcceptance_125_13TeV"
    
    params["inputDir"] = "."
    params["inputName"] = "effFitter"
    params["outDir"] = "."
    params["outName"] = "effFitter_out"
    
    params["ncats"] = 3
    params["genBranches"] = ["genPt","genRapidity",
                             "genJet2p5Pt0","genJet2p5Rapidity0",
                             "genJet2p5Pt1","genJet2p5Rapidity1",
                             "genJet2p5Pt2","genJet2p5Rapidity2",
                             "genJet2p5Pt3","genJet2p5Rapidity3",
                             "weight",
                             "genNjets2p5"
                         ]
    params["recoBranches"] = ['recoPt','recoRapidity',"recoNjets2p5"]

    params["rndseed"] = 9347865
    params["rndseed2"] = 2315645
    params["split_frac"] = 0.75
    
    params["load"] = True
    params["forceMake"] = False
    
    params["clean"] = []
    
    params["classifiers"] = []

    params["defineBins"] = {}
    
    params.update(kwargs)
    

# -------------------------------------------------------------------------------------
def setParams(default=None,config_files=None):
    global params 
    if default:
        params = copy(default)
    
    if not config_files:
        config_files = os.environ.get('my_train_config',None)

    if config_files:
        for cfg in config_files.split(','):
            print("reading %s" % cfg)
            with open(cfg) as fin: 
                loadparams = json.loads(fin.read())
                params.update(loadparams)

# -------------------------------------------------------------------------------------
def loadOrMake():

    global params
    pprint(params)

    name = params["inputName"]
    load = params["load"]
    forceMake = params["forceMake"]
    
    make = False
    if load:
        onDisk = tn.IO.load(name, path=params["inputDir"], nodata=forceMake)
        pprint(onDisk)
        if not forceMake:
            pprint(onDisk.df.columns)
        pprint(onDisk.clfs)
        if onDisk.genBranches != params["genBranches"] or onDisk.recoBranches != params["recoBranches"]:
            make = True
        if onDisk.ncats != params["ncats"]:
            make = True
            load = False
    else:
        make = True

    if make or forceMake:
        if not load:
            made = tn.EfficiencyFitter(name)
        else:
            made = onDisk

        files = params.get("dataFiles",[])
        if len(files) == 0:
            files.append( (0,params["dataFname"]) )
        inputs = [ (os.path.join(params["dataDir"],ifil[1]),
                    ifil[0],params["pfx"] if len(ifil)<3 else ifil[2],True,
                    params.get("genpfx",None) if len(ifil)<4 else ifil[3] )
                   for ifil in files ]
        made.readData(params["ncats"],params["genBranches"],params["recoBranches"],
                      inputs)
        ## fileName = os.path.join(params["dataDir"],params["dataFname"])
        ## made.readData(params["ncats"],params["genBranches"],params["recoBranches"],
        ##               [(fileName,None,params["pfx"])])
        
        print('shuffling dataset')
        np.random.seed(params['rndseed'])
        made.df['random_index'] = np.random.permutation(range(made.df.index.size))
        made.df.sort_values(by='random_index',inplace=True)
        made.df.set_index('random_index',inplace=True)
        made.split_frac = params['split_frac']
        
        print('defining bins')
        if 'genRapidity' in made.df.columns and ( not 'absGenRapidity' in made.df.columns ):
            made.df['absGenRapidity'] = np.abs(made.df['genRapidity'])
        runDefineBins(made,params["defineBins"])
        
    else:
        made = onDisk
    
    made.cleanClfs(params["clean"])
    
    made.outdir = params["outDir"]
    made.name = params["outName"]
    return made

# -------------------------------------------------------------------------------------
def setupJoblib(ipp_profile='default'):
    from sklearn.externals.joblib import Parallel, parallel_backend, register_parallel_backend
    
    import ipyparallel as ipp
    from ipyparallel.joblib import IPythonParallelBackend
    global joblib_rc,joblib_view,joblib_be
    joblib_rc = ipp.Client(profile=ipp_profile)
    joblib_view = joblib_rc.load_balanced_view()
    joblib_be = IPythonParallelBackend(view=joblib_view)
    
    register_parallel_backend('ipyparallel',lambda : joblib_be,make_default=True)


# -------------------------------------------------------------------------------------
def runDefineBins(fitter,binsDef):
    for name,params in binsDef.iteritems(): fitter.defineBins(name,**params)


# -------------------------------------------------------------------------------------
def target_name(key):
    postFix = ""
    if key != 'class':
        postFix = 'Cat' if (not key in params or not params[key][1].get('factorized',False)) else 'Bin'
    return key+postFix 

# -------------------------------------------------------------------------------------
def runTraining(effFitter):
    global params
    to_train = filter(lambda x: x not in effFitter.clfs.keys(), params["classifiers"])

    print("We need to train the following classifiers %s" % " ".join(to_train) )
    for key in to_train:
        classifier,train_params = params[key]
        pack,cls = classifier.rsplit('.',1)
        classifier = getattr(importlib.import_module(pack),cls)
        print("Fitting %s" % key)
        print(classifier)
        print(train_params)
        if key == 'class':
            effFitter.fitClass(classifier=classifier,**train_params)
        else:
            effFitter.fitBins(key,classifier=classifier,**train_params)


# -------------------------------------------------------------------------------------
def runEvaluation(effFitter):
    clf_keys = filter(lambda x: x in effFitter.clfs.keys(),params["classifiers"])
     
    for key in clf_keys:
        target = target_name(key)
        catKey = '%s_prob_0' % (target)
        if not catKey in effFitter.df.columns:
            print('running prediction for %s' % key)
            effFitter.runPrediction(target,effFitter.clfs[key])

    pprint(effFitter.df.columns)

