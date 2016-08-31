
import numpy as np
import root_numpy as rnp
import ROOT as RT

from math import sqrt,ceil

# ---------------------------------------------------------------------------
class MyROOTCache(object):
    def __init__(self,tmp="tmp.root"):
        self.objs_ = []
        
        RT.tmp = RT.TFile.Open(tmp,"recreate")
                
    def __call__(self,name):
        obj = RT.gDirectory.Get(name)
        try:
            obj.SetDirectory(0)
            self.objs_.append(obj)
        except:
            pass
        return obj

RT.getObj = MyROOTCache()

# ---------------------------------------------------------------------------
from collections import namedtuple

ProcessRecord = namedtuple("ProcessRecord","file proc trees gtree")

# ---------------------------------------------------------------------------
def readIn(fname,process,treepfx,ncat):
    # read-in trees: selected events
    fin = RT.TFile.Open(fname)
    trees = map(lambda x: fin.Get(treepfx+"_SigmaMpTTag_%d" % x ), xrange(ncat))
    
    # add also not selected events and make tree with all events
    lst = RT.TList()
    map(lambda x: lst.Add(x), trees+[fin.Get(treepfx+"_NoTag_0")])
    RT.tmp.cd()
    gtree = RT.TTree.MergeTrees(lst)
    
    return ProcessRecord(fin,process,trees,gtree)

# ---------------------------------------------------------------------------
def getHists(record,expr,name,binning,weight):
    proc  = record.proc
    trees = record.trees
    gtree = record.gtree
    hists = []
    print
    print proc
    for it,tree in enumerate(trees):
        RT.tmp.cd()
        hname = "%s%s_%d" % (name,proc,it)
        dstr = "%s>>%s%s" % (expr,hname,binning)
        print(dstr)
        tree.Draw(dstr,weight,"goff")
        hists.append( RT.getObj(hname) )
        
    return hists

# ---------------------------------------------------------------------------
def getGenHist(record,expr,name,binning,weight):
    proc  = record.proc
    gtree = record.gtree

    hname = "%s%s_gen" % (name,proc)
    
    gtree.Draw("%s>>%s%s" % (expr,hname,binning),weight,"goff")
    return RT.getObj(hname)

# ---------------------------------------------------------------------------
def mergeHists(hists):
    lst = RT.TList()
    map(lambda x: lst.Add(x), hists)
    hsum = hists[0].Clone()
    hsum.SetDirectory(0)
    hsum.Reset()
    hsum.Merge(lst)
    return hsum

# ---------------------------------------------------------------------------
def drawHists(procHistos,ncols=None,htitle=None,nplots=None):
    canv = RT.TCanvas()
    
    if not nplots:
        nplots = procHistos.ncats
    if nplots < 0:
        nplots = len(procHistos.hists)
    if ncols == None:
        ncols = int(ceil(sqrt(nplots)))
    nrows = nplots / ncols
    if nplots % ncols != 0: nrows+=1
    canv.Divide(nrows,ncols)
    
    for it,hist in enumerate(procHistos.hists[:nplots]):
        canv.cd(it+1)
        if htitle:
            hist.SetTitle(htitle % it)
        hist.GetZaxis().SetRangeUser(0,1)
        hist.Draw("colz")
        RT.gPad.RedrawAxis()
    canv.Draw()
    RT.getObj.objs_.append(canv)
    return canv

# ---------------------------------------------------------------------------
class ProcessHistos(object):
    
    # ---------------------------------------------------------------------------
    def __init__(self,process,fname,name,expr,binning,weight,ncats,treepfx="genDiphotonDumper/trees/InsideAcceptance_125_13TeV",doOverall=False,genOnly=False):
        record = readIn(fname,process,treepfx,ncats)
        
        if not genOnly:
            self.hists = getHists(record,expr,name,binning,weight)
            if doOverall:
                self.hists.append(mergeHists(self.hists))
        self.genHist = getGenHist(record,expr,name,binning,weight)
        self.ncats = ncats
        self.iseff = False
        
        record.file.Close()
        
    # ---------------------------------------------------------------------------
    def computeEfficiencies(self):
        if self.iseff: return
        # self.effs = map(lambda x: x.Clone("%s_eff"%x.GetName()), self.hists)
        map(lambda x: x.Divide(self.genHist), self.hists)
        self.iseff = True
        
    # ---------------------------------------------------------------------------
    def predictDistrib(self,eff,normalize=False):
        
        if normalize:
            self.genHist.Scale(1./self.genHist.Integral())
            
        rho = self.genHist
        effs = eff.hists
        self.predict  = map(lambda x: rho.Clone("predict%s_%d" % (rho.GetName(),x)), xrange(len(effs)))
        map(lambda x: x[0].Multiply(x[1]), zip(self.predict,effs) )
        
        return self.predict

    # ---------------------------------------------------------------------------
    def predictYields(self,eff,normalize=False):
        return map(lambda x: x.Integral(), self.predictDistrib(eff,normalize))
        

# ---------------------------------------------------------------------------
DataRecord = namedtuple("DataRecord","meas errs corr")

FitRecord  = namedtuple("FitRecord","meas err effRatios bias")

# ---------------------------------------------------------------------------
class Fitter(object):

    # ---------------------------------------------------------------------------
    def __init__(self,data,effs,ref):
        
        self.data = data
        self.effs = effs
        self.ref  = ref
        
        if data.meas.size != data.errs.size:
            raise Exception("Invalid data: size of measurements and errors do not coincide %d %d" % ( data.meas.size, data.errs.size ))
            
        if data.corr and ( data.errs.size**2 != data.corr.size ):
            raise Exception("Invalid data: size of the correlation matrix does not match that of the errors vector %d %d" % ( data.errs.size, data.corr.size ))
            
        if not data.corr:
            print "Warning: no covariance matrix given for data. Will assume uncorrelated errors."

        if data.meas.size != effs.ncats:
            raise Exception("Number of measurements different from number of categories %d %d" % ( data.meas.size, effs.ncats ))

        if effs.ncats != ref.ncats:
            raise Exception("Number of categories in reference model does not match number of efficiency maps %d %d" % ( effs.ncats, ref.ncats ))
        
        self.ncats = ref.ncats
        
        self.refEffs = np.array(self.ref.predictYields(self.effs,True)[:self.ncats])
        
        self.refWeights = 1./self.data.errs**2
        # print(self.refWeights)
        
        self.refWsum = self.refWeights.sum()
        
    # ---------------------------------------------------------------------------
    def fitModel(self,model):
        
        return self.fitEfficiencies( np.array(model.predictYields(self.effs,True)[:self.ncats]) )
                
    # ---------------------------------------------------------------------------
    def fitEfficiencies(self,efficiencies,isratio=False):
        
        effRatios = efficiencies / self.refEffs if not isratio else efficiencies
        
        num = ( effRatios *  self.refWeights * self.data.meas ).sum()
        den = ( effRatios **2 *  self.refWeights ).sum()
        
        meas = num/ den
        err  = 1./ sqrt( den )
        
        bias = ( effRatios  *  self.refWeights ).sum() / self.refWsum
        
        return FitRecord(meas,err,effRatios,bias)
