#!/usr/bin/env python
''' Analysis script for standard plots
'''
#
# Standard imports and batch mode
#
import ROOT, os
ROOT.gROOT.SetBatch(True)
c1 = ROOT.TCanvas() # do this to avoid version conflict in png.h with keras import ...
c1.Draw()
c1.Print('delete.png')
import itertools
import copy
import array
import operator
from math                                import sqrt, cos, sin, pi, atan2, cosh, exp

# RootTools
from RootTools.core.standard             import *

# MTopCorrelations
from MLUnfolding.Tools.user                      import plot_directory
from MLUnfolding.Tools.cutInterpreter            import cutInterpreter

# Analysis
from Analysis.Tools.helpers              import deltaPhi, deltaR
from Analysis.Tools.puReweighting        import getReweightingFunction
from Analysis.Tools.WeightInfo           import WeightInfo


# import Analysis.Tools.syncer # Update Cern Web Directory
import numpy as np

################################################################################
# Arguments
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',       action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--plot_directory', action='store', default='MLUnfolding_v3')
argParser.add_argument('--selection',      action='store', default='noSelection')
argParser.add_argument('--era',            action='store', type=str, default="Run2018")
args = argParser.parse_args()

################################################################################
# Logger
import MLUnfolding.Tools.logger as logger
logger    = logger.get_logger(   args.logLevel, logFile = None)
import RootTools.core.logger as logger_rt
logger_rt = logger_rt.get_logger(args.logLevel, logFile = None)

################################################################################
# Define the MC samples
from MLUnfolding.samples.nanoTuples_RunII_nanoAOD import *

lumi_scale = 0
if args.era == "Run2016":
    mc = [Run2016.TTbar]
    lumi_scale = 36.3
elif args.era == "Run2017":
    mc = [Run2017.TTbar]
    lumi_scale = 41.5
elif args.era == "Run2018":
    mc = [Run2018.TTbar]
    lumi_scale = 59.7
elif args.era == "RunII":
    mc = [TTbar]
    lumi_scale = 138

#coment
################################################################################
# Modify plotdir

################################################################################
# Text on the plots
tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.04)
tex.SetTextAlign(11) # align right

################################################################################
# Functions needed specifically for this analysis routine

def drawObjects( plotData, lumi_scale ):
    lines = [
      (0.15, 0.95, 'CMS Preliminary' if plotData else 'CMS Simulation'),
      (0.45, 0.95, 'L=%3.1f fb{}^{-1} (13 TeV) '% ( lumi_scale ) ) if plotData else (0.45, 0.95, 'L=%3.1f fb{}^{-1} (13 TeV)' % lumi_scale)
    ]
    return [tex.DrawLatex(*l) for l in lines]

def drawPlots(plots):
    for log in [False, True]:
        plot_directory_ = os.path.join(plot_directory, 'analysisPlots', args.plot_directory, args.era, ("log" if log else "lin"), args.selection)
        for plot in plots:
            if not max(l.GetMaximum() for l in sum(plot.histos,[])): continue # Empty plot

            _drawObjects = []
            n_stacks=len(plot.histos)
            plotData=False
            if isinstance( plot, Plot):
                plotting.draw(plot,
                  plot_directory = plot_directory_,
                  ratio =  None,
                  logX = False, logY = log, sorting = True,
                  yRange = (0.03, "auto") if log else (0.001, "auto"),
                  scaling = {},
                  legend = ( (0.18,0.88-0.03*sum(map(len, plot.histos)),0.9,0.88), 2),
                  drawObjects = drawObjects( plotData , lumi_scale ) + _drawObjects,
                  copyIndexPHP = True, extensions = ["png", "pdf", "root"],
                )

################################################################################
# Define sequences
sequence       = []

def buildSubjets( event, sample ):
    sub1_rec = ROOT.TLorentzVector()
    sub2_rec = ROOT.TLorentzVector()
    sub3_rec = ROOT.TLorentzVector()
    tmp_rec  = ROOT.TLorentzVector()
    
    sub1_gen = ROOT.TLorentzVector()
    sub2_gen = ROOT.TLorentzVector()
    sub3_gen = ROOT.TLorentzVector()
    tmp_gen  = ROOT.TLorentzVector()
    
    sub1_corr= event.sub1_factor_cor * event.sub1_factor_jec * event.sub1_factor_jer
    sub2_corr= event.sub2_factor_cor * event.sub2_factor_jec * event.sub2_factor_jer
    sub3_corr= event.sub3_factor_cor * event.sub3_factor_jec * event.sub3_factor_jer

    sub1_rec.SetPxPyPzE(event.sub1_px_rec * sub1_corr, event.sub1_py_rec * sub1_corr, event.sub1_pz_rec * sub1_corr, event.sub1_E_rec * sub1_corr)
    sub2_rec.SetPxPyPzE(event.sub2_px_rec * sub2_corr, event.sub2_py_rec * sub2_corr, event.sub2_pz_rec * sub2_corr, event.sub2_E_rec * sub2_corr)
    sub3_rec.SetPxPyPzE(event.sub3_px_rec * sub3_corr, event.sub3_py_rec * sub3_corr, event.sub3_pz_rec * sub3_corr, event.sub3_E_rec * sub3_corr)    
    
    sub1_gen.SetPxPyPzE(event.sub1_px_gen, event.sub1_py_gen, event.sub1_pz_gen, event.sub1_E_gen)
    sub2_gen.SetPxPyPzE(event.sub2_px_gen, event.sub2_py_gen, event.sub2_pz_gen, event.sub2_E_gen)
    sub3_gen.SetPxPyPzE(event.sub3_px_gen, event.sub3_py_gen, event.sub3_pz_gen, event.sub3_E_gen)
    
    tmp_rec = sub1_rec + sub2_rec + sub3_rec
    tmp_gen = sub1_gen + sub2_gen + sub3_gen
    
    weight = event.gen_weight

    if event.passed_measurement_rec!=0 : 
        weight *= event.rec_weight

        
    # if event.passed_measurement_rec!=0 : event.Mrec= tmp_rec.M()
    # elif: event.Mrec= float("nan")
    
    # if event.passed_measurement_gen!=0 : event.Mgen= tmp_gen.M()
    # elif: event.Mgen= float("nan")
    event.weight=weight
    event.Mrec= tmp_rec.M()
    event.Mgen= tmp_gen.M()
    
    #Test Comment File Upload 2
    #event.sub1_pt_rec = sub1_rec.Pt()
sequence.append(buildSubjets)

################################################################################
# Read variables

read_variables = [
    "rec_weight/F", "gen_weight/F",
    "sub1_E_rec/F", "sub1_px_rec/F", "sub1_py_rec/F", "sub1_pz_rec/F",
    "sub2_E_rec/F", "sub2_px_rec/F", "sub2_py_rec/F", "sub2_pz_rec/F",
    "sub3_E_rec/F", "sub3_px_rec/F", "sub3_py_rec/F", "sub3_pz_rec/F",
    "sub1_E_gen/F", "sub1_px_gen/F", "sub1_py_gen/F", "sub1_pz_gen/F",
    "sub2_E_gen/F", "sub2_px_gen/F", "sub2_py_gen/F", "sub2_pz_gen/F",
    "sub3_E_gen/F", "sub3_px_gen/F", "sub3_py_gen/F", "sub3_pz_gen/F",
    "sub1_factor_cor/F", "sub1_factor_jec/F", "sub1_factor_jer/F",
    "sub2_factor_cor/F", "sub2_factor_jec/F", "sub2_factor_jer/F",
    "sub3_factor_cor/F", "sub3_factor_jec/F", "sub3_factor_jer/F",
    "passed_measurement_gen/F",
    "passed_measurement_rec/F"
]

################################################################################
# Set up plotting
weightnames = ['weight']
getters = map(operator.attrgetter, weightnames)
def weight_function( event, sample):
    # Calculate weight, this becomes: w = event.weightnames[0]*event.weightnames[1]*...
    w = reduce(operator.mul, [g(event) for g in getters], 1)
    return w


for sample in mc:
    sample.style = styles.fillStyle(sample.color)
    sample.weight = weight_function

stack = Stack(mc)

# Use some defaults
weight_ = lambda event, sample: event.weight
Plot.setDefaults(stack = stack, weight = staticmethod(weight_), selectionString = cutInterpreter.cutString(args.selection))
Plot2D.setDefaults(stack = stack, weight = staticmethod(weight_), selectionString = cutInterpreter.cutString(args.selection))

################################################################################
# Now define the plots

plots = []
plots2D = []

plots.append(Plot(
    name = "Mrec",
    texX = 'Invariant Jet-Mass Det-Lvl [GeV]', texY = 'Number of Events',
    attribute = lambda event, sample: event.Mrec,
    binning=[25, 0., 500.],
))

plots.append(Plot(
    name = "Mgen",
    texX = 'Invariant Jet-Mass Gen-Par-Lvl [GeV]', texY = 'Number of Events',
    attribute = lambda event, sample: event.Mgen,
    binning=[25, 0., 500.],
))

boundaries_gen  = list(range(0,501,25))
boundaries_rec = list(range(0,501,25))

binning_gen  = Binning.fromThresholds(boundaries_gen)
binning_rec = Binning.fromThresholds(boundaries_rec)


plots2D.append(Plot2D(
    name = "TransfereMatrix",
    texX = 'Mgen', texY = 'Mrec',
    attribute = (
        lambda event, sample: event.Mgen,
        lambda event, sample: event.Mrec,
    ),
    binning = [binning_gen, binning_rec],
))

plotting.fill(plots+plots2D, read_variables = read_variables, sequence = sequence)

drawPlots(plots)
drawPlots(plots2D)

# Also store plots in root file
logger.info( "Now write results in root files." )
plots_root = ["Mgen", "Mrec","TransfereMatrix"]
plot_dir = os.path.join(plot_directory, 'analysisPlots', args.plot_directory, args.era, "lin", args.selection)
if not os.path.exists(plot_dir):
    try:
        os.makedirs(plot_dir)
    except:
        print 'Could not create', plot_dir
outfilename = plot_dir+'/Results.root'
logger.info("Saving in %s"%outfilename)
outfile = ROOT.TFile(outfilename, 'recreate')
outfile.cd()
for plot in plots+plots2D:
    if plot.name in plots_root:
        for idx, histo_list in enumerate(plot.histos):
            for j, h in enumerate(histo_list):
                histname = h.GetName()
                if "Wjets" in histname: process = "Wjets"
                elif "SingleTop" in histname: process = "SingleTop"
                elif "TTbar_1665" in histname: process = "TTbar_1665"
                elif "TTbar_1695" in histname: process = "TTbar_1695"
                elif "TTbar_1715" in histname: process = "TTbar_1715"
                elif "TTbar_1735" in histname: process = "TTbar_1735"
                elif "TTbar_1755" in histname: process = "TTbar_1755"
                elif "TTbar_1785" in histname: process = "TTbar_1785"
                elif "TTbar" in histname: process = "TTbar"
                elif "data" in histname: process = "data"
                h.Write(plot.name+"__"+process)
outfile.Close()

logger.info( "Done with prefix %s and selectionString %s"%(args.selection, cutInterpreter.cutString(args.selection)) )
