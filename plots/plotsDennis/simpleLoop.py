#!/usr/bin/env python
''' Analysis script with event loop
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

import Analysis.Tools.syncer
import numpy as np

################################################################################
# Arguments
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',       action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--plot_directory', action='store', default='MLUnfolding_v1')
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

################################################################################
# Define sequences
sequence       = []

def buildJets( event, sample ):

    # RECO
    sub1_rec = ROOT.TLorentzVector()
    sub1_rec.SetPxPyPzE(event.sub1_px_rec, event.sub1_py_rec, event.sub1_pz_rec, event.sub1_E_rec)
    sub1_rec = sub1_rec * event.sub1_factor_cor * event.sub1_factor_jec * event.sub1_factor_jer

    sub2_rec = ROOT.TLorentzVector()
    sub2_rec.SetPxPyPzE(event.sub2_px_rec, event.sub2_py_rec, event.sub2_pz_rec, event.sub2_E_rec)
    sub2_rec = sub2_rec * event.sub2_factor_cor * event.sub2_factor_jec * event.sub2_factor_jer

    sub3_rec = ROOT.TLorentzVector()
    sub3_rec.SetPxPyPzE(event.sub3_px_rec, event.sub3_py_rec, event.sub3_pz_rec, event.sub3_E_rec)
    sub3_rec = sub3_rec * event.sub3_factor_cor * event.sub3_factor_jec * event.sub3_factor_jer

    jet_rec = sub1_rec+sub2_rec+sub3_rec
    event.jet_rec_pt = jet_rec.Pt()

    # GEN
    sub1_gen = ROOT.TLorentzVector()
    sub1_gen.SetPxPyPzE(event.sub1_px_gen, event.sub1_py_gen, event.sub1_pz_gen, event.sub1_E_gen)
    sub2_gen = ROOT.TLorentzVector()
    sub2_gen.SetPxPyPzE(event.sub2_px_gen, event.sub2_py_gen, event.sub2_pz_gen, event.sub2_E_gen)
    sub3_gen = ROOT.TLorentzVector()
    sub3_gen.SetPxPyPzE(event.sub3_px_gen, event.sub3_py_gen, event.sub3_pz_gen, event.sub3_E_gen)

    jet_gen = sub1_gen+sub2_gen+sub3_gen
    event.jet_gen_pt = jet_gen.Pt()

sequence.append(buildJets)

################################################################################
# Read variables

read_variables = [
    "rec_weight/F", "gen_weight/F",

    "sub1_E_rec/F", "sub1_px_rec/F", "sub1_py_rec/F", "sub1_pz_rec/F",
    "sub1_factor_jer/F", "sub1_factor_jec/F", "sub1_factor_cor/F",

    "sub2_E_rec/F", "sub2_px_rec/F", "sub2_py_rec/F", "sub2_pz_rec/F",
    "sub2_factor_jer/F", "sub2_factor_jec/F", "sub2_factor_cor/F",

    "sub3_E_rec/F", "sub3_px_rec/F", "sub3_py_rec/F", "sub3_pz_rec/F",
    "sub3_factor_jer/F", "sub3_factor_jec/F", "sub3_factor_cor/F",


    "sub1_E_gen/F", "sub1_px_gen/F", "sub1_py_gen/F", "sub1_pz_gen/F",
    "sub2_E_gen/F", "sub2_px_gen/F", "sub2_py_gen/F", "sub2_pz_gen/F",
    "sub3_E_gen/F", "sub3_px_gen/F", "sub3_py_gen/F", "sub3_pz_gen/F",

    "passed_measurement_rec/F",
    "passed_measurement_gen/F",
]


################################################################################
# Event loop
h_pt_rec_noweight = {}
h_pt_rec = {}
h_pt_gen_noweight = {}
h_pt_gen = {}

for sample in mc:
    print "Running on", sample.name
    h_pt_rec_noweight[sample.name] = ROOT.TH1F("pt_rec_noweight__"+sample.name, "", 70, 300, 1000)
    h_pt_rec[sample.name]          = ROOT.TH1F("pt_rec__"+sample.name,          "", 70, 300, 1000)
    h_pt_gen_noweight[sample.name] = ROOT.TH1F("pt_gen_noweight__"+sample.name, "", 70, 300, 1000)
    h_pt_gen[sample.name]          = ROOT.TH1F("pt_gen__"+sample.name,          "", 70, 300, 1000)


    r = sample.treeReader( variables = read_variables, sequence = sequence, selectionString = cutInterpreter.cutString(args.selection) )
    r.start()
    eventCount = 0
    count_threshold = 10000
    while r.run():
        event = r.event
        eventCount+=1
        if eventCount >= count_threshold:
            count_threshold += 10000
            print "Processed", eventCount, "events"

        if event.passed_measurement_gen and event.passed_measurement_rec:
            h_pt_rec_noweight[sample.name].Fill(event.jet_rec_pt, 1.)
            h_pt_rec[sample.name].Fill(event.jet_rec_pt, event.rec_weight*event.gen_weight)
            h_pt_gen_noweight[sample.name].Fill(event.jet_gen_pt, 1.)
            h_pt_gen[sample.name].Fill(event.jet_gen_pt, event.gen_weight)
# Now store in root file
plot_dir = os.path.join(plot_directory, 'analysisPlots', args.plot_directory, args.era, "lin", args.selection)
if not os.path.exists( plot_dir ): os.makedirs( plot_dir )
outfilename = plot_dir+'/Results.root'
outfile = ROOT.TFile(outfilename, 'recreate')
outfile.cd()
for sample in mc:
    h_pt_rec_noweight[sample.name].Write()
    h_pt_rec[sample.name].Write()
    h_pt_gen_noweight[sample.name].Write()
    h_pt_gen[sample.name].Write()
outfile.cd()
