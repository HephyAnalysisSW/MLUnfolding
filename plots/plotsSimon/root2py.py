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
from MLUnfolding.Tools.user                      import processing_tmp_directory
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
argParser.add_argument('--plot_directory', action='store', default='MLUnfolding_tmp')
argParser.add_argument('--selection',      action='store', default='noSelection')
argParser.add_argument('--era',            action='store', type=str, default="RunII")
argParser.add_argument('--shuffle',  action='store', type=str, default="random") # random, false, <Pathfile>
argParser.add_argument('--split',    action='store', type=str, default="random") # random, false, <Pathfile>
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
    mc = [Run2018.TTbar_1735] #
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

h_pt_rec_noweight = {}
h_pt_rec = {}
h_pt_gen_noweight = {}
h_pt_gen = {}

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
    event.ptrec = tmp_rec.Pt()
    event.ptgen = tmp_gen.Pt()
    
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
print(cutInterpreter.cutString(args.selection))
tot_Events = 0
for sample in mc:
    r = sample.treeReader( variables = read_variables, selectionString = cutInterpreter.cutString(args.selection) )
    tot_Events+= r.nEvents
    
print("nEvents:")
print(tot_Events)
print("\n")
    
event_array = np.zeros((tot_Events,4))

print(np.shape(event_array))
array_Count = 0

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
        
        #SH: Write to event_array
        event_array[array_Count,3] = event.ptrec
        event_array[array_Count,2] = event.Mrec
        event_array[array_Count,1] = event.ptgen
        event_array[array_Count,0] = event.Mgen
        
        eventCount+=1
        array_Count+=1
        if eventCount >= count_threshold:
            count_threshold += 10000
            print "Processed", eventCount, "events"

        if event.passed_measurement_gen and event.passed_measurement_rec:
            h_pt_rec_noweight[sample.name].Fill(event.ptrec, 1.)
            h_pt_rec[sample.name].Fill(event.ptrec, event.rec_weight*event.gen_weight)
            h_pt_gen_noweight[sample.name].Fill(event.ptgen, 1.)
            h_pt_gen[sample.name].Fill(event.ptgen, event.gen_weight)


data_dir = os.path.join(processing_tmp_directory,"data", args.plot_directory, args.era,"TTbar_1735", args.selection)
meta_dir = os.path.join(processing_tmp_directory,"meta", args.plot_directory, args.era,"TTbar_1735", args.selection)
if not os.path.exists( data_dir ): os.makedirs( data_dir )
if not os.path.exists( meta_dir ): os.makedirs( meta_dir )
print("Now save file to " +data_dir)

with open(data_dir+"/ML_Data.npy", 'wb') as f:
    np.save(f, event_array)
    
#SH:
data_lenght = np.shape(event_array)[0]
data_n_cols = np.shape(event_array)[1]

print("Lenght of loaded array: "+str(data_lenght)+"\n")

#print("Pre shuffle:")
#print(event_array)  

#SH: Shuffle
shuffle_order = np.arange(data_lenght)
data_shuffled = np.zeros_like(event_array)

if args.shuffle == "random" : #Shuffle using random order
    print("Shuffling randomly ...")
    np.random.shuffle(shuffle_order) # https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html
    with open(meta_dir+"/shuffle.npy", 'wb') as fs: #SH: Save shuffle file
        np.save(fs, shuffle_order)
        
elif args.shuffle == "false" : #Do not Shuffle
    print("Do not shuffle")
    
else :                        #Shuffle using file
    print("Shuffle using file: \"" +args.shuffle+ "\"")
    try :
        with open(args.shuffle, "rb") as f:
            shuffle_order = np.load(f)
            f.close()
        assert (np.shape(shuffle_order)[0] == data_lenght), "Shuffle File Size Not Applicable! (" + str(np.shape(shuffle_order)[0]) + " vs " + str(data_lenght)+ ")"
         
    except FileNotFoundError :
        print("Shuffle file \""+ args.shuffle +"\" not found.")
        exit(1)

for i in range(data_lenght) :
    data_shuffled[i] = event_array[shuffle_order[i]]

del event_array, shuffle_order
#print("Post shuffle:")
#print(data_shuffled)

#SH: Splitting in 3 Parts (60,20,20% statistically, therefore array size is bigger )
train = np.zeros((int(data_lenght),data_n_cols), dtype=float) 
test = np.zeros((int(data_lenght/2),data_n_cols), dtype=float)
validate = np.zeros((int(data_lenght/2),data_n_cols), dtype=float)

#SH: Running Indices
train_lenght = 0
test_lenght = 0
validate_lenght = 0

if args.split == "random" : #Splitting Randomly
    print("Splitting randomly ...")
    random_test = np.random.randint(1,11, data_lenght)
    with open(meta_dir+"/split.npy", 'wb') as fs: #SH: Save shuffle file
        np.save(fs, random_test)
else :                        #Shuffle using file
    print("Splitting using file: \"" +args.split+ "\"...")
    try :
        with open(args.split, "rb") as f:
            random_test = np.load(f)
            f.close()
        assert (np.shape(random_test)[0] == data_lenght), "Splitting File Size Not Applicable! (" + str(np.shape(random_test)[0]) + " vs " + str(data_lenght)+ ")"
         
    except FileNotFoundError :
        print("Splitting file \""+ args.split +"\" not found.")
        exit(1)

for i in range(data_lenght) :
    rand_test = np.random.randint(1,11)
    if random_test[i] <= 6 :
        train[train_lenght] = data_shuffled[i]
        train_lenght += 1
    elif random_test[i] >= 9 :
        test[test_lenght] = data_shuffled[i]
        test_lenght += 1
    else :
        validate[validate_lenght] = data_shuffled[i]
        validate_lenght += 1
        
#Out-process data_shuffled       
with open(data_dir+"/ML_Data_shuffled.npy", 'wb') as f0:
    np.save(f0, data_shuffled) 
del data_shuffled

#Out-process train ing-data 
train = train[:train_lenght,:]    
print("Train Data:" + " (Lenght: "+str(np.shape(train)[0])+")")
#print(train)
with open(data_dir+"/ML_Data_train.npy", 'wb') as f0:
    np.save(f0, train) 
del train

#Out-process test ing-data 
test = test[:test_lenght,:]   
print("Test Data:" + " (Lenght: "+str(np.shape(test)[0])+")")
#print(test)
with open(data_dir+"/ML_Data_test.npy", 'wb') as f2:
    np.save(f2, test) 
del test

#Out-process validate ion-data 
validate = validate[:validate_lenght,:]   
print("Validation Data:" + " (Lenght: "+str(np.shape(validate)[0])+")")
#print(validate)
with open(data_dir+"/ML_Data_validate.npy", 'wb') as f3:
    np.save(f3, validate)
del validate

 

plot_dir = os.path.join(plot_directory, 'analysisPlots', args.plot_directory, args.era,"TTbar_1715", "lin", args.selection)
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