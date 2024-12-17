#!/usr/bin/env python
''' Analysis script for standard plots
'''
#
# Standard imports and batch mode
#
import ROOT, os
from sys import getsizeof
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

# EEEC
from MLUnfolding.Tools.user                      import plot_directory
from MLUnfolding.Tools.user                      import processing_tmp_directory
from EEEC.Tools.cutInterpreter            import cutInterpreter
#from EEEC.Tools.objectSelection           import cbEleIdFlagGetter, vidNestedWPBitMapNamingList
#from EEEC.Tools.objectSelection           import lepString
#from EEEC.Tools.helpers                   import getCollection
from EEEC.Tools.energyCorrelators         import getTriplets_pp_TLorentz

# Analysis
from Analysis.Tools.helpers              import deltaPhi, deltaR
from Analysis.Tools.puProfileCache       import *
from Analysis.Tools.puReweighting        import getReweightingFunction
from Analysis.Tools.leptonJetArbitration     import cleanJetsAndLeptons

import Analysis.Tools.syncer
import numpy as np

################################################################################
# Arguments
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',       action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--plot_directory', action='store', default='EEEC_v3_with_mass')
argParser.add_argument('--selection',      action='store', default='nAK82p-AK8pt')
argParser.add_argument('--era',            action='store', type=str, default="UL2018")
argParser.add_argument('--shuffle',  action='store', type=str, default="random") # random, false, <Pathfile>
argParser.add_argument('--split',    action='store', type=str, default="random") # random, false, <Pathfile>
argParser.add_argument('--max_used_part',    action='store', type=int, default=50) # random, false, <Pathfile>
args = argParser.parse_args()

################################################################################
# Logger
import EEEC.Tools.logger as logger
import RootTools.core.logger as logger_rt
logger    = logger.get_logger(   args.logLevel, logFile = None)
logger_rt = logger_rt.get_logger(args.logLevel, logFile = None)

################################################################################
# Define the MC samples
# from EEEC.samples.nanoTuples_UL_RunII_nanoAOD import *
from EEEC.samples.nanoTuples_UL_RunII_nanoAOD_onesample import *

save_shuffled = False
save_unsplit_data = False

mc = [UL2018.TTbar,UL2018.TTbar_older,UL2018.TTbar_173p5,UL2018.TTbar_171p5 ]
# mc = [UL2018.TTbar_3]

lumi_scale = 60

pt_particle_cut = 5 #GeV

print(str(pt_particle_cut) + " Gev particle cut in use")

def get_atoptop_mass(event):
    # First find the two tops
    foundTop = False
    foundATop = False
    for i in range(event.nGenPart):
        if foundTop and foundATop:
            break
        if event.GenPart_pdgId[i] == 6:
            top = ROOT.TLorentzVector()
            top.SetPtEtaPhiM(event.GenPart_pt[i],event.GenPart_eta[i],event.GenPart_phi[i],event.GenPart_m[i])
            foundTop = True
        elif event.GenPart_pdgId[i] == -6:
            atop = ROOT.TLorentzVector()
            atop.SetPtEtaPhiM(event.GenPart_pt[i],event.GenPart_eta[i],event.GenPart_phi[i],event.GenPart_m[i])
            foundATop = True
    # Now search for leptons
    # if grandmother is the top, the atop is hadronice and vice versa
    return top.M()+atop.M()

def getZetas(scale, constituents, exponent=2, max_zeta=None, max_delta_zeta=None, delta_legs=None, shortest_side=None, log=False,part_max=50):
    # in pp, zeta = (Delta R)^2 and weight = (pT1*pT2*pT3 / pTjet^3)^exponent

    # transform coordinates to np.array
    constituents         = np.array( [[np.sqrt(c.Px()*c.Px()+c.Py()*c.Py()), c.Eta(), c.Phi(), c.M()]  for c in constituents])
    
    # keep track of indices
    pt   = 0
    eta  = 1
    phi  = 2
    mass = 3
    
    #Sort by pt
    constituents = constituents[constituents[:, pt].argsort()[::-1]] # Possbile place to print used actually used particles 
    constituents = constituents[0:part_max]
   
    
    # make triplet combinations
    #if log:
    #    # if we make a log plot, we cannot have dR=0, so we use the variant without repeting elements
    triplet_combinations = np.array(list(itertools.combinations( range(len(constituents)), 3))) # this gives all combinations WITHOUT repeating particles
    #else:
    #    triplet_combinations = np.array(list(itertools.combinations_with_replacement( range(len(constituents)), 3))) # this gives all combinations WITH repeating particles
    
    try:
        c = constituents[triplet_combinations]
    except IndexError:
        return np.empty((0,3)), np.empty((0))
    # first create an array of the dPhi since this has to be adjusted to by inside [-pi, pi]
    dPhiValues = np.zeros( ( len(c), 3), dtype='f' )
    dPhiValues[:,0] = c[:,0,phi] - c[:,1,phi]
    dPhiValues[:,1] = c[:,0,phi] - c[:,2,phi]
    dPhiValues[:,2] = c[:,1,phi] - c[:,2,phi]

    dPhiValues[dPhiValues  >  np.pi] += -2*np.pi
    dPhiValues[dPhiValues <= -np.pi] += 2*np.pi

    zeta_values = np.zeros( ( len(c), 3), dtype='f' )
    zeta_values[:,0] = (c[:,0,eta]-c[:,1,eta])*(c[:,0,eta]-c[:,1,eta]) + dPhiValues[:,0]*dPhiValues[:,0]
    zeta_values[:,1] = (c[:,0,eta]-c[:,2,eta])*(c[:,0,eta]-c[:,2,eta]) + dPhiValues[:,1]*dPhiValues[:,1]
    zeta_values[:,2] = (c[:,1,eta]-c[:,2,eta])*(c[:,1,eta]-c[:,2,eta]) + dPhiValues[:,2]*dPhiValues[:,2]

    zeta_values = np.sort( zeta_values, axis=1)

    if log:
        zeta_values = np.log10( np.sqrt(zeta_values) ) # this returns log(dR) instead of dR^2


    # Check if smallest dR is small enough
    mask = np.ones( len(zeta_values), dtype=bool)
    if max_zeta is not None:
        mask &= (zeta_values[:,0]<=max_zeta)

    # Check if the dRs form an equilateral triangle (for top)
    if max_delta_zeta is not None:
        mask &= (zeta_values[:,2]-zeta_values[:,0]<=max_delta_zeta)

    # Check if the dRs form a 2-point correlator (for W)
    if delta_legs is not None and shortest_side is not None:
        mask &= (~( (zeta_values[:,2]-zeta_values[:,1] > delta_legs) & (zeta_values[:,0] > shortest_side)))

    zeta_values = zeta_values[mask]
    c           = c[mask]
    del mask

    # pT weight
    weight = ( c[:,0,pt]*c[:,1,pt]*c[:,2,pt]) # / scale**3 )**exponent
    weight = weight / scale**3


    # The zeta triplet is sorted by size
    # zeta_values[0][0] is the shortest side of the first triplet
    # zeta_values[12][1] is the medium side of the 13th triplet
    # zeta_values[0][2] is the longest side of the first triplet

    # Also create a transformed version:
    # X = (zeta_medium+zeta_large)/2
    # Y = zeta_large-zeta_medium
    # Z = zeta_short
    # 1. Make an array with same dimensions and fill it with zeros
    # 2. Transform into new values
    transformed_values = np.full_like(zeta_values, 0)

    transformed_values[:,0] = pow( (np.sqrt(zeta_values[:,2]) + np.sqrt(zeta_values[:,1]))/2 , 2 )
    transformed_values[:,1] = pow(  np.sqrt(zeta_values[:,2]) - np.sqrt(zeta_values[:,1])    , 2 )
    transformed_values[:,2] = zeta_values[:,0]

    long = np.zeros( ( len(c), 1), dtype='f' )
    long[:,0] = zeta_values[:,2]

    medium = np.zeros( ( len(c), 1), dtype='f' )
    medium[:,0] = zeta_values[:,1]

    return zeta_values, transformed_values, long, medium, weight

################################################################################
# Functions needed specifically for this analysis routine
def getHadronicTop(event):
    # First find the two tops
    foundTop = False
    foundATop = False
    for i in range(event.nGenPart):
        if foundTop and foundATop:
            break
        if event.GenPart_pdgId[i] == 6:
            top = ROOT.TLorentzVector()
            top.SetPtEtaPhiM(event.GenPart_pt[i],event.GenPart_eta[i],event.GenPart_phi[i],event.GenPart_m[i])
            foundTop = True
        elif event.GenPart_pdgId[i] == -6:
            atop = ROOT.TLorentzVector()
            atop.SetPtEtaPhiM(event.GenPart_pt[i],event.GenPart_eta[i],event.GenPart_phi[i],event.GenPart_m[i])
            foundATop = True
    # Now search for leptons
    # if grandmother is the top, the atop is hadronice and vice versa
    for i in range(event.nGenPart):
        if abs(event.GenPart_pdgId[i]) in [11, 13, 15]:
            if event.GenPart_grmompdgId[i] == 6:
                return atop
            elif event.GenPart_grmompdgId[i] == -6:
                return top
    return None

def getClosestJetIdx(object, event, maxDR, genpf_switch):
    Njets = event.nGenJetAK8 if genpf_switch == "gen" else event.nPFJetAK8
    minDR = maxDR
    idx_match = None
    for i in range(Njets):
        jet = ROOT.TLorentzVector()
        if genpf_switch == "gen":
            jet.SetPtEtaPhiM(event.GenJetAK8_pt[i],event.GenJetAK8_eta[i],event.GenJetAK8_phi[i],event.GenJetAK8_mass[i])
        else:
            jet.SetPtEtaPhiM(event.PFJetAK8_pt[i],event.PFJetAK8_eta[i],event.PFJetAK8_phi[i],event.PFJetAK8_mass[i])
        if jet.DeltaR(object) < minDR:
            idx_match = i
            minDR = jet.DeltaR(object)
    return idx_match

################################################################################
# Define sequences
sequence       = []


def getConstituents( event, sample ):
    
    #print("m_T + m_aT =  " + str(get_atoptop_mass(event)))
    passSel = False
    genParts = []
    pfParts = []
    hadTop = getHadronicTop(event)
    scale_gen = None
    scale_rec = None
    if hadTop is not None:
        idx_genJet = getClosestJetIdx(hadTop, event, 0.8, "gen")
        if idx_genJet is not None:
            genJet = ROOT.TLorentzVector()
            genJet.SetPtEtaPhiM(event.GenJetAK8_pt[idx_genJet],event.GenJetAK8_eta[idx_genJet],event.GenJetAK8_phi[idx_genJet],event.GenJetAK8_mass[idx_genJet])
            if genJet.Pt() > 400:
                scale_gen = genJet.Pt()
                idx_pfJet = getClosestJetIdx(genJet, event, 0.8, "pf")
                if idx_pfJet is not None:
                    pfJet = ROOT.TLorentzVector()
                    pfJet.SetPtEtaPhiM(event.PFJetAK8_pt[idx_pfJet],event.PFJetAK8_eta[idx_pfJet],event.PFJetAK8_phi[idx_pfJet],event.PFJetAK8_mass[idx_pfJet])
                    if pfJet.Pt() > 400:
                        scale_pf = pfJet.Pt()
                        passSel = True
                        for iGen in range(event.nGenJetAK8_cons):
                            if event.GenJetAK8_cons_jetIndex[iGen] != idx_genJet:
                                continue
                            if abs(event.GenJetAK8_cons_pdgId[iGen]) not in [211, 13, 11, 1, 321, 2212, 3222, 3112, 3312, 3334]:
                                continue
                            genPart = ROOT.TLorentzVector()
                            genPart.SetPtEtaPhiM(event.GenJetAK8_cons_pt[iGen],event.GenJetAK8_cons_eta[iGen],event.GenJetAK8_cons_phi[iGen],event.GenJetAK8_cons_mass[iGen])
                            if genPart.Pt() < pt_particle_cut : #SH nur particles mit pt >= 5GeV verwenden
                                continue                            
                            genPartCharge = 1 if event.GenJetAK8_cons_pdgId[iGen]>0 else -1
                            genParts.append( (genPart, genPartCharge) )
                        for iRec in range(event.nPFJetAK8_cons):
                            if event.PFJetAK8_cons_jetIndex[iRec] != idx_pfJet:
                                continue
                            if abs(event.PFJetAK8_cons_pdgId[iRec]) not in [211, 13, 11, 1, 321, 2212, 3222, 3112, 3312, 3334]:
                                continue
                            pfPart = ROOT.TLorentzVector()
                            pfPart.SetPtEtaPhiM(event.PFJetAK8_cons_pt[iRec],event.PFJetAK8_cons_eta[iRec],event.PFJetAK8_cons_phi[iRec],event.PFJetAK8_cons_mass[iRec])
                            if pfPart.Pt() < pt_particle_cut :
                                continue    
                            pfPartCharge = 1 if event.PFJetAK8_cons_pdgId[iRec]>0 else -1
                            pfParts.append( (pfPart, pfPartCharge) )
    event.nGenParts = len(genParts)
    event.nPFParts = len(pfParts)
    maxDR_part = 0.05
    genMatches = {}
    alreadyMatched = []
    for i, (genPart, genCharge) in enumerate(genParts):
        matches = []
        for j, (pfPart, pfCharge) in enumerate(pfParts):
            if j in alreadyMatched:
                continue
            if genCharge == pfCharge and genPart.DeltaR(pfPart) < maxDR_part:
                matches.append(j)
        matchIDX = None
        if len(matches) == 0:
            matchIDX = None
        elif len(matches) == 1:
            matchIDX = matches[0]
        else:
            minPtDiff = 1000
            for idx in matches:
                PtDiff = abs(genPart.Pt()-pfParts[idx][0].Pt())
                if PtDiff < minPtDiff:
                    minPtDiff = PtDiff
                    gmatchIDX = idx
        genMatches[i] = matchIDX
        alreadyMatched.append(matchIDX)

    genParts_matched = []
    pfParts_matched = []
    for i, (genPart, genCharge) in enumerate(genParts):
        if genMatches[i] is not None:
            genParts_matched.append(genParts[i][0])
            pfParts_matched.append(pfParts[genMatches[i]][0])
    
    event.nGenAll = len(genParts) if len(genParts) > 0 else float('nan')
    event.nGenMatched = len(genParts_matched) if len(genParts) > 0 else float('nan')
    event.matchingEffi = float(len(genParts_matched))/float(len(genParts)) if len(genParts) > 0 else float('nan')
    event.passSel = passSel


    event.zeta_gen = np.zeros( ( len([]), 3), dtype='f' )
    event.weight_gen = np.zeros( ( len([]), 1), dtype='f' )
    event.zeta_rec = np.zeros( ( len([]), 3), dtype='f' )
    event.weight_rec = np.zeros( ( len([]), 1), dtype='f' )

    event.zeta_flag = False
    if len(genParts_matched) > 2:
        event.zeta_flag = True
        
        _, event.zeta_gen, _, _, event.weight_gen =  getZetas(scale_gen, genParts_matched, exponent=1, max_zeta=None, max_delta_zeta=None, delta_legs=None, shortest_side=None, log=False,part_max=args.max_used_part)
        _, event.zeta_rec, _, _, event.weight_rec =  getZetas(scale_pf,   pfParts_matched, exponent=1, max_zeta=None, max_delta_zeta=None, delta_legs=None, shortest_side=None, log=False,part_max=args.max_used_part)

        cut_min_shortside = 0.1
        cut_max_asymm     = (172.5 / scale_gen)**2

        filter_array = np.logical_and( event.zeta_gen[:,2] > cut_min_shortside,  event.zeta_gen[:,1] < cut_max_asymm)
        
        event.zeta_gen   = event.zeta_gen[filter_array]
        event.zeta_rec   = event.zeta_rec[filter_array]
        event.weight_gen = event.weight_gen[filter_array]
        event.weight_rec = event.weight_rec[filter_array]
        
        event.zeta_gen = event.zeta_gen[:,0]
        event.zeta_rec = event.zeta_rec[:,0]
        
        event.pt_rec = scale_pf
        event.pt_gen = scale_gen
        
        event.mt_gen = genJet.M()
        
sequence.append( getConstituents )

cut ="shortsidep1"

################################################################################
# Read variables

read_variables = [
    "nGenPart/I",
    "GenPart[pt/F,eta/F,phi/F,m/F,pdgId/I,mompdgId/I,grmompdgId/I]",
    "nGenJetAK8/I",
    "GenJetAK8[pt/F,eta/F,phi/F,mass/F]",
    "nGenJetAK8_cons/I",
    VectorTreeVariable.fromString( "GenJetAK8_cons[pt/F,eta/F,phi/F,mass/F,pdgId/I,jetIndex/I]", nMax=1000),
    "nPFJetAK8/I",
    "PFJetAK8[pt/F,eta/F,phi/F,mass/F]",
    "nPFJetAK8_cons/I",
    VectorTreeVariable.fromString( "PFJetAK8_cons[pt/F,eta/F,phi/F,mass/F,pdgId/I,jetIndex/I]", nMax=1000),
]

################################################################################
# Histograms
histograms = {
    "Weight_gen": ROOT.TH1F("Weight_gen", "Weight_gen", 100, 0, 0.04),
    "Weight_rec": ROOT.TH1F("Weight_rec", "Weight_rec", 100, 0, 0.04),
    "WeightZoom_gen": ROOT.TH1F("WeightZoom_gen", "WeightZoom_gen", 100, 0, 0.002),
    "WeightZoom_rec": ROOT.TH1F("WeightZoom_rec", "WeightZoom_rec", 100, 0, 0.002),
    "Weight_matrix": ROOT.TH2F("Weight_matrix", "Weight_matrix", 100, 0, 0.04, 100, 0, 0.04),
    "Zeta_gen": ROOT.TH1F("Zeta_gen", "Zeta_gen", 100, 0, 3.0),
    "Zeta_rec": ROOT.TH1F("Zeta_rec", "Zeta_rec", 100, 0, 3.0),
    "ZetaNoWeight_gen": ROOT.TH1F("ZetaNoWeight_gen", "ZetaNoWeight_gen", 100, 0, 3.0),
    "ZetaNoWeight_rec": ROOT.TH1F("ZetaNoWeight_rec", "ZetaNoWeight_rec", 100, 0, 3.0),
    "ZetaNoWeight_matrix": ROOT.TH2F("ZetaNoWeight_matrix", "ZetaNoWeight_matrix", 100, 0, 3.0, 100, 0, 3.0),
    "MatchingEfficiency": ROOT.TH1F("MatchingEfficiency", "MatchingEfficiency", 50, 0, 1.0),
}

#SH To Avoid Index Confusion
zeta_gen_index = 0
zeta_rec_index = 1
weight_gen_index = 2
weight_rec_index = 3
pt_gen_index = 4
pt_rec_index = 5
mass_gen_index= 6


outdir = "/groups/hephy/cms/simon.hablas/www/EEEC/results/"

print("Samples")
print(mc)

for sample in mc:
    print()
    print("Starting with sample "+sample.name)
    
    maximum_event_number = 100000000
    event_array = np.zeros((maximum_event_number,7))
    event_pt_bin = np.zeros(maximum_event_number)
    event_Count = 0

    r = sample.treeReader( variables = read_variables, sequence = sequence, selectionString = cutInterpreter.cutString(args.selection))
    r.start()
    array_Count = 0

    while r.run():
        if array_Count >= maximum_event_number :
            break
        event_Count += 1
        event = r.event
        if event.passSel and event.zeta_flag:
            for i in range(len(event.zeta_gen)):
                if array_Count >= maximum_event_number :
                    break
                event_array[array_Count,pt_rec_index] =      event.pt_rec
                event_array[array_Count,pt_gen_index] =      event.pt_gen
                event_array[array_Count,weight_rec_index] =  event.weight_rec[i]
                event_array[array_Count,zeta_rec_index] =    event.zeta_rec[i] * event.pt_rec**2 / 172.5**2
                event_array[array_Count,weight_gen_index] =  event.weight_gen[i]
                event_array[array_Count,zeta_gen_index] =    event.zeta_gen[i] * event.pt_gen**2 / 172.5**2
                event_array[array_Count,mass_gen_index] =    event.mt_gen
                array_Count+=1
    logger.info( "Done with sample "+sample.name+" and selectionString "+cutInterpreter.cutString(args.selection) )
    #For Loop End

    print(np.shape(event_array))

    event_array = event_array[0:array_Count]
    print(np.shape(event_array))
    data_dir = os.path.join(processing_tmp_directory,"data", args.plot_directory, args.era, args.selection,cut,str(args.max_used_part),"ptcut"+str(pt_particle_cut)+"GeV",sample.name)
    if not os.path.exists( data_dir ): os.makedirs( data_dir )
    print("Now save file to " +data_dir)

    #SH: Shuffle 
    np.random.shuffle(event_array)

    test_share = 0.0
    val_share = 0.4

    #SH :Split
    test     = [event_array[i] for i in range(int(event_array.shape[0] * test_share))]
    validate = [event_array[i] for i in range(int(event_array.shape[0] * test_share), int(len(event_array) * (test_share + val_share)))]
    train    = [event_array[i] for i in range(int(event_array.shape[0] * (test_share + val_share)), len(event_array))]

    #Out-process train ing-data 
    print("Train Data:" + " (Lenght: "+str(np.shape(train)[0])+")")
    #print(train)
    with open(data_dir+"/ML_Data_train.npy", 'wb') as f0:
        np.save(f0, train) 
    del train

    #Out-process test ing-data  
    print("Test Data:" + " (Lenght: "+str(np.shape(test)[0])+")")
    #print(test)
    with open(data_dir+"/ML_Data_test.npy", 'wb') as f2:
        np.save(f2, test) 
    del test

    #Out-process validation-data   
    print("Validation Data:" + " (Lenght: "+str(np.shape(validate)[0])+")")
    #print(validate)
    with open(data_dir+"/ML_Data_validate.npy", 'wb') as f3:
        np.save(f3, validate)
    del validate
    
    print("End Sample" + sample.name + " | Printed to: " + data_dir)
