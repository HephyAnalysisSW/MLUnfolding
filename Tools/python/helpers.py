#Standard imports
import  ROOT
from    math    import pi, sqrt, cos, sin, sinh, log, cosh, acos
from    array   import array
import  itertools

# Logging
import  logging
logger  = logging.getLogger(__name__)

#scripts
ROOT.gROOT.LoadMacro("$CMSSW_BASE/src/EEEC/Tools/scripts/tdrstyle.C")
ROOT.setTDRStyle()
mZ=91.1876

def add_histos( l ):
    res = l[0].Clone()
    for h in l[1:]: res.Add(h)
    return res

def map_level(f, item, level):
    if level == 0:
        return f(item)
    else:
        return [map_level(f, i, level - 1) for i in item]

def make_TH1D( h, ignore_binning = False):
    # remove infs from thresholds
    vals, thrs = h
    if ignore_binning:
        histo = ROOT.TH1D("h","h",len(vals),0,len(vals))
    else:
        histo = ROOT.TH1D("h","h",len(thrs)-1,array('d', thrs))
    for i_v, v in enumerate(vals):
        histo.SetBinContent(i_v+1, v)
    return histo

def make_TH3D(h):
    # remove infs from thresholds
    vals, ( thrs1, thrs2, thrs3) = h

    histo = ROOT.TH3D("h","h",len(thrs1)-1,array('d', thrs1), len(thrs2)-1,array('d', thrs2), len(thrs3)-1,array('d', thrs3))

    for i_bx in range(1, histo.GetNbinsX()+1):
        for i_by in range(1, histo.GetNbinsY()+1):
            for i_bz in range(1, histo.GetNbinsZ()+1):
    
                histo.SetBinContent(i_bx, i_by, i_bz, vals[i_bx-1][i_by-1][i_bz-1])
    return histo

def deltaPhi(phi1, phi2):
    dphi = phi2-phi1
    if  dphi > pi:
        dphi -= 2.0*pi
    if dphi <= -pi:
        dphi += 2.0*pi
    return abs(dphi)

def deltaR2(l1, l2):
    return deltaPhi(l1['phi'], l2['phi'])**2 + (l1['eta'] - l2['eta'])**2

def deltaR(l1, l2):
    return sqrt(deltaR2(l1,l2))
    
def deltaRGenparts(g1, g2):
    dphi = deltaPhi(g1.phi(), g2.phi())
    deta = g1.eta() - g2.eta()
    return sqrt( deta**2 + dphi**2 )

def bestDRMatchInCollection(l, coll, deltaR = 0.2, deltaRelPt = 0.5 ):
    lst = []
    for l2 in coll:
        dr2 = deltaR2(l, l2)
        if  ( dr2 < deltaR**2 ) and (abs( -1 + l2['pt']/l['pt']) < deltaRelPt or deltaRelPt < 0 ):
            lst.append((dr2, l2))
    lst.sort()
    if len(lst)>0:
        return lst[0][1]
    else:
        return None

def getFileList(dir, histname='histo', maxN=-1):
    import os
    filelist = os.listdir(os.path.expanduser(dir))
    filelist = [dir+'/'+f for f in filelist if histname in f and f.endswith(".root")]
    if maxN>=0:
        filelist = filelist[:maxN]
    return filelist

def getSortedZCandidates(leptons):
    inds = range(len(leptons))
    vecs = [ ROOT.TLorentzVector() for i in inds ]
    for i, v in enumerate(vecs):
        v.SetPtEtaPhiM(leptons[i]['pt'], leptons[i]['eta'], leptons[i]['phi'], 0.)
    dlMasses = [((vecs[comb[0]] + vecs[comb[1]]).M(), comb[0], comb[1])  for comb in itertools.combinations(inds, 2) if leptons[comb[0]]['pdgId']*leptons[comb[1]]['pdgId'] < 0 and abs(leptons[comb[0]]['pdgId']) == abs(leptons[comb[1]]['pdgId']) ]
    # sort the candidates, only keep the best ones
    dlMasses = sorted(dlMasses, key=lambda (m,i1,i2):abs(m-mZ))
    usedIndices = []
    bestCandidates = []
    for m in dlMasses:
        if m[1] not in usedIndices and m[2] not in usedIndices:
            usedIndices += m[1:3]
            bestCandidates.append(m)
    return bestCandidates

def getMinDLMass(leptons):
    inds = range(len(leptons))
    vecs = [ ROOT.TLorentzVector() for i in inds ]
    for i, v in enumerate(vecs):
        v.SetPtEtaPhiM(leptons[i]['pt'], leptons[i]['eta'], leptons[i]['phi'], 0.)
    dlMasses = [((vecs[comb[0]] + vecs[comb[1]]).M(), comb[0], comb[1])  for comb in itertools.combinations(inds, 2) ]
    return min(dlMasses)[0] if len(dlMasses)>0 else -1.

# Returns (closest mass, index1, index2)
def closestOSDLMassToMZ(leptons):
    inds = [i for i in range(len(leptons))]
    vecs = [ROOT.TLorentzVector() for i in range(len(leptons))]
    for i, v in enumerate(vecs):
        v.SetPtEtaPhiM(leptons[i]['pt'], leptons[i]['eta'], leptons[i]['phi'], 0.)
    dlMasses = [((vecs[comb[0]] + vecs[comb[1]]).M(), comb[0], comb[1])  for comb in itertools.combinations(inds, 2) if leptons[comb[0]]['pdgId']*leptons[comb[1]]['pdgId'] < 0 and abs(leptons[comb[0]]['pdgId']) == abs(leptons[comb[1]]['pdgId']) ]
    return min(dlMasses, key=lambda (m,i1,i2):abs(m-mZ)) if len(dlMasses)>0 else (float('nan'), -1, -1)

def m3( jets ):
    if not len(jets)>=3: return float('nan'), -1, -1, -1
    vecs = [(i, ROOT.TLorentzVector()) for i in range(len(jets))]
    for i, v in enumerate(vecs):
        v[1].SetPtEtaPhiM(jets[i]['pt'], jets[i]['eta'], jets[i]['phi'], 0.)
    maxSumPt = 0
    m3 = float('nan')
    i1, i2, i3 = -1, -1, -1
    for j3_comb in itertools.combinations(vecs, 3):
        vecSum = sum( [v[1] for v in j3_comb], ROOT.TLorentzVector())
        if vecSum.Pt()>maxSumPt:
            maxSumPt = vecSum.Pt()
            m3 = vecSum.M()
            i1, i2, i3 =  [v[0] for v in j3_comb]
    return m3, i1, i2, i3

def cosThetaStar( Z_mass, Z_pt, Z_eta, Z_phi, l_pt, l_eta, l_phi ):

    Z   = ROOT.TVector3()
    l   = ROOT.TVector3()
    Z.SetPtEtaPhi( Z_pt, Z_eta, Z_phi )
    l.SetPtEtaPhi( l_pt, l_eta, l_phi )

    # get cos(theta) and the lorentz factor, calculate cos(theta*)
    cosTheta = Z*l / (sqrt(Z*Z) * sqrt(l*l))
    if Z_mass == 0:
        return -1 # (-1+ct)/(1-ct)
    gamma   = sqrt( 1 + Z_pt**2/Z_mass**2 * cosh(Z_eta)**2 )
    beta    = sqrt( 1 - 1/gamma**2 )
    if 1 - beta*cosTheta !=0:
        return (-beta + cosTheta) / (1 - beta*cosTheta)
    else:
        return -1

def checkRootFile(f, checkForObjects=[]):
    rf = ROOT.TFile.Open(f)
    if not rf: return False
    try:
        good = (not rf.IsZombie()) and (not rf.TestBit(ROOT.TFile.kRecovered))
    except:
        if rf: rf.Close()
        return False
    for o in checkForObjects:
        if not rf.GetListOfKeys().Contains(o):
            print "[checkRootFile] Failed to find object %s in file %s"%(o, f)
            rf.Close()
            return False
#    print "Keys recoveredd %i zombie %i tb %i"%(rf.Recover(), rf.IsZombie(), rf.TestBit(ROOT.TFile.kRecovered))
    rf.Close()
    return good

def nonEmptyFile(f, treeName='Events'):
    rf = ROOT.TFile.Open(f)
    if not rf: return False
    tree = getattr(rf, treeName)
    nonEmpty = True if tree.GetEntries() else False
    rf.Close()
    return nonEmpty

def getObjFromFile(fname, hname):
    f = ROOT.TFile(fname)
    assert not f.IsZombie()
    f.cd()
    htmp = f.Get(hname)
    if not htmp:  return htmp
    ROOT.gDirectory.cd('PyROOT:/')
    res = htmp.Clone()
    f.Close()
    return res

def writeObjToFile(fname, obj, update=False):
    gDir = ROOT.gDirectory.GetName()
    if update:
        f = ROOT.TFile(fname, 'UPDATE')
    else:
        f = ROOT.TFile(fname, 'recreate')
    objw = obj.Clone()
    objw.Write()
    f.Close()
    ROOT.gDirectory.cd(gDir+':/')
    return

def getVarValue(c, var, n=-1):
    try:
        att = getattr(c, var)
    except AttributeError:
        return float('nan')
    if n>=0:
#    print "getVarValue %s %i"%(var,n)
        if n<att.__len__():
            return att[n]
        else:
            return float('nan')
    return att

def getEList(chain, cut, newname='eListTMP'):
    chain.Draw('>>eListTMP_t', cut)
    #elistTMP_t = ROOT.gROOT.Get('eListTMP_t')
    elistTMP_t = ROOT.gDirectory.Get('eListTMP_t')
    elistTMP = elistTMP_t.Clone(newname)
    del elistTMP_t
    return elistTMP

def getObjDict(c, prefix, variables, i):
    res={var: getVarValue(c, prefix+var, i) for var in variables}
    res['index']=i
    return res

def getCollection(c, prefix, variables, counter_variable):
    return [getObjDict(c, prefix+'_', variables, i) for i in range(int(getVarValue(c, counter_variable)))]

import re
def natural_sort(list, key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)

    lc = sorted(list, key=sort_key)
    return lc

def getTheta(lep1, lep2, H):
	beam = ROOT.TLorentzVector()
	tmp_lep1, tmp_lep2, tmp_H = ROOT.TLorentzVector(), ROOT.TLorentzVector(), ROOT.TLorentzVector()

	tmp_lep1.SetPtEtaPhiM(lep1.Pt(),lep1.Eta(),lep1.Phi(),lep1.M())
	tmp_lep2.SetPtEtaPhiM(lep2.Pt(),lep2.Eta(),lep2.Phi(),lep2.M())
	tmp_H.SetPtEtaPhiM(H.Pt(),H.Eta(),H.Phi(),H.M())

	if(lep1.Eta()<-10 or lep2.Eta()<-10 or tmp_H.Eta()<-10):
		return -100

	beam.SetPxPyPzE(0,0,6500,6500)

	V_mom, bVH = ROOT.TLorentzVector(), ROOT.TVector3()
	V_mom = tmp_lep1+tmp_lep2
	bVH = (tmp_lep1+tmp_lep2+tmp_H).BoostVector()

	V_mom.Boost(-bVH)

	Theta = float('nan')

	try:
#	Theta  = acos((V_mom.Vect().Unit()).Dot(beam.Vect().Unit()))
		Theta = (V_mom.Vect().Unit()).Angle(beam.Vect().Unit())
	except Exception:
		pass
		Theta = -100

	return Theta

def gettheta(lep1, lep2, H):
	tmp_lep1, tmp_lep2, tmp_H = ROOT.TLorentzVector(), ROOT.TLorentzVector(), ROOT.TLorentzVector()
	tmp_lep1.SetPtEtaPhiM(lep1.Pt(),lep1.Eta(),lep1.Phi(),lep1.M())
	tmp_lep2.SetPtEtaPhiM(lep2.Pt(),lep2.Eta(),lep2.Phi(),lep2.M())
	tmp_H.SetPtEtaPhiM(H.Pt(),H.Eta(),H.Phi(),H.M())
	if(lep1.Eta()<-10 or lep2.Eta()<-10 or tmp_H.Eta()<-10):
		return -100

	V_mom, bVH, bV = ROOT.TLorentzVector(), ROOT.TVector3(), ROOT.TVector3()

	bVH = (tmp_lep1 + tmp_lep2 + tmp_H).BoostVector()
	V_mom = (tmp_lep1 + tmp_lep2)

	V_mom.Boost(-bVH)
	tmp_lep1.Boost(-bVH)

	bV = V_mom.BoostVector()
	tmp_lep1.Boost(-bV)

	theta = float('nan')
	try:
		theta = (V_mom).Angle(tmp_lep1.Vect())
	except Exception:
		pass
		theta = -100
	return theta

def getphi(lep1, lep2, H):
	beam = ROOT.TLorentzVector()
	tmp_lep1, tmp_lep2, tmp_H = ROOT.TLorentzVector(), ROOT.TLorentzVector(), ROOT.TLorentzVector()
	tmp_lep1.SetPtEtaPhiM(lep1.Pt(),lep1.Eta(),lep1.Phi(),lep1.M())
	tmp_lep2.SetPtEtaPhiM(lep2.Pt(),lep2.Eta(),lep2.Phi(),lep2.M())
	tmp_H.SetPtEtaPhiM(H.Pt(),H.Eta(),H.Phi(),H.M())

	if(lep1.Eta()<-10 or lep2.Eta()<-10 or tmp_H.Eta()<-10):
		return -100

	beam.SetPxPyPzE(0,0,6500,6500)

	V_mom, bVH, n_scatter, n_decay = ROOT.TLorentzVector(), ROOT.TVector3(), ROOT.TVector3(), ROOT.TVector3()
	bVH = (tmp_lep1+tmp_lep2+tmp_H).BoostVector()
	V_mom = tmp_lep1+tmp_lep2

	tmp_lep1.Boost(-bVH)
	tmp_lep2.Boost(-bVH)
	V_mom.Boost(-bVH)

	n_scatter = ((beam.Vect().Unit()).Cross(V_mom.Vect())).Unit()
	n_decay   = (tmp_lep1.Vect().Cross(tmp_lep2.Vect())).Unit()

	sign_flip =  1 if ( ((n_scatter.Cross(n_decay))*(V_mom.Vect())) > 0 ) else -1

	try:
		phi = sign_flip*acos(n_scatter.Dot(n_decay))
	except Exception:
		pass
		phi = -100

	return phi

def cosThetaStarNew(l, Z):
    lepton = ROOT.TLorentzVector()
    lepton.SetPtEtaPhiM(l.Pt(), l.Eta(), l.Phi(), l.M())
    Zboson = ROOT.TLorentzVector()
    Zboson.SetPtEtaPhiM(Z.Pt(), Z.Eta(), Z.Phi(), Z.M())

    boostvector = Zboson.BoostVector()
    lepton_newSys = ROOT.TLorentzVector()
    lepton_newSys = lepton
    lepton_newSys.Boost(-boostvector)
    return lepton_newSys.CosTheta()

def cosThetaStarTop(l, W, t):
    # Angle between lepton and W, lepton is measured in W rest system, W is measured in top rest system
    lepton = ROOT.TLorentzVector()
    lepton.SetPtEtaPhiM(l.Pt(), l.Eta(), l.Phi(), l.M())
    Wboson = ROOT.TLorentzVector()
    Wboson.SetPtEtaPhiM(W.Pt(), W.Eta(), W.Phi(), W.M())
    top = ROOT.TLorentzVector()
    top.SetPtEtaPhiM(t.Pt(), t.Eta(), t.Phi(), t.M())

    boostvectorW = Wboson.BoostVector()
    boostvectortop = top.BoostVector()

    lepton_newSys = ROOT.TLorentzVector()
    lepton_newSys = lepton
    lepton_newSys.Boost(-boostvectorW)
    W_newSys = ROOT.TLorentzVector()
    W_newSys = Wboson
    W_newSys.Boost(-boostvectortop)
    costhetastar = cos( lepton_newSys.Angle(W_newSys.Vect()) )
    return costhetastar
