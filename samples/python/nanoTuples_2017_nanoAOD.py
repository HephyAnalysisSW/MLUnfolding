import copy, os, sys
from RootTools.core.Sample import Sample
import ROOT

# Logging
import logging
logger = logging.getLogger(__name__)

from MLUnfolding.samples.color import color

# Data directory
try:
    directory_ = sys.modules['__main__'].directory_
except:
    import MLUnfolding.samples.sample_locations as locations
    directory_ = locations.mc_2017

logger.info("Loading MC samples from directory %s", directory_)

def make_dirs( dirs ):
    return [ os.path.join( directory_, dir_ ) for dir_ in dirs ]

dirs = {}

dirs['TTbar']               = ["TTbar"]
TTbar = Sample.fromDirectory(name="TTbar", treeName="AnalysisTree", isData=False, color=color.TTbar, texName="t#bar{t}", directory=make_dirs( dirs['TTbar']))

dirs['TTbar_1715']               = ["TTbar_1715"]
TTbar_1715 = Sample.fromDirectory(name="TTbar_1715", treeName="AnalysisTree", isData=False, color=color.TTbar, texName="t#bar{t} m_{t} = 171.5 GeV", directory=make_dirs( dirs['TTbar_1715']))

dirs['TTbar_1735']               = ["TTbar_1735"]
TTbar_1735 = Sample.fromDirectory(name="TTbar_1735", treeName="AnalysisTree", isData=False, color=color.TTbar, texName="t#bar{t} m_{t} = 173.5 GeV", directory=make_dirs( dirs['TTbar_1735']))

dirs['SingleTop']               = ["SingleTop"]
SingleTop = Sample.fromDirectory(name="SingleTop", treeName="AnalysisTree", isData=False, color=color.singleTop, texName="Single t", directory=make_dirs( dirs['SingleTop']))

dirs['Wjets']               = ["Wjets"]
Wjets = Sample.fromDirectory(name="Wjets", treeName="AnalysisTree", isData=False, color=color.WJets, texName="W+jets", directory=make_dirs( dirs['Wjets']))
