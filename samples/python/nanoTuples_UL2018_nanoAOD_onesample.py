import copy, os, sys
from RootTools.core.Sample import Sample
import ROOT

# Logging
import logging
logger = logging.getLogger(__name__)

from EEEC.samples.color import color

# Data directory
try:
    directory_ = sys.modules['__main__'].directory_
except:
    import EEEC.samples.UL_nanoAODv9_locations as locations
    directory_ = locations.mc_UL2018

logger.info("Loading MC samples from directory %s", directory_)

def make_dirs( dirs ):
    return [ os.path.join( directory_, dir_ ) for dir_ in dirs ]

dirs = {}

# sample_path = '/scratch-cbe/users/dennis.schwarz/MTopCorrelations/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8'
dirs['TTbar'] = ["TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8"]
TTbar = Sample.fromDirectory(name="TTbar", treeName="Events", isData=False, color=color.TTbar, texName="t#bar{t}", directory=make_dirs(dirs['TTbar']))