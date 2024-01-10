#
eval `scram runtime -sh`
cd $CMSSW_BASE/src

# nanoAOD tools (for MET Significance, JEC/JER...)
git clone git@github.com:cms-nanoAOD/nanoAOD-tools.git PhysicsTools/NanoAODTools
cd $CMSSW_BASE/src

# RootTools (for plotting, sample handling, processing)
git clone git@github.com:HephyAnalysisSW/RootTools.git
cd $CMSSW_BASE/src

# Shared samples (miniAOD/nanoAOD)
git clone git@github.com:HephyAnalysisSW/Samples.git
cd $CMSSW_BASE/src

# Shared analysis tools and data
git clone git@github.com:HephyAnalysisSW/Analysis.git
cd $CMSSW_BASE/src

#compile
eval `scram runtime -sh`
cd $CMSSW_BASE/src && scram b -j 8
