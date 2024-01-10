from RootTools.core.standard import *

import MLUnfolding.samples.nanoTuples_2016_nanoAOD as Run2016
import MLUnfolding.samples.nanoTuples_2017_nanoAOD as Run2017
import MLUnfolding.samples.nanoTuples_2018_nanoAOD as Run2018


TTbar = Sample.combine( "TTbar", [Run2016.TTbar, Run2017.TTbar, Run2018.TTbar],texName = "t#bar{t}")
TTbar_1715 = Sample.combine( "TTbar_1715", [Run2016.TTbar_1715, Run2017.TTbar_1715, Run2018.TTbar_1715],texName = "t#bar{t} m_{t} = 171.5 GeV")
TTbar_1735 = Sample.combine( "TTbar_1735", [Run2016.TTbar_1735, Run2017.TTbar_1735, Run2018.TTbar_1735],texName = "t#bar{t} m_{t} = 173.5 GeV")
SingleTop = Sample.combine( "SingleTop", [Run2016.SingleTop, Run2017.SingleTop, Run2018.SingleTop],texName = "Single t")
Wjets = Sample.combine( "Wjets", [Run2016.Wjets, Run2017.Wjets, Run2018.Wjets],texName = "W+jets")
