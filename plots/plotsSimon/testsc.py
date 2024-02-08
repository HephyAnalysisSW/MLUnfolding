#!/usr/bin/env python

import os
import ROOT
from MLUnfolding.Tools.user                              import plot_directory
from MLUnfolding.Tools.helpers                           import getObjFromFile

ROOT.gROOT.SetBatch(ROOT.kTRUE)

plotdir = plot_directory+"/TestPlots/"

print(plotdir)