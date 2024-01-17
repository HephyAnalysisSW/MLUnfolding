#!/usr/bin/env python

import os
import ROOT
from MLUnfolding.Tools.user                              import plot_directory
from MLUnfolding.Tools.helpers                           import getObjFromFile

ROOT.gROOT.SetBatch(ROOT.kTRUE)

plotdir = plot_directory+"/TestPlots/"
if not os.path.exists( plotdir ): os.makedirs( plotdir )

filename = "/groups/hephy/cms/simon.hablas/www/MLUnfolding/plots/analysisPlots/MLUnfolding_v2/Run2016/lin/noSelection/Results.root"
histname_rec = "Mrec__TTbar"
# histname_gen = "Mgen__TTbar"


histname_rec = getObjFromFile(filename, histname_rec)
# histname_gen = getObjFromFile(filename, histname_gen)

canvas = ROOT.TCanvas("canvas", "canvas", 600, 600)
histname_rec.SetFillColor(ROOT.kAzure+7)
histname_rec.GetXaxis().SetTitle("Jet mass")
histname_rec.GetYaxis().SetTitle("Events")


# histname_gen.SetFillColor(0)
# histname_gen.SetLineColor(ROOT.kRed)
# histname_gen.SetLineWidth(2)
# histname_gen.SetLineStyle(2)


histname_rec.Draw("HIST")
# histname_gen.Draw("HIST SAME")


canvas.SaveAs(plotdir+"MeinPlot.pdf")
