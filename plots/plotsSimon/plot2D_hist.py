#!/usr/bin/env python

import os
import ROOT
from ROOT import gStyle, TCanvas, TH1D, TLegend, kRed, kBlue
from MLUnfolding.Tools.user                              import plot_directory
from MLUnfolding.Tools.helpers                           import getObjFromFile

ROOT.gROOT.SetBatch(ROOT.kTRUE)

plotdir = plot_directory+"/TestPlots/"
if not os.path.exists( plotdir ): os.makedirs( plotdir )

# filename = "/groups/hephy/cms/simon.hablas/www/MLUnfolding/plots/analysisPlots/MLUnfolding_v2/Run2016/lin/noSelection/Results.root"
filename = "/groups/hephy/cms/simon.hablas/www/MLUnfolding/plots/analysisPlots/MLUnfolding_v3/Run2018/lin/rec_pass-gen_pass/Results.root"
histname_rec_name = "Mrec__TTbar"
histname_gen_name = "Mgen__TTbar"
histname_rec = "Mrec__TTbar"
histname_gen = "Mgen__TTbar"
histname_2d = "TransfereMatrix__TTbar"

histname_rec = getObjFromFile(filename, histname_rec)
histname_gen = getObjFromFile(filename, histname_gen)
histname_2d = getObjFromFile(filename, histname_2d)

canvas = ROOT.TCanvas("canvas", "canvas",800,800)
# histname_rec.SetFillColor(ROOT.kAzure+7)
# histname_rec.SetTitle("Invariant jet mass ;mass [GeV];Events")
# histname_rec.GetYaxis().SetTitleOffset(1.3)
# histname_rec.GetYaxis().SetRangeUser(1., 20000.);
# histname_rec.SetStats(0)

histname_2d

canvas.SetRightMargin(0.09)
canvas.SetLeftMargin(0.2)
canvas.SetBottomMargin(0.2)



histname_gen.SetFillColor(0)
histname_gen.SetLineColor(ROOT.kRed)
histname_gen.SetLineWidth(2)
histname_gen.SetLineStyle(2)


leg = TLegend(0.9,1,0.9,1)
leg.AddEntry(histname_rec,"Det-Lvl")
leg.AddEntry(histname_gen,"Part-Lvl")


# histname_rec.Draw("HIST")
# histname_gen.Draw("HIST SAME")
histname_2d.Draw("colz")
#leg.Draw()

canvas.SaveAs(plotdir+"Run2018_transfereMat.pdf")


