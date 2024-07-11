#!/usr/bin/env python

import os
import ROOT
from ROOT import gStyle, TCanvas, TH1D, TLegend, kRed, kBlue
from MLUnfolding.Tools.user                              import plot_directory
from MLUnfolding.Tools.helpers                           import getObjFromFile

ROOT.gROOT.SetBatch(ROOT.kTRUE)

plotdir = plot_directory+"/TestPlots/"
if not os.path.exists( plotdir ): os.makedirs( plotdir )

filedir = "/groups/hephy/cms/simon.hablas/www/MLUnfolding/plots/analysisPlots/MLUnfolding_v3/Run2018/lin/"
filedir = "/groups/hephy/cms/simon.hablas/www/MLUnfolding/plots/analysisPlots/MLUnfolding_tmp/RunII/lin/"
file = "Results.root"
filedirs = []
filedirs.append("rec_pass-gen_pass")
#filedirs.append("gen_pass")
#filedirs.append("rec_pass")
#filedirs.append("noSelection")
histname_rec_name = "pt_rec__TTbar"
histname_gen_name = "pt_gen__TTbar"
#histname_rec_name = "Mrec__TTbar"
#histname_gen_name = "Mgen__TTbar"

canvas = ROOT.TCanvas("canvas", "Title",800,600)

canvas.SetRightMargin(0.09)
canvas.SetLeftMargin(0.2)
canvas.SetBottomMargin(0.2)
canvas.SetTopMargin(0.1)

for subdir in filedirs:
    filename = filedir + subdir +"/"+ file

    histname_rec = histname_rec_name
    histname_gen = histname_gen_name

    histname_rec = getObjFromFile(filename, histname_rec)
    histname_gen = getObjFromFile(filename, histname_gen)    

    histname_rec.SetFillColor(ROOT.kAzure+7)
    histname_rec.GetYaxis().SetTitleOffset(1.3)
    histname_rec.GetYaxis().SetRangeUser(1., 6000.);
    histname_rec.SetStats(0)

    histname_rec.GetXaxis().SetTitle("mass [GeV]")
    histname_rec.GetYaxis().SetTitle("Events")

    histname_gen.SetFillColor(0)
    histname_gen.SetLineColor(ROOT.kRed)
    histname_gen.SetLineWidth(2)
    histname_gen.SetLineStyle(2)

    leg = TLegend(.63,.7,.85,.9)
    leg.AddEntry(histname_rec,"Det-Lvl")
    leg.AddEntry(histname_gen,"Part-Lvl")
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)

    histname_rec.Draw("HIST")
    histname_gen.Draw("HIST SAME")
    leg.Draw()
    canvas.Update()
    canvas.SaveAs(plotdir+"Run2018_"+subdir+".pdf")