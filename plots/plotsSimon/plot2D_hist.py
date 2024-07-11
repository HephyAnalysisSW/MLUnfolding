import os
import ROOT
from ROOT import gStyle, TCanvas, TH1D, TLegend, kRed, kBlue
from MLUnfolding.Tools.user                              import plot_directory
from MLUnfolding.Tools.helpers                           import getObjFromFile

ROOT.gROOT.SetBatch(ROOT.kTRUE)

plotdir = plot_directory+"/TestPlots/"
if not os.path.exists( plotdir ): os.makedirs( plotdir )

filename = "/groups/hephy/cms/simon.hablas/www/MLUnfolding/plots/analysisPlots/MLUnfolding_v3/Run2018/lin/rec_pass-gen_pass/Results.root"

histname_2d = "TransfereMatrix__TTbar"
histname_2d = getObjFromFile(filename, histname_2d)

projection = histname_2d.ProjectionX("proj1", 1, histname_2d.GetXaxis().GetNbins())
for i in range(histname_2d.GetXaxis().GetNbins()) :
    for j in range(histname_2d.GetYaxis().GetNbins()) :
        cont = histname_2d.GetBinContent(i+1,j+1)
        if projection.GetBinContent(i+1) != 0 :
            histname_2d.SetBinContent(i+1,j+1,cont/projection.GetBinContent(i+1))

canvas = ROOT.TCanvas("canvas", "canvas",800,800)
histname_2d.SetFillColor(ROOT.kAzure+7)
histname_2d.SetTitle("Invariant jet mass ;m_{par} [GeV];m_{det} [GeV]")

histname_2d.GetYaxis().SetTitleOffset(1.3)
histname_2d.GetXaxis().SetNdivisions(505)
histname_2d.GetYaxis().SetNdivisions(505)  

histname_2d.SetFillColor(0)
histname_2d.SetLineColor(ROOT.kRed)
histname_2d.SetLineWidth(2)
histname_2d.SetLineStyle(2)
histname_2d.SetStats(0)

canvas.SetRightMargin(0.2)
canvas.SetLeftMargin(0.2)
canvas.SetBottomMargin(0.2)
canvas.SetTopMargin(.1)

ROOT.gStyle.SetPalette(ROOT.kRainBow)

histname_2d.Draw("colz")

canvas.SaveAs(plotdir+"Run2018_transfereMat.pdf")


