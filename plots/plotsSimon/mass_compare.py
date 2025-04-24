import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
import sys
import gc
import mplhep as hep
import psutil
from datetime import datetime
import os
#from __future__ import print_function
import transformations as trf                             

from MLUnfolding.Tools.user  import plot_directory

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel', action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--file_m3',     action='store', type=str, default="NA")
argParser.add_argument('--file_m2',     action='store', type=str, default="NA")
argParser.add_argument('--file_m1',     action='store', type=str, default="NA")

argParser.add_argument('--file',     action='store', type=str, default="NA")
argParser.add_argument('--val',     action='store', type=str, default="NA")

argParser.add_argument('--file_p1',     action='store', type=str, default="NA")
argParser.add_argument('--file_p2',     action='store', type=str, default="NA")
argParser.add_argument('--file_p3',     action='store', type=str, default="NA")

argParser.add_argument('--plot_dir',    action='store', type=str, default="MLUnfolding_tmp") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--shift', action='store', type=float, default=0.0) 
args = argParser.parse_args()

plot_dir = os.path.join(plot_directory, args.plot_dir)# Zusammenpasten von plot_dir
if not os.path.exists( plot_dir ): os.makedirs( plot_dir )


#SH To Avoid Index Confusion
zeta_gen_index = 0
zeta_rec_index = 1
weight_gen_index = 2
weight_rec_index = 3
pt_gen_index = 4
pt_rec_index = 5
mass_gen_index = 6

gen_index = [zeta_gen_index,weight_gen_index,pt_gen_index]
rec_index = [zeta_rec_index,weight_rec_index,pt_rec_index]

zeta_sample_index = 0
weight_sample_index = 1
pt_sample_index = 2

#SH Falls generierte Files verwendet
zeta_gen_index = zeta_sample_index
weight_gen_index =weight_sample_index
pt_gen_index = pt_sample_index
mass_gen_index = 0


files = [args.file_m3, args.file_m2, args.file_m1, args.file_p1, args.file_p2, args.file_p3]
colors = np.array(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
files = [args.file_m2,args.file_p2]
colors = np.array([ "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])

try :
    with open(args.file, "rb") as f:
        data = np.load(f)
        f.close()
        print("Shape is", data.shape)
except FileNotFoundError :
    print("File "+ args.file+" (Data) not found.")
    exit(1)
    
    
#data[:, weight_gen_index] > 1.5e-5


# Extract directory, filename, and extension
dir_name, base_name = os.path.split(args.file)  # Splits into directory and filename
name, ext = os.path.splitext(base_name)  # Splits into name and extension

#SH: add new m_top to filename
plot_filename = "unfolded_zetaprime_compare.png"
new_plot_path = os.path.join(dir_name, plot_filename)

plt_w = 1 # SH: plot Weight
plt_wz = 2 #SH: plot weighted Zeta
plt_z = 0#

plt_d = 0 #SH: Plot Data
plt_r = 1 #SH: Plot Ration Pad


#SH: Plot Everything for sanitycheck
fig, axs =  plt.subplots(2, 3, sharex = "col", tight_layout=True,figsize=(15, 6), gridspec_kw=
                                dict(height_ratios=[6, 1],
                                      width_ratios=[1, 1, 1]))
                                      
#SH: Add Ticks to every side
for ax_row in axs:
    for ax in ax_row:
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        
fig.suptitle('Comparison of unfolded zeta\' ')

number_of_bins = 20
lower_border_1 = 100
upper_border_1 = 220
upper_border = upper_border_1 *100
lower_border = lower_border_1 *100
step =  (upper_border- lower_border) // (number_of_bins*2)
n_bins = [x / 100.0 for x in range(lower_border,upper_border+1,step)] 

hist1_ori,_ = np.histogram(data[:,mass_gen_index], bins= n_bins)
#hep.histplot(hist1_ori,n_bins, ax=axs[0,0],color = "grey",   label = "m_t = 172.5 GeV")

upper_border_2 = 6
upper_border = upper_border_2 * 1000
upper_border_2 = upper_border_2 / 10000
step = upper_border // (number_of_bins)
n_bins = [x / 10000000.0 for x in range(0,upper_border+1,step)]

hist2_ori,_ = np.histogram(data[:,weight_gen_index],bins= n_bins)
hep.histplot(hist2_ori,n_bins, ax=axs[plt_d,plt_w],color = "grey",label = "m_t = 172.5 GeV")

upper_border_3 = 7
upper_border = upper_border_3 *100
step = upper_border // number_of_bins
n_bins = [x / 100.0 for x in range(0,upper_border+1,step)] 

hist3_ori,_ = np.histogram(data[:,zeta_gen_index] , weights= data[:,weight_gen_index] , bins= n_bins)
hep.histplot(hist3_ori,n_bins, ax=axs[plt_d,plt_wz],color = "grey", label = "m_t = 172.5 GeV")

area_ori = np.sum(data[:,weight_gen_index])
print("m_t = 172.5 GeV",area_ori)

alpha = 0.8




# --------------------------------------------------------------------------------------------
for count, for_file in enumerate(files):
    print(count, for_file)
    

    try :
        with open(for_file, "rb") as f:
            data_for = np.load(f)
            f.close()
    except FileNotFoundError :
        print("File "+ for_file+" (Data) not found.")
        exit(1)
    
    dir_name, base_name = os.path.split(for_file)  # Splits into directory and filename
    name, ext = os.path.splitext(base_name)  # Splits into name and extension

    label = "m_t = " + name[-5:].replace("p",".") + "GeV"
    label = "m_t = " + for_file[69:74].replace("p",".") + "GeV"
    color = colors[count]
    
    scale = area_ori / np.sum(data_for[:,weight_gen_index])
    
    data_for[:,weight_gen_index] *= scale

    number_of_bins = 20
    lower_border_1 = 100
    upper_border_1 = 220
    upper_border = upper_border_1 *100
    lower_border = lower_border_1 *100
    step =  (upper_border- lower_border) // (number_of_bins*2)
    n_bins = [x / 100.0 for x in range(lower_border,upper_border+1,step)] 


    #hist1,_ = np.histogram(data_for[:,mass_gen_index], weights =  (data[:,weight_gen_index] / data_for[:,weight_gen_index]) , bins= n_bins)
    #hist_1b = np.divide(hist1, hist1_ori, where=hist1_ori!=0)

    #hep.histplot(hist1,n_bins, ax=axs[0,0],color = color)
    #hep.histplot(hist_1b, n_bins, ax=axs[1,0],color = color) 

    upper_border_2 = 6
    upper_border = upper_border_2 * 1000
    upper_border_2 = upper_border_2 / 10000
    step = upper_border // (number_of_bins)
    n_bins = [x / 10000000.0 for x in range(0,upper_border+1,step)]

    hist2,_ = np.histogram(data_for[:,weight_gen_index],bins= n_bins)
    hist_2b = np.divide(hist2, hist2_ori, where=hist2_ori!=0)

    hep.histplot(hist2,n_bins, ax=axs[plt_d,plt_w],color = color,label = label, alpha = alpha)
    hep.histplot(hist_2b, n_bins, ax=axs[plt_r,plt_w],color = color) 

    upper_border_3 = 7
    upper_border = upper_border_3 *100
    step = upper_border // number_of_bins
    n_bins = [x / 100.0 for x in range(0,upper_border+1,step)] 

    hist3,bin_edges = np.histogram(data_for[:,zeta_gen_index] , weights= data_for[:,weight_gen_index] , bins= n_bins)
    hist_3b = np.divide(hist3, hist3_ori, where=hist3_ori!=0)

    hep.histplot(hist3,n_bins, ax=axs[plt_d,plt_wz],color = color, alpha = alpha)
    hep.histplot(hist_3b, n_bins, ax=axs[plt_r,plt_wz],color = color) 
    
    print(label,np.sum(data_for[:,weight_gen_index]))
    
    if True:
        poly_bins = bin_edges[6:15]  # Use bin_edges, NOT n_bins
        bin_centers = (poly_bins[:-1] + poly_bins[1:]) / 2  # Compute bin centers
        hist_values = hist3[6:14]  # Match bins (1 less than edges)
        
        print("bin_centers:",bin_centers)
        print("hist_values:",hist_values)
        
        # Quadratic fit
        coeffs = np.polyfit(bin_centers, hist_values, 2)
        x_smooth = np.linspace(bin_centers[0], bin_centers[-1], 100)
        y_smooth = np.polyval(coeffs, x_smooth)

        # Plot quadratic fit
        axs[plt_d, plt_wz].plot(x_smooth, y_smooth, color=color, linestyle='--', alpha=0.5)


axs[plt_d,plt_z].set_yscale("log")
axs[plt_d,plt_w].set_yscale("log")

axs[plt_d,plt_z].legend(frameon = False, fontsize="18")
axs[plt_d,plt_w].legend(frameon = False, fontsize="18")
axs[plt_d,plt_wz].legend(frameon = False, fontsize="14", loc=8)

axs[plt_r,plt_z].set_xlabel("m$_{jet}$") #("$\zeta$ * pt$^2$ / 172.5$^2$")
axs[plt_r,plt_w].set_xlabel("weight ($ \\prod \\frac{p_i}{p_t}$)")
axs[plt_r,plt_wz].set_xlabel("$\zeta$ * pt$^2$ / 172.5$^2$")
axs[plt_r,plt_z].set_ylabel(" $\\frac{\\mathrm{ML Particle Lvl}}{\\mathrm{Val Particle Lvl}}$")
axs[plt_r,plt_w].set_ylabel(" $\\frac{\\mathrm{ML Particle Lvl}}{\\mathrm{Val Particle Lvl}}$")
axs[plt_r,plt_wz].set_ylabel(" $\\frac{\\mathrm{ML Particle Lvl}}{\\mathrm{Val Particle Lvl}}$")
axs[plt_d,plt_z].set_ylabel("Events")
axs[plt_d,plt_w].set_ylabel("Events")
axs[plt_d,plt_wz].set_ylabel("Events (weighted)")

axs[plt_r,plt_z].grid(axis = "y")
axs[plt_r,plt_w].grid(axis = "y")
axs[plt_r,plt_wz].grid(axis = "y")

axs[plt_r,plt_z].set_ylim([0.5, +2])
axs[plt_r,plt_w].set_ylim([0.5, +2])
axs[plt_r,plt_wz].set_ylim([0.5, +2])

#axs[0,0].set_ylim([1, 1e5])
axs[plt_d,plt_w].set_ylim([1e2, 1e6])
axs[plt_d,plt_z].set_xlim([lower_border_1, upper_border_1])
axs[plt_d,plt_w].set_xlim([0, upper_border_2])
axs[plt_d,plt_wz].set_xlim([0, upper_border_3])


plt.savefig(new_plot_path)
plt.close("all")

