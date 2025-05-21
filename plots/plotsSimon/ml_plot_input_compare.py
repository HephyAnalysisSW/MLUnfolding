print("Start of Script")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import torch
from torch import nn
from torch import optim
import matplotlib as mpl
import mplhep as hep
import sys
from datetime import datetime
import os
import plotutils
import transformations as trf
import psutil
import gc

from indices import (
    zeta_gen_index,
    zeta_rec_index,
    weight_gen_index,
    weight_rec_index,
    pt_gen_index,
    pt_rec_index,
    mass_gen_index,
    mass_jet_gen_index,
    gen_index,
    rec_index,
    zeta_sample_index,
    weight_sample_index,
    
    sample_index
)

from ml_functions import (calc_squared_weights,calculate_chisq)
from MLUnfolding.Tools.user  import plot_directory
#plot_directory = "./plots"

from matplotlib.ticker import FuncFormatter

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel', action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--train',  action='store', type=str, default="NA")
argParser.add_argument('--val',    action='store', type=str, default="NA")
argParser.add_argument('--plot_dir',    action='store', type=str, default="Input_Compare") 
argParser.add_argument('--file1',    action='store', type=str, default="") # File Path
argParser.add_argument('--file2',    action='store', type=str, default="", help="2nd Input file path") 
argParser.add_argument('--file3',    action='store', type=str, default="") 
argParser.add_argument('--file4',    action='store', type=str, default="") 
argParser.add_argument('--l1',    action='store', type=str, default="") #labels
argParser.add_argument('--l2',    action='store', type=str, default="") 
argParser.add_argument('--l3',    action='store', type=str, default="", help = "3rd label") 
argParser.add_argument('--l4',    action='store', type=str, default="") 
argParser.add_argument('--c1',    action='store', type=float, default=0) 
argParser.add_argument('--c2',    action='store', type=float, default=0) 
argParser.add_argument('--c3',    action='store', type=float, default=0) 
argParser.add_argument('--c4',    action='store', type=float, default=0, help ="4th Weight Cut value (default 0, recommended 1.5e-5)") 

args = argParser.parse_args()



plt_w = 1 # SH: plot Weight
plt_wz = 2#SH: plot weighted Zeta
plt_z = 0#
plt_d = 0 #SH: Plot Data
plt_r = 0 #SH: Plot Ration Pad

plot_dir = os.path.join(plot_directory,args.plot_dir)# Zusammenpasten von plot_dir
print(plot_dir)
if not os.path.exists( plot_dir ): os.makedirs( plot_dir )

# Original file inputs
files = [args.file1, args.file2, args.file3, args.file4]
labels = [args.l1, args.l2, args.l3, args.l4]
cuts = [args.c1, args.c2, args.c3, args.c4]

cmap = plt.get_cmap("Set1")

# Extract the first 4 colors
colors = [cmap(i) for i in range(4)]

# Filter out empty strings
files_logic = [f != "" for f in files]
files = [f for f, keep in zip(files, files_logic) if keep]
colors = [c for c, keep in zip(colors, files_logic) if keep]
labels = [l for l, keep in zip(labels, files_logic) if keep]
cuts = [cut for cut, keep in zip(cuts, files_logic) if keep]

# Load matrices and keep only successful ones
matrices = []
valid_colors = []
valid_labels = []

for f, c, l, cut in zip(files, colors, labels, cuts):
    try:
        mat = np.load(f)
         #Apply Weight Cut
        
        matrices.append(mat)
        valid_colors.append(c)
        valid_labels.append(l)
        valif_cuts.append(cut)
        #print(f"Could load {f}. Colored {c}, Labeled {l} ")
    except Exception as e:
        print(f"Could not load {f}: {e}")

# Now you have:
# - `matrices`: list of loaded numpy arrays
# - `valid_colors`: matching colors
# - `valid_labels`: matching labels

# #_________________________________________________________________________________________________________________
# #--SH: Start Plot in Mathplotlib
# fig, axs =  plt.subplots(2, 3, sharex = "col", tight_layout=True,figsize=(15, 6), gridspec_kw=
                                # dict(height_ratios=[6, 1],
                                      # width_ratios=[1, 1, 1]))
# #SH: Add Ticks to every side
# for ax_row in axs:
    # for ax in ax_row:
        # ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
# --SH: Start Plot in Matplotlib
fig, axs = plt.subplots(1, 3, sharex="col", tight_layout=True, figsize=(15, 5), 
                        gridspec_kw=dict(width_ratios=[1, 1, 1]))

# Ensure axs is iterable (in case it's returned as a 1D array)
for ax in axs:
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)        

#--SH: Plot Zeta`
number_of_bins = 20
upper_border = 7
upper_border = upper_border *100
step = upper_border // number_of_bins
n_bins = [x / 100.0 for x in range(0,upper_border+1,step)] 

for m, c, l in zip(matrices, valid_colors, valid_labels):
    hist1,bin_edges = np.histogram(m[:,zeta_gen_index], bins= n_bins)
    hep.histplot(hist1,       n_bins, ax=axs[plt_z],color = c,alpha = 1 ,      label = l)#, histtype="fill")


#_________________________________________________________________________________________________________________
#--SH: Plot Weight
upper_border = 1000 #300
step = upper_border // number_of_bins
n_bins = [x / 10000000.0 for x in range(0,upper_border+1,step)]

for m, c, l in zip(matrices, valid_colors, valid_labels):
    hist1,bin_edges = np.histogram(m[:,weight_gen_index], bins= n_bins)
    hep.histplot(hist1,       n_bins, ax=axs[plt_w],color = c,alpha = 1 ,      label = l)#, histtype="fill")

#_________________________________________________________________________________________________________________
#--SH: Plot Weighted Zeta'
upper_border = 7
upper_border = upper_border *100
step = upper_border // number_of_bins
n_bins = [x / 100.0 for x in range(0,upper_border+1,step)] 



for m, c, l in zip(matrices, valid_colors, valid_labels):
    hist1,bin_edges = np.histogram(m[:,zeta_gen_index],weights= m[:,weight_gen_index], bins= n_bins)
    hep.histplot(hist1,       n_bins, ax=axs[plt_wz],color = c,alpha = 1 ,      label = l)#, histtype="fill")

#_________________________________________________________________________________________________________________
#--SH: Plot Style and Axis
axs[plt_z].set_yscale("log")
axs[plt_w].set_yscale("log")

axs[plt_w].legend(frameon = False, fontsize="18")

# axs[plt_z].grid(axis="both")
# axs[plt_w].grid(axis="both")
# axs[plt_wz].grid(axis="both")

formatter = FuncFormatter(lambda x, _: f'{x:.0e}')
axs[plt_w].xaxis.set_major_formatter(formatter)

axs[plt_z].set_ylim([1, 1e7])
axs[plt_w].set_ylim([1e3, 1e6])
axs[plt_z].set_xlim([0, 7])
axs[plt_w].set_xlim([0, 0.0001])
#axs[0,plt_w].set_xlim([0, 0.00003])
axs[plt_wz].set_xlim([0, 7])

xlabels = [
    r"$\zeta$ * pt$^2$ / 172.5$^2$",
    r"weight ($ \prod \frac{p_i}{p_t}$)",
    r"$\zeta$ * pt$^2$ / 172.5$^2$"
]
ylabels = ["Events","Events","Events (weighted)"]

for col in range(3):
    axs[ col].text(
        1.02, -0.1,
        xlabels[col],
        transform=axs[col].transAxes,
        fontsize=16,
        va="top", ha="right"
    )
    axs[ col].text(
        -0.1, 1.0,
        ylabels[col],
        transform=axs[col].transAxes,
        fontsize=16,
        va="top", ha="right",
        rotation=90
    )
   
now = datetime.now()
current_time = now.strftime("%H_%M_%S")
plt.savefig(plot_dir+"/input_compare"+current_time+".png")
plt.close("all")