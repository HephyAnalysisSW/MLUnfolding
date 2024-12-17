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

from MLUnfolding.Tools.user  import plot_directory
#plot_directory = "./plots"

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel', action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--plot_dir',    action='store', type=str, default="MLUnfolding_tmp") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--f171',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--f172',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--f173',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy

args = argParser.parse_args()

gc.collect() #SH Test Garbage Collection
print(psutil.Process().memory_info().rss / (1024 * 1024))
print("Hi")

plot_dir = os.path.join(plot_directory, args.plot_dir)# Zusammenpasten von plot_dir
if not os.path.exists( plot_dir ): os.makedirs( plot_dir )


#SH To Avoid Index Confusion
zeta_gen_index = 0
zeta_rec_index = 1
weight_gen_index = 2
weight_rec_index = 3
pt_gen_index = 4
pt_rec_index = 5
mass_gen_index= 6

gen_index = [zeta_gen_index,weight_gen_index,pt_gen_index]
rec_index = [zeta_rec_index,weight_rec_index,pt_rec_index]

zeta_sample_index = 0
weight_sample_index = 1
pt_sample_index = 2


sample_index = [zeta_sample_index,weight_sample_index,pt_sample_index]

print("Started")
try :
    with open(args.f171, "rb") as f:
        f171 = np.load(f)
        f.close()
except FileNotFoundError :
    print("File \""+ args.f171+"\" not found.")
    exit(1)

try :
    with open(args.f172, "rb") as f:
        f172 = np.load(f)
        f.close()
except FileNotFoundError :
    print("File \""+ args.f171+"\" not found.")
    exit(1)
    
try :
    with open(args.f173, "rb") as f:
        f173 = np.load(f)
        f.close()
except FileNotFoundError :
    print("File \""+ args.f171+"\" not found.")
    exit(1)


fig, axs =  plt.subplots(1, 3, sharex = "col", tight_layout=True,figsize=(15,5), gridspec_kw=
                                dict(height_ratios=[1],
                                      width_ratios=[1,1,1]))
# #SH: Add Ticks to every side,
# for ax_row in axs:
    # for ax in ax_row:
        # ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        
number_of_bins = 20
lower_border = 100 * 0
upper_border = 100 * 7
step = (upper_border - lower_border) // number_of_bins
n_bins = [x / 100.0 for x in range(lower_border,upper_border+1,step)] 
        
# Assume f172, f171, f173 are your arrays, and zeta_gen_index and n_bins are defined
count_172 = f172.shape[0]
count_171 = f171.shape[0]
count_173 = f173.shape[0]
sum_172 = np.sum(f172[:, weight_gen_index])
sum_171 = np.sum(f171[:, weight_gen_index])
sum_173 = np.sum(f173[:, weight_gen_index])
min_count = min(count_172, count_171, count_173)
min_sum = min(sum_172, sum_171, sum_173)
factor_172 = min_count / count_172
factor_171 = min_count / count_171
factor_173 = min_count / count_173
sum_factor_172 = min_sum / sum_172
sum_factor_171 = min_sum / sum_171
sum_factor_173 = min_sum / sum_173

lower_border = 100 * 0
upper_border = 100 * 7
step = (upper_border - lower_border) // number_of_bins
n_bins = [x / 100.0 for x in range(lower_border,upper_border+1,step)] 
  
# Compute histograms
hist1, _ = np.histogram(f172[:, zeta_gen_index], bins=n_bins)
hist2, _ = np.histogram(f171[:, zeta_gen_index], bins=n_bins)
hist3, _ = np.histogram(f173[:, zeta_gen_index], bins=n_bins)

# Reweight histograms
hist1 = hist1 * factor_172
hist2 = hist2 * factor_171
hist3 = hist3 * factor_173

hep.histplot(hist1,       n_bins, ax=axs[0],color = "grey",alpha = 0.5, label = "m_t = 172.5GeV (alt)", histtype="fill")
hep.histplot(hist2,       n_bins, ax=axs[0],color = "blue",           label = "m_t = 171.5GeV") 
hep.histplot(hist3,       n_bins, ax=axs[0],color = "yellow",         label = "m_t = 173.5GeV"    )

#axs[0].set_yscale("log")
axs[0].legend(frameon = False, fontsize="18")
axs[0].set_xlabel("zeta")
axs[0].set_ylabel("Events")
axs[0].set_ylim([0, 4e4])
axs[0].set_xlim([lower_border/100, upper_border/100])


lower_border = 100 * 150
upper_border = 100 * 190
step = (upper_border - lower_border) // number_of_bins
n_bins = [x / 100.0 for x in range(lower_border,upper_border+1,step)] 
  
# Compute histograms
hist1, _ = np.histogram(f172[:, mass_gen_index], bins=n_bins)
hist2, _ = np.histogram(f171[:, mass_gen_index], bins=n_bins)
hist3, _ = np.histogram(f173[:, mass_gen_index], bins=n_bins)

# Reweight histograms
hist1 = hist1 * factor_172
hist2 = hist2 * factor_171
hist3 = hist3 * factor_173

hep.histplot(hist1,       n_bins, ax=axs[1],color = "grey",alpha = 0.5, label = "m_t = 172.5GeV (alt)", histtype="fill")
hep.histplot(hist2,       n_bins, ax=axs[1],color = "blue",           label = "m_t = 171.5GeV") 
hep.histplot(hist3,       n_bins, ax=axs[1],color = "yellow",         label = "m_t = 173.5GeV"    )

#axs[1].set_yscale("log")
axs[1].legend(frameon = False, fontsize="18")
axs[1].set_xlabel("m_t")
axs[1].set_ylabel("Events")
axs[1].set_ylim([0, 4e4])
axs[1].set_xlim([lower_border/100, upper_border/100])


lower_border = 100 * 0
upper_border = 100 * 7
step = (upper_border - lower_border) // number_of_bins
n_bins = [x / 100.0 for x in range(lower_border,upper_border+1,step)] 
  
# Compute histograms
hist1, _ = np.histogram(f172[:, zeta_gen_index],weights = f172[:, weight_gen_index] , bins=n_bins)
hist2, _ = np.histogram(f171[:, zeta_gen_index],weights = f171[:, weight_gen_index], bins=n_bins)
hist3, _ = np.histogram(f173[:, zeta_gen_index],weights = f173[:, weight_gen_index], bins=n_bins)

# Reweight histograms
hist1 = hist1 * sum_factor_172
hist2 = hist2 * sum_factor_171
hist3 = hist3 * sum_factor_173

hep.histplot(hist1,       n_bins, ax=axs[2],color = "grey",alpha = 0.5, label = "m_t = 172.5GeV (alt)", histtype="fill")
hep.histplot(hist2,       n_bins, ax=axs[2],color = "blue",           label = "m_t = 171.5GeV") 
hep.histplot(hist3,       n_bins, ax=axs[2],color = "yellow",         label = "m_t = 173.5GeV"    )

#axs[2].set_yscale("log")
axs[2].legend(frameon = False, fontsize="18")
axs[2].set_xlabel("zeta weighted")
axs[2].set_ylabel("Events")
#axs[2].set_ylim([0, 4e4])
axs[2].set_xlim([lower_border/100, upper_border/100])




plt.savefig(plot_dir+"/compare.png")
plt.close("all")
