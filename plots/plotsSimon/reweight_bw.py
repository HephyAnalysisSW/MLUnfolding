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
argParser.add_argument('--file_in',     action='store', type=str, default="NA")
argParser.add_argument('--file_out',    action='store', type=str, default="NA") 
argParser.add_argument('--plot_dir',    action='store', type=str, default="MLUnfolding_tmp") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--shift', action='store', type=float, default=0.0) 
args = argParser.parse_args()

plot_dir = os.path.join(plot_directory, args.plot_dir)# Zusammenpasten von plot_dir
if not os.path.exists( plot_dir ): os.makedirs( plot_dir )

def get_k(Gamma,m):
    gam = np.sqrt(m**2 *(m**2 + Gamma**2))
    k = 2 * np.sqrt(2) * Gamma * gam / ( np.pi * np.sqrt(m**2 + gam) )
    return k

def bw_reweight(s,m_old,m_new,Gamma,k_old,k_new):
    k = k_new / k_old
    
    a =  ((s**2 - m_old**2)**2 + (m_old**2 * Gamma**2)) / ((s**2 - m_new**2)**2+ (m_new**2 * Gamma**2)) #SH: Beware of denomminator
    return a*k

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

print("Reweight File in",args.file_in)
try :
    with open(args.file_in, "rb") as f:
        data = np.load(f)
        f.close()
except FileNotFoundError :
    print("File "+ args.file_in+" (Data) not found.")
    exit(1)

a = 500 
b = 525

#data = data[(data[:, pt_gen_index] >= a) & (data[:, pt_gen_index] < b)]

#SH: Copy Data
data_ori = data.copy()
    
#SH: Perform the Breit Wigner Shift:
m_old = 172.5
m_new = m_old + args.shift
Gamma = 1.3

k_old = get_k(Gamma,m_old)
k_new = get_k(Gamma,m_new)


#SH: Get New Weights
new_weights = bw_reweight(data[:,mass_gen_index],m_old,m_new,Gamma,1,1)

#SH: Apply new weights

test_sum_gen_before = np.sum(data[:, weight_gen_index])
test_sum_rec_before = np.sum(data[:, weight_rec_index])

data[:, weight_gen_index] *= new_weights
data[:, weight_rec_index] *= new_weights

test_sum_gen_after = np.sum(data[:, weight_gen_index])
test_sum_rec_after = np.sum(data[:, weight_rec_index])

print("Gen - Weight - Factor= ", test_sum_gen_after/test_sum_gen_before)
print("Rec - Weight - Factor= ", test_sum_rec_after/test_sum_rec_before)

print("Shape of data:", data.shape)
print("Shape of data[:, weight_gen_index]:", data[:, weight_gen_index].shape)
print("Shape of new_weights:", new_weights.shape)


print(args.shift)

# Extract directory, filename, and extension
dir_name, base_name = os.path.split(args.file_in)  # Splits into directory and filename
name, ext = os.path.splitext(base_name)  # Splits into name and extension

#SH: add new m_top to filename
new_filename = name + str(m_new).replace(".", "p") + ext
plot_filename = name + str(m_new).replace(".", "p") + ".png"
new_path = os.path.join(dir_name, new_filename)
new_plot_path = os.path.join(dir_name, plot_filename)


#SH: Plot Everything for sanitycheck
fig, axs =  plt.subplots(2, 3, sharex = "col", tight_layout=True,figsize=(15, 6), gridspec_kw=
                                dict(height_ratios=[6, 1],
                                      width_ratios=[1, 1, 1]))
                                      
#SH: Add Ticks to every side
for ax_row in axs:
    for ax in ax_row:
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        
fig.suptitle("m_t"+ str(m_new) +"GeV")
    
number_of_bins = 40
lower_border_1 = 100
upper_border_1 = 220
upper_border = upper_border_1 *100
lower_border = lower_border_1 *100
step =  (upper_border- lower_border) // (number_of_bins)
n_bins = [x / 100.0 for x in range(lower_border,upper_border+1,step)] 

hist1,_ = np.histogram(data_ori[:,mass_gen_index], bins= n_bins)
hist2,_ = np.histogram(    data[:,mass_gen_index],weights = new_weights, bins= n_bins)
hist5 = np.divide(hist1, hist2, where=hist2!=0)

hep.histplot(hist1,n_bins, ax=axs[0,0],color = "blue",   label = "Rec Ori", histtype="fill", alpha = 0.5)
hep.histplot(hist2,n_bins, ax=axs[0,0],color = "blue",   label = "Rec Mod") 
hep.histplot(hist5, n_bins, ax=axs[1,0],color = "blue", alpha = 0.5) 

upper_border_2 = 6
upper_border = upper_border_2 * 1000
upper_border_2 = upper_border_2 / 10000
step = upper_border // (number_of_bins)
n_bins = [x / 10000000.0 for x in range(0,upper_border+1,step)]


hist1,_ = np.histogram(data_ori[:,weight_rec_index],bins= n_bins)
hist2,_ = np.histogram(    data[:,weight_rec_index] , bins= n_bins)
hist3,_ = np.histogram(data_ori[:,weight_gen_index],bins= n_bins)
hist4,_ = np.histogram(    data[:,weight_gen_index] , bins= n_bins)
hist5 = np.divide(hist1, hist2, where=hist2!=0)
hist6 = np.divide(hist3, hist4, where=hist4!=0)

hep.histplot(hist1,n_bins, ax=axs[0,1],color = "blue",label = "Rec Ori", histtype="fill", alpha = 0.5 )
hep.histplot(hist2,n_bins, ax=axs[0,1],color = "blue",label = "Rec Mod") 
hep.histplot(hist3,n_bins, ax=axs[0,1],color = "green", label = "Gen Ori"   , histtype="fill", alpha = 0.5 )
hep.histplot(hist4,n_bins, ax=axs[0,1],color = "green", label = "Gen Mod"    )
hep.histplot(hist5, n_bins, ax=axs[1,1],color = "blue", alpha = 0.5)   
hep.histplot(hist5, n_bins, ax=axs[1,1],color = "green", alpha = 0.5) 

upper_border_3 = 7
upper_border = upper_border_3 *100
step = upper_border // number_of_bins
n_bins = [x / 100.0 for x in range(0,upper_border+1,step)] 


hist1,_ = np.histogram(data_ori[:,zeta_rec_index] , weights= data_ori[:,weight_rec_index] , bins= n_bins)
hist2,_ = np.histogram(    data[:,zeta_rec_index] , weights=     data[:,weight_rec_index] , bins= n_bins)
hist3,_ = np.histogram(data_ori[:,zeta_gen_index] , weights= data_ori[:,weight_gen_index] , bins= n_bins)
hist4,_ = np.histogram(    data[:,zeta_gen_index] , weights=     data[:,weight_gen_index] , bins= n_bins)
hist5 = np.divide(hist1, hist2, where=hist2!=0)
hist6 = np.divide(hist3, hist4, where=hist4!=0)

hep.histplot(hist1,n_bins, ax=axs[0,2],color = "blue", label = "Rec Ori", histtype="fill", alpha = 0.5)
hep.histplot(hist2,n_bins, ax=axs[0,2],color = "blue", label = "Rec Mod") 
hep.histplot(hist3,n_bins, ax=axs[0,2],color = "green", label = "Gen Ori", histtype="fill", alpha = 0.5)
hep.histplot(hist4,n_bins, ax=axs[0,2],color = "green", label = "Gen Mod")
hep.histplot(hist5, n_bins, ax=axs[1,2],color = "blue", alpha = 0.5)   
hep.histplot(hist5, n_bins, ax=axs[1,2],color = "green", alpha = 0.5) 

axs[0,0].set_yscale("log")
axs[0,1].set_yscale("log")

axs[0,0].legend(frameon = False, fontsize="18")
axs[0,1].legend(frameon = False, fontsize="18")
axs[0,2].legend(frameon = False, fontsize="14", loc=8)

axs[0,0].set_xlabel("m$_{jet}$") #("$\zeta$ * pt$^2$ / 172.5$^2$")
axs[0,1].set_xlabel("weight ($ \\prod \\frac{p_i}{p_t}$)")
axs[0,2].set_xlabel("$\zeta$ * pt$^2$ / 172.5$^2$")
axs[1,0].set_ylabel(" $\\frac{\\mathrm{ML Particle Lvl}}{\\mathrm{Val Particle Lvl}}$")
axs[1,1].set_ylabel(" $\\frac{\\mathrm{ML Particle Lvl}}{\\mathrm{Val Particle Lvl}}$")
axs[1,2].set_ylabel(" $\\frac{\\mathrm{ML Particle Lvl}}{\\mathrm{Val Particle Lvl}}$")
axs[0,0].set_ylabel("Events")
axs[0,1].set_ylabel("Events")
axs[0,2].set_ylabel("Events (weighted)")

axs[1,0].grid(axis = "y")
axs[1,1].grid(axis = "y")
axs[1,2].grid(axis = "y")

axs[1,0].set_ylim([0.5, +2])
axs[1,1].set_ylim([0.5, +2])
axs[1,2].set_ylim([0.5, +2])

#axs[0,0].set_ylim([1, 1e5])
axs[0,1].set_ylim([1e2, 1e6])
axs[0,0].set_xlim([lower_border_1, upper_border_1])
axs[0,1].set_xlim([0, upper_border_2])
axs[0,2].set_xlim([0, upper_border_3])

plt.savefig(new_plot_path)
plt.close("all")

with open(new_path, 'wb') as f3:
    np.save(f3, data)
del data
