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
from datetime import datetime

from MLUnfolding.Tools.user  import plot_directory
#plot_directory = "./plots"

# Function to calculate sum of squared weights for each bin
def calc_squared_weights(data, weights, bins):
    squared_weights = np.square(weights)
    hist_squared, _ = np.histogram(data, bins=bins, weights=squared_weights)
    return hist_squared

def calculate_chisq(hist1, hist2, hist_squared1):
    # Create a mask to exclude rows where hist_squared1 is 0
    valid_mask = hist_squared1 != 0
    # Apply the mask to the arrays
    chi_squared = np.sum(
        np.square(hist1[valid_mask] - hist2[valid_mask]) / hist_squared1[valid_mask]
    )
    return chi_squared
    
def get_k(Gamma,m):
    gam = np.sqrt(m**2 *(m**2 + Gamma**2))
    k = 2 * np.sqrt(2) * Gamma * gam / ( np.pi * np.sqrt(m**2 + gam) )
    return k

def bw_reweight(s,m_old,m_new,Gamma,k_old,k_new):
    k = k_new / k_old
    
    a =  ((s**2 - m_old**2)**2 + (m_old**2 * Gamma**2)) / ((s**2 - m_new**2)**2+ (m_new**2 * Gamma**2)) #SH: Beware of denomminator
    return a*k


import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel', action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--train',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--val',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--unfolded',    action='store', type=str, default="NA")
argParser.add_argument('--plot_dir',    action='store', type=str, default="MLUnfolding_tmp") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--info',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--weight_cut', action='store', type=float, default=0.0) # ./mldata/ML_Data_validate.npy
argParser.add_argument('--text_debug',    action='store', type=bool, default=True) #./mldata/ML_Data_validate.npy
argParser.add_argument('--shift', action='store', type=float, default=0.0) 

args = argParser.parse_args()

gc.collect() #SH Test Garbage Collection
print(psutil.Process().memory_info().rss / (1024 * 1024))
print("Hi")

w_cut = args.weight_cut
text_debug= args.text_debug

# the nflows functions what we will need in order to build our flow
from nflows.flows.base import Flow # a container that will wrap the parts that make up a normalizing flow
from nflows.distributions.normal import StandardNormal # Gaussian latent space distribution
from nflows.transforms.base import CompositeTransform # a wrapper to stack simpler transformations to form a more complex one
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform # the basic transformation, which we will stack several times
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform # the basic transformation, which we will stack several times
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.permutations import ReversePermutation # a layer that simply reverts the order of outputs

cuda = torch.cuda.is_available() 
if cuda:
    device = torch.device("cuda:0")       
else:
    device = torch.device("cpu")  
    
print(device)

text_debug= True#False
plot_debug = False#
table_debug = False#True#False

#SH To Avoid Index Confusion
zeta_gen_index = 0
zeta_rec_index = 1
weight_gen_index = 2
weight_rec_index = 3
pt_gen_index = 4
pt_rec_index = 5
mass_gen_index= 6
mass_jet_gen_index= 7

gen_index = [zeta_gen_index,weight_gen_index,pt_gen_index]
rec_index = [zeta_rec_index,weight_rec_index,pt_rec_index]

zeta_sample_index = 0
weight_sample_index = 1
pt_sample_index = 2


sample_index = [zeta_sample_index,weight_sample_index,pt_sample_index]

shifts = [-2,0,2]
colors = np.array(["red", "blue", "green", "#d62728", "#9467bd", "#8c564b"])
linestyles = ["-",":","--","-.","-."]

try :
    with open(args.train, "rb") as f:
        train_data_uncut = np.load(f)
        train_data = train_data_uncut[0:1000000]
        train_data = train_data[train_data[:, weight_gen_index] > w_cut] #SH: Apply cut for weight
        f.close()
        
except FileNotFoundError :
    print("File \""+ args.train+"\" (Train Data) not found.")
    exit(1)

## transform data and save max, min, mean, std values for the backtransformation later
max_values = np.max(train_data, keepdims=True, axis=0)*1.1
min_values = np.min(train_data, keepdims=True, axis=0)/1.1

    
try :
    with open(args.unfolded, "rb") as f:
        unfolded = np.load(f)
        f.close()  
except FileNotFoundError :
    print("File \""+ args.unfolded+"\" not found.")
    exit(1)
    
try :
    with open(args.val, "rb") as f:
        val = np.load(f)
        val = val[val[:, weight_gen_index] > w_cut] 
        _ , mask = trf.normalize_data(val, max_values, min_values)
        val = val[mask]
        f.close()  
except FileNotFoundError :
    print("File \""+ args.val+"\" not found.")
    exit(1)
    


print("unfolded",unfolded.shape)
print("val",val.shape)
    

plt_w = 1 # SH: plot Weight
plt_wz = 2#SH: plot weighted Zeta
plt_z = 0#

plt_d = 0 #SH: Plot Data
plt_r = 1 #SH: Plot Ration Pad

unfolding_shifts = [-1,0,1]

#SH: Plot Everything for sanitycheck
fig, axs =  plt.subplots(2, 3, sharex = "col", tight_layout=True,figsize=(15, 6), gridspec_kw=
                                dict(height_ratios=[6, 1],
                                      width_ratios=[1, 1, 1]))
                                      
#SH: Add Ticks to every side
for ax_row in axs:
    for ax in ax_row:
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        
fig.suptitle('Comparison of unfolded zeta\' ')

for i, for_shift in enumerate(unfolding_shifts):
    plt_wz = i
    
    print(plt_wz)

    #SH Normalize
    alpha = 0.8
    number_of_bins = 15

    for count, shift in enumerate(shifts):
        
        #SH: Perform the Breit Wigner Shift:
        m_old = 172.5
        m_new = m_old + shift
        Gamma = 1.3

        k_old = get_k(Gamma,m_old)
        k_new = get_k(Gamma,m_new)

        #SH: Get New Weights
        new_weights = bw_reweight(val[:,mass_gen_index],m_old,m_new,Gamma,k_old,k_new)
        
        label = "Validation m_t= " + str(m_new) + " GeV"
        color = colors[count]
        linestyle = linestyles[count]

        upper_border_3 = 7
        upper_border = upper_border_3 *100
        step = upper_border // number_of_bins
        n_bins = [x / 100.0 for x in range(0,upper_border+1,step)] 

        weights= val[:,weight_gen_index] * new_weights
        weights /= np.sum(weights)

        hist3,bin_edges = np.histogram(val[:,zeta_gen_index] , weights= weights , bins= n_bins)
        hep.histplot(hist3,n_bins, ax=axs[plt_d,plt_wz],color = color, alpha = alpha, linestyle = linestyle, label = label)

    m_old = 172.5
    m_new = m_old + for_shift
    Gamma = 1.3

    k_old = get_k(Gamma,m_old)
    k_new = get_k(Gamma,m_new)

    new_weights = bw_reweight(val[:,mass_gen_index],m_old,m_new,Gamma,k_old,k_new)

    upper_border_3 = 7
    upper_border = upper_border_3 *100
    step = upper_border // number_of_bins
    n_bins = [x / 100.0 for x in range(0,upper_border+1,step)] 

    weights= unfolded[:,weight_sample_index] * new_weights
    weights /= np.sum(weights)

    hist3_marker,bin_edges = np.histogram(unfolded[:,zeta_sample_index] , weights= weights , bins= n_bins)
    hist3_marker_error, _        = np.histogram(unfolded[:,zeta_sample_index] , weights= weights**2 , bins= n_bins)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    hist3_marker_error = np.sqrt(hist3_marker_error)
    axs[plt_d,plt_wz].errorbar(bin_centers, hist3_marker, yerr=hist3_marker_error, fmt='.',color ='black', ecolor='black', alpha=0.5, capsize=5, capthick=1, label = "Unfolded m_t= "+ str(m_new) +"GeV")

    #Ratio Pad:
    weights= val[:,weight_gen_index] * new_weights
    weights /= np.sum(weights)
    hist3_ori,bin_edges = np.histogram(val[:,zeta_gen_index] , weights= weights , bins= n_bins)
    
    ratio1 = np.divide(hist3_ori, hist3_marker, where=hist3_marker!=0)
    hep.histplot(ratio1, n_bins, ax=axs[plt_r,plt_wz],color = "black", alpha = 0.5) 
  
    ratio_error = np.divide(hist3_marker_error, hist3_marker, where=hist3_marker!=0)
    hep.histplot(ratio1, n_bins, ax=axs[plt_r,plt_wz],color = "black", alpha = 0.5) 
    axs[plt_r,plt_wz].errorbar(bin_centers, np.ones_like(ratio_error), yerr=ratio_error, fmt='.',color ='black', ecolor='black', alpha=0.5, capsize=5, capthick=1, label = "Unfolded m_t= "+ str(m_new) +"GeV")
  
    axs[plt_d,plt_wz].legend(frameon = False, fontsize="9", loc=8)
    axs[plt_d,plt_wz].legend(frameon = False, fontsize="9", loc=8)
    axs[plt_r,plt_wz].set_xlabel("$\zeta$ * pt$^2$ / 172.5$^2$")
    axs[plt_r,plt_wz].set_ylabel(" $\\frac{\\mathrm{Val "+str(m_new)+"GeV}}{\\mathrm{Unfolded}}$")
    axs[plt_d,plt_wz].set_ylabel("Events (weighted)")
    axs[plt_r,plt_wz].grid(axis = "y")
    axs[plt_r,plt_wz].set_ylim([1/1.2, 1.2])
    axs[plt_d,plt_wz].set_xlim([0, upper_border_3])

# Extract directory, filename, and extension
dir_name, base_name = os.path.split(args.unfolded)  # Splits into directory and filename

#SH: add new m_top to filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

plot_filename = "unfolded_zetaprime_compare_"+timestamp+".png"
new_plot_path = os.path.join(dir_name, plot_filename)

plt.savefig(new_plot_path)
plt.close("all")