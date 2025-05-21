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
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress


#from MLUnfolding.Tools.user  import plot_directory
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

shifts = [-1,0,1]
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
    
Gamma = 1.3
m_old = 172.5
m_new = m_old + args.shift
k_old = get_k(Gamma, m_old)
k_new = get_k(Gamma, m_new)

new_weights = bw_reweight(val[:, mass_gen_index], m_old, m_new, Gamma, k_old, k_new)


val_gen_weights = val[:, weight_gen_index] * new_weights
val_gen_weights /= np.sum(val_gen_weights)

val_rec_weights = val[:, weight_rec_index] * new_weights
val_rec_weights /= np.sum(val_rec_weights)

unfold_weights = val[:, weight_gen_index] * new_weights
unfold_weights /= np.sum(unfold_weights)

#train_weights /= np.sum(train_weights)


#______________________________________________________________________________________________________________
#--SH: Plot Zeta`
number_of_bins = 20
upper_border = 7
upper_border = upper_border *100
step = upper_border // number_of_bins
n_bins = [x / 100.0 for x in range(0,upper_border+1,step)] 

hist1,bin_edges = np.histogram(unfolded[:,zeta_sample_index], bins= n_bins)
hist2,_ = np.histogram(val[:,zeta_rec_index]    ,bins= n_bins)
hist3,_ = np.histogram(val[:,zeta_gen_index] ,bins= n_bins)
hist4 = np.divide(hist1, hist3, where=hist3!=0)

#SH Zetas
z_bins = n_bins
z_part_unfolded = hist1
z_det = hist2
z_part_truth = hist3


#_________________________________________________________________________________________________________________
#--SH: Plot Weight
upper_border = 1000 #300
step = upper_border // number_of_bins
n_bins = [x / 10000000.0 for x in range(0,upper_border+1,step)]


#SH: To choose a single bin
target_bin_index = 10
bin_min = bin_edges[target_bin_index]
bin_max = bin_edges[target_bin_index + 1]
    
hist1,_ = np.histogram(unfold_weights, bins= n_bins)
hist2,_ = np.histogram(val_rec_weights    ,bins= n_bins)
hist3,_ = np.histogram(val_gen_weights , bins= n_bins)
hist4 = np.divide(hist1, hist3, where=hist3!=0)

#SH Weights
w_bins = n_bins
w_part_unfolded = hist1
w_det = hist2
w_part_truth = hist3

#_________________________________________________________________________________________________________________
#--SH: Plot Weighted Zeta'
upper_border = 7
upper_border = upper_border *100
step = upper_border // number_of_bins
n_bins = [x / 100.0 for x in range(0,upper_border+1,step)] 

hist1, bin_edges = np.histogram(unfolded[:,zeta_sample_index] , weights= unfold_weights , bins= n_bins)
hist2,_ = np.histogram(val[:,zeta_rec_index]  , weights= val_rec_weights  , bins= n_bins)
hist3,_ = np.histogram(val[:,zeta_gen_index]  , weights= val_gen_weights   , bins= n_bins)
hist4 = np.divide(hist1, hist3, where=hist3!=0)
#hist5,_ = np.histogram(train_data[:,zeta_gen_index] , weights= train_weights   , bins= n_bins)

hist1_error, _ = np.histogram(unfolded[:,zeta_sample_index] , weights= unfold_weights**2 , bins= n_bins) 
hist1_error = np.sqrt(hist1_error)

hist2_error, _ = np.histogram(val[:,zeta_rec_index]  , weights= val_rec_weights**2   , bins= n_bins)
hist2_error = np.sqrt(hist2_error)

hist3_error, _ = np.histogram(val[:,zeta_gen_index]  , weights= val_gen_weights**2   , bins= n_bins)
hist3_error = np.sqrt(hist3_error)

#SH: weighted zeta
wz_bins = n_bins
wz_part_unfolded = hist1
wz_det = hist2
wz_part_truth = hist3
wz_part_unfolded_err = hist1_error
wz_det_err = hist2_error
wz_part_truth_err = hist3_error

weight_squared = calc_squared_weights(unfolded[:,zeta_sample_index] , weights= unfolded[:,weight_sample_index] , bins= n_bins)
chi2 = calculate_chisq(hist1,hist3,weight_squared)


#SH:---------------Save -------------------
dir_name, base_name = os.path.split(args.unfolded)
plot_filename = f"histograms_{str(m_new).replace('.','_')}.npz"
new_plot_path = os.path.join(dir_name, plot_filename)


#SH gespeichtern mit: 
np.savez(
    new_plot_path,
    z_bins = z_bins,
    z_part_unfolded=z_part_unfolded,
    z_det=z_det,
    z_part_truth=z_part_truth,
    w_bins = w_bins,
    w_part_unfolded=w_part_unfolded,
    w_det=w_det,
    w_part_truth=w_part_truth,
    wz_bins = wz_bins,
    wz_part_unfolded=wz_part_unfolded,
    wz_det=wz_det,
    wz_part_truth=wz_part_truth,
    wz_part_unfolded_err=wz_part_unfolded_err,
    wz_det_err=wz_det_err,
    wz_part_truth_err=wz_part_truth_err
)

print("saved under: ", new_plot_path)
