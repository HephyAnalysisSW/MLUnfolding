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


from MLUnfolding.Tools.user  import plot_directory
#plot_directory = "./plots"

from ml_functions import (calc_squared_weights,calculate_chisq,count_parameters,get_k,bw_reweight)
from indices import (
    zeta_gen_index,
    zeta_rec_index,
    weight_gen_index,
    weight_rec_index,
    pt_gen_index,
    pt_rec_index,
    mass_gen_index,
    mass_jet_gen_index,
    weight_sample_index,
    zeta_sample_index,
    pt_sample_index
)


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
    


print("unfolded",unfolded.shape)
print("val",val.shape)
    

# Parameters
plt_d = 0  # Data plot row
plt_r = 1  # Ratio plot row

# Normalization and bin setup
alpha = 0.8
number_of_bins = 15
upper_border_3 = 7
upper_border = upper_border_3 * 100
step = upper_border // number_of_bins
n_bins = [x / 100.0 for x in range(0, upper_border + 1, step)]

shifts = [-2,-1,0,1,2]
for shift in shifts:
    
    # Set up single plot pair (2 rows, 1 column)
    fig, axs = plt.subplots(2, 1, sharex="col", tight_layout=True,
                        figsize=(5, 6), gridspec_kw=dict(height_ratios=[6, 1]))

    # Add ticks to both sides
    for ax in axs:
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    
    #SH:---------------Reweighting -------------------
    Gamma = 1.3
    m_old = 172.5
    m_new = m_old + shift
    k_old = get_k(Gamma, m_old)
    k_new = get_k(Gamma, m_new)
    
    new_weights = bw_reweight(val[:, mass_gen_index], m_old, m_new, Gamma, k_old, k_new)
    
    print("")
    print("Start m_t=",m_new)
    fig.suptitle("Unfolding $m_t$="+str(m_new)+" GeV", fontsize=14)
    
    #SH:---------------Histogramms -------------------
    
    #SH: calc val truth lvl histogram
    weights = val[:, weight_gen_index]
    weights /= np.sum(weights)
    hist_val_truth, bin_edges = np.histogram(val[:, zeta_gen_index], weights=weights, bins=n_bins)

    #SH: calc shifted val truth lvl histogram
    weights = val[:, weight_gen_index] * new_weights
    weights /= np.sum(weights)
    hist_shifted_val_truth, bin_edges = np.histogram(val[:, zeta_gen_index], weights=weights, bins=n_bins)

    #SH: calc shifted unfolded lvl markers with errorbars
    weights = unfolded[:, weight_sample_index] * new_weights
    weights /= np.sum(weights)
    hist_marker, bin_edges = np.histogram(unfolded[:, zeta_sample_index], weights=weights, bins=n_bins)
    hist_marker_error, _   = np.histogram(unfolded[:, zeta_sample_index], weights=weights**2, bins=n_bins)
    hist_marker_error = np.sqrt(hist_marker_error)

    #SH:---------------Splines -------------------
    bin_centers =(bin_edges[:-1] + bin_edges[1:]) / 2

    # Select bins 4 to 16
    fit_start, fit_end = 4, 12
    x_fit = bin_centers[fit_start:fit_end]
    x_smooth = np.linspace(x_fit[0], x_fit[-1], 300)

    #SH: calc val splines
    y_fit_val = hist_val_truth[fit_start:fit_end]
    spline_val = UnivariateSpline(x_fit, y_fit_val, s=0)
    y_smooth_val = spline_val(x_smooth)

    #SH: calc marker splines
    y_fit_marker = hist_marker[fit_start:fit_end]
    spline_marker = UnivariateSpline(x_fit, y_fit_marker, s=0)
    y_smooth_marker = spline_marker(x_smooth)
    
    #SH:---------------Plots-------------------

    #SH: pred Splines
    axs[plt_d].plot(x_smooth, y_smooth_marker, color='#ff0000', lw=1) #, label="prediction Spline")

    #SH: pred Markers
    axs[plt_d].errorbar(bin_centers, hist_marker, yerr=hist_marker_error, fmt='.',
                        color='#ff0000', ecolor='#ff0000', alpha=0.5, capsize=5, capthick=1,
                        label=f"prediction (${m_new}$ GeV)") #label=f"prediction $m_t$= {m_new} GeV")

    #SH: truth hist
    hep.histplot(hist_val_truth, n_bins, ax=axs[plt_d], color="black", alpha=0.8, label="truth ($172.5$ GeV)")
    
    #SH: shifted truth hist
    if shift != 0:
        hep.histplot(hist_shifted_val_truth, n_bins, ax=axs[plt_d], color="#0000ff",ls = "--", alpha=0.5, label=f"truth (${m_new}$ GeV)")

    #SH: truth splines
    axs[plt_d].plot(x_smooth, y_smooth_val,color='black', alpha=0.8, lw=1)# label="truth Spline",

    #SH:---------------Ratio Pad -------------------
    ratio1 = np.divide(hist_shifted_val_truth, hist_marker, where=hist_marker != 0)
    hep.histplot(ratio1, n_bins, ax=axs[plt_r], color="black", alpha=0.5)

    ratio_error = np.divide(hist_marker_error, hist_marker, where=hist_marker != 0)
    axs[plt_r].errorbar(bin_centers, np.ones_like(ratio_error), yerr=ratio_error, fmt='.',
                        color='#ff0000', ecolor='#ff0000', alpha=0.5, capsize=5, capthick=1,
                        label=f"Unfolded m_t= {m_new} GeV")
                        
    #SH: Truth Lvl / Prediction Lvl Spline in Ratio Pad
    if False:
        axs[plt_r].plot(x_smooth, y_smooth_val/y_smooth_marker,color='black', alpha=0.8, lw=1)# label="truth Spline",
    
    #SH: Einzeichnen der Ausgleichsgerade im Ratio
    ratio2 = np.divide(hist_val_truth, hist_marker, where=hist_marker != 0)
    hep.histplot(ratio2, n_bins, ax=axs[plt_r], color="#0000ff", alpha=0.2)
    slope, intercept, r_value, p_value, std_err = linregress(bin_centers[fit_start:fit_end], ratio2[fit_start:fit_end])
    axs[plt_r].plot(bin_centers[fit_start:fit_end], slope * bin_centers[fit_start:fit_end] + intercept, color='#0000ff')

    print("Slope="+str(slope))
   

    #SH:---------------Legends, Text, ect. -------------------
    axs[plt_d].legend(frameon=False, fontsize="8", loc=2)
    axs[plt_r].set_xlabel(r"$\zeta * \frac{p_t^2}{172.5^2}$", fontsize=16)
    axs[plt_r].set_ylabel(r"$\frac{\mathrm{truth}}{\mathrm{prediction}}$", fontsize=16)
    axs[plt_d].set_ylabel("Events (weighted)", fontsize=16)
    axs[plt_r].grid(axis="y")
    axs[plt_r].set_ylim([1 / 1.2, 1.2])
    axs[plt_d].set_xlim([0, upper_border_3])
    
    axs[plt_r].set_xlabel("")
    axs[plt_d].set_ylabel("")

    # Add custom "x-label" to the right end of the x-axis
    axs[plt_r].text(
        1., -0.2,  # x and y in axis coords (1.01 slightly outside right edge)
        r"$\zeta * \frac{p_t^2}{172.5^2}$",
        transform=axs[plt_r].transAxes,
        fontsize=16,
        va="top", ha="right"
    )
    
    # Likewise for lower subplot (just y-label on top end)
    axs[plt_d].set_ylabel("")
    axs[plt_d].text(
        -0.1, 1.,
        r"Events (weighted)",
        transform=axs[plt_d].transAxes,
        fontsize=16,
        va="top", ha="right",
        rotation=90
    )
    
    axs[plt_d].legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        frameon=False
    )

    #SH:---------------Save -------------------
    dir_name, base_name = os.path.split(args.unfolded)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"test_unfld_{m_new}.png"
    new_plot_path = os.path.join(dir_name, plot_filename)
    
    print("saved ", new_plot_path)
    
    plt.savefig(new_plot_path)
    plt.close("all")
    
#Reweight val data to prefered mass

fit_start, fit_end = 4, 12
pref_shifts = [-2,-1,0,1,2]

for pref_shift in pref_shifts:

    Gamma = 1.3
    m_old = 172.5
    m_new = m_old + pref_shift
    m_new_truth = m_new
    k_old = get_k(Gamma, m_old)
    k_new = get_k(Gamma, m_new)

    new_weights = bw_reweight(val[:, mass_gen_index], m_old, m_new, Gamma, k_old, k_new)

    weights = val[:, weight_gen_index] * new_weights
    weights /= np.sum(weights)
    truth , bin_edges = np.histogram(val[:, zeta_gen_index], weights=weights, bins=n_bins)
    truthsq , bin_edges = np.histogram(val[:, zeta_gen_index], weights=weights**2, bins=n_bins)



    chi_list = []
    shifts = list(np.arange(-2.5, 2.5 + 0.5, 0.5))

    for shift in shifts:
        Gamma = 1.3
        m_old = 172.5
        m_new = m_old + shift
        k_old = get_k(Gamma, m_old)
        k_new = get_k(Gamma, m_new)

        new_weights = bw_reweight(val[:, mass_gen_index], m_old, m_new, Gamma, k_old, k_new)

        weights = unfolded[:, weight_sample_index] * new_weights
        weights /= np.sum(weights)
        prediction , bin_edges = np.histogram(unfolded[:, zeta_sample_index], weights=weights, bins=n_bins)

        chi2 = calculate_chisq(truth[fit_start:fit_end], prediction[fit_start:fit_end], truthsq[fit_start:fit_end])


        chi_list.append(chi2)

    masses = [shift + 172.5 for shift in shifts]
    chi_list = np.array(chi_list)

    coeffs = np.polyfit(masses, chi_list, deg=2)

    m_fit = np.linspace(min(masses), max(masses), 100)
    chi_fit = np.polyval(coeffs, m_fit)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(masses, chi_list,color = "red", marker='o', linestyle='-.',label='$\chi^2$')
    plt.plot(m_fit, chi_fit, '-', color ="black", label='Quadratic Fit')
    plt.axvline(x=172.5+pref_shift, color='#0000ff', linestyle='-', label='truth')


    plt.xlabel("Mass Hypothesis [GeV]")
    plt.ylabel("Chi-squared")
    plt.title("Chi-squared vs Mass Hypothesis")
    plt.grid(True)

    dir_name, base_name = os.path.split(args.unfolded)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"chi2_to{m_new_truth}.png"
    data_filename = f"chi2_to{m_new_truth}.npz"
    new_plot_path = os.path.join(dir_name, plot_filename)
    new_data_path = os.path.join(dir_name, data_filename)
    
    plt.legend()
    
    # Save the plot
    
    truth_value = 172.5 + pref_shift

    np.savez(new_data_path,
         masses=masses,
         chi_list=chi_list,
         m_fit=m_fit,
         chi_fit=chi_fit,
         truth_value=np.array([truth_value]))
    
    print("saved ", new_plot_path)
    
    plt.savefig(new_plot_path)
    plt.close("all")