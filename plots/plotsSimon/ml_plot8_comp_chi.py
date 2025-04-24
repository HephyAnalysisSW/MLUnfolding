print("Start of Script")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from datetime import datetime
import os
import plotutils
import psutil
import gc
import re

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
    
def extract_eeec_number(path):
    match = re.search(r'EEEC_(\d+)', path)
    if match:
        return int(match.group(1))
    else:
        return None


import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel', action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--tomass', action='store', type=float, default=173.5)
argParser.add_argument('--c1',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--c2',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--c3',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--c4',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--plot_dir',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy


#/groups/hephy/cms/simon.hablas/www/MLUnfolding/plots/EEEC_79_old_data/weight_cut_1p5e-05/stack_with_sample_cut/data

args = argParser.parse_args()

filename = f"chi2_to{args.tomass}.npz"

filenames = []
legends = []

colors = ["red", "blue", "green", "black"]

for c in [args.c1, args.c2, args.c3, args.c4]:
    if c != "NA":
        filenames.append(os.path.join(c, filename))
        legends.append(extract_eeec_number(c))



plt.figure(figsize=(8, 5))
   
for i, f in enumerate(filenames):
    data = np.load(f)

    masses = data['masses']
    chi_list = data['chi_list']
    m_fit = data['m_fit']
    chi_fit = data['chi_fit']
    truth_value = data['truth_value'][0]
    
        # Plotting

    plt.plot(masses, chi_list,color = colors[i], marker='o', linestyle='-.',label=str(legends[i]) + " data")
    #plt.plot(m_fit, chi_fit, '-', color =colors[i], label=str(legends[i]) + " fit")

plt.axvline(x=truth_value, color="grey", linestyle='-', label='truth')
plt.xlabel("Mass Hypothesis [GeV]")
plt.ylabel("Chi-squared")
plt.title("Chi-squared vs Mass Hypothesis")
plt.grid(True)   
    

collapsed = "_".join(str(n) for n in legends)
plot_path = os.path.join(args.plot_dir, f"compare_{collapsed}.png")
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
print(plot_path)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()  # Adjust layout to fit legend

plt.savefig(plot_path, bbox_inches='tight')  # 'tight' ensures everything fits
plt.close("all")