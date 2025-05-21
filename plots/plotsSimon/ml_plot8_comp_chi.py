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

def extract_eeec_number(path):
    match = re.search(r'EEEC_(\d+)', path)
    if match:
        return int(match.group(1))
    else:
        return None

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel', action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument(
    '--tomass',
    nargs='+',               # Accepts one or more values
    type=float,
    default=[173.5],
    help="One or many truth value comparison mass."
)
argParser.add_argument('--f1',    action='store', type=str, default="NA", help = "1st input numpy file path") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--f2',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--f3',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--f4',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--plot_dir',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy


#/groups/hephy/cms/simon.hablas/www/MLUnfolding/plots/EEEC_79_old_data/weight_cut_1p5e-05/stack_with_sample_cut/data

args = argParser.parse_args()

for tomass in args.tomass:

    filename = f"chi2_to{tomass}.npz"

    filenames = []
    legends = []

    colors = ["red", "blue", "green", "black"]

    for c in [args.f1, args.f2, args.f3, args.f4]:
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
    #plt.xlabel("Test Mass [GeV]")
    #plt.ylabel("Chi-squared")
    plt.grid(True)

    # Add custom x-label explanation at the right end of the x-axis
    plt.text(
        1.02, -0.1,
        r"Test mass [GeV]",
        transform=plt.gca().transAxes,
        fontsize=16,
        va="top", ha="right"
    )

    # Add custom y-label explanation at the top end of the y-axis
    plt.text(
        -0.1, 1.02,
        r"$\chi^2$",
        transform=plt.gca().transAxes,
        fontsize=16,
        va="top", ha="right"
    )


    collapsed = "_".join(str(n) for n in legends)
    plot_path = os.path.join(args.plot_dir, f"compare_{collapsed}_{tomass}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    print(plot_path)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)
    plt.tight_layout()


    plt.savefig(plot_path, bbox_inches='tight')  # 'tight' ensures everything fits
    plt.close("all")