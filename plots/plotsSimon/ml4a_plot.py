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

from MLUnfolding.Tools.user  import plot_directory

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel', action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--train',  action='store', type=str, default="NA")
argParser.add_argument('--val',    action='store', type=str, default="NA")
argParser.add_argument('--plot_dir',    action='store', type=str, default="MLUnfolding_tmp")
argParser.add_argument('--load_model_file',    action='store', type=str, default="NA")
argParser.add_argument('--save_model_path',    action='store', type=str, default="NA")
argParser.add_argument('--load_model_path',    action='store', type=str, default="NA")
argParser.add_argument('--info',    action='store', type=str, default="NA") 
argParser.add_argument('--text_debug',    action='store', type=bool, default=True) 

args = argParser.parse_args()

print("Hi")

pt_bins = np.array([400,425,450,475,500,525,550,575,600,1000])
n_pt_bins = pt_bins.shape[0]

text_debug= args.text_debug

plot_dir = os.path.join(plot_directory, args.plot_dir)# Zusammenpasten von plot_dir
if not os.path.exists( plot_dir ): os.makedirs( plot_dir )


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

gen_index = [zeta_gen_index,weight_gen_index,pt_gen_index]
rec_index = [zeta_rec_index,weight_rec_index,pt_rec_index]

zeta_sample_index = 0
weight_sample_index = 1
pt_sample_index = 2


sample_index = [zeta_sample_index,weight_sample_index,pt_sample_index]


try :
    with open(args.train, "rb") as f:
        train_data = np.load(f)
        
        train_data = train_data[0:1000000]
        
        f.close()
except FileNotFoundError :
    print("File \""+ args.train+"\" (Train Data) not found.")
    exit(1)

train_data_lenght = np.shape(train_data)[0]
train_data_n_cols = np.shape(train_data)[1]

if text_debug == True : 
    print("Lenght of training data: " + str(train_data_lenght))
    print("Cols of training data: "   + str(train_data_n_cols))
    
    #Print for
if table_debug == True : 
    print("Imported Raw Training Data")
    print(train_data)

## transform data and save max, min, mean, std values for the backtransformation later
max_values = np.max(train_data, keepdims=True, axis=0)*1.1
min_values = np.min(train_data, keepdims=True, axis=0)/1.1

transformed_data, mask = trf.normalize_data(train_data, max_values, min_values)
transformed_data = trf.logit_data(transformed_data)
mean_values = np.mean(transformed_data, keepdims=True, axis=0)
std_values = np.std(transformed_data, keepdims=True, axis=0)
transformed_data = trf.standardize_data(transformed_data, mean_values, std_values)

print("\nSampling now:")

#SH: Redundant Exception Handling. Works though
try :
    with open(args.val, "rb") as f:
        val_data = np.load(f)
        val_data = val_data[0:1000000]
        f.close()
except FileNotFoundError :
    print("File \""+ args.val+"\" not found.")
    exit(1)
 
val_transformed_data, mask = trf.normalize_data(val_data, max_values, min_values)
val_transformed_data = trf.logit_data(val_transformed_data)
val_transformed_data = trf.standardize_data(val_transformed_data, mean_values, std_values)
val_trans_cond = torch.tensor(val_transformed_data[:,rec_index], device=device).float()
val_data = val_data[mask]

print(val_trans_cond.shape)

models = os.listdir(args.load_model_path)
models.sort()
loss_function_in = []
loss_function_out=[]

#Calculate In Error
x_train = transformed_data[:,gen_index]
x_train = torch.tensor(x_train, device=device).float()
y_train = transformed_data[:,rec_index]
y_train = torch.tensor(y_train, device=device).float()#.view(-1, 1)

#Calculate Out Error 
x_val = val_transformed_data[:,gen_index]
x_val = torch.tensor(x_val, device=device).float()
y_val = val_transformed_data[:,rec_index]
y_val = torch.tensor(y_val, device=device).float()

#models = models[-2:-1]

for modelname in models:
    modelpath = args.load_model_path + "/"+ modelname

    try:
        flow =torch.load(modelpath)
        flow.eval()
    except Exception as e:
        print("Not able to load given flow " + modelpath)
        exit(0)
    print("Sampling from flow " + modelpath)


    nll_in = -flow.log_prob(x_train, context=y_train) # Feed context
    loss_in = nll_in.mean()
    loss_function_in.append(loss_in.item())
    
    nll_out = -flow.log_prob(x_val, context=y_val)
    loss_out = nll_out.mean()
    loss_function_out.append(loss_out.item())  

    with torch.no_grad():
      samples = flow.sample(1, context=val_trans_cond).view(val_trans_cond.shape[0], -1).cpu().numpy()
    ## inverse standardize
    retransformed_samples = trf.standardize_inverse(samples, mean_values[:,gen_index], std_values[:,gen_index])
    ## inverse logit
    retransformed_samples = trf.logit_inverse(retransformed_samples)
    ## inverse normalize
    retransformed_samples = trf.normalize_inverse(retransformed_samples, max_values[:,gen_index], min_values[:,gen_index]) 

    if text_debug :
        print("val data shape =",     val_trans_cond.shape)
        print("sampled data shape =", retransformed_samples.shape)

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    
    #plotweight = np.shape(val_data)[0] / np.shape(train_data)[0]
    #plotweights = np.full((np.shape(train_data)[0]),plotweight)

    modelname = modelname.replace(".pt", "")
    modelname = modelname.replace("m2f3e", "")
    modelname = modelname.zfill(2)

    number_of_bins = 50
    
    #Hier wäre eine Gute Stelle für die For-Schleife
    #Über alle pt in pb_bins iterieren
    #chechen ob in bin (achtung bei letztem bin)
    
    for i in range(n_pt_bins) :
        ptbin = pt_bins[i]
        
        #SH: Filter pt bin (pt_gen)
        if ptbin == pt_bins[n_pt_bins-1] : #SH: Funny Filter
            filter_array = val_data[:,pt_gen_index] >= ptbin
        else : 
            filter_array = (val_data[:,pt_gen_index] >= ptbin) & (val_data[:,pt_gen_index] < pt_bins[i+1])
            
        plot_samples = retransformed_samples[filter_array]
        plot_val_data= val_data[filter_array]

        fig, axs =  plt.subplots(2, 3, sharex = "col", tight_layout=True,figsize=(15, 6), gridspec_kw=
                                        dict(height_ratios=[6, 1],
                                              width_ratios=[1, 1, 1]))
        
        #SH: Add Ticks to every side
        for ax_row in axs:
            for ax in ax_row:
                ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


        if args.info == "NA" :
            fig.suptitle("Epoch: "+modelname)
        else :
            fig.suptitle(args.info + " | Epoch: "+modelname)
        number_of_bins = 20

        upper_border = 7
        upper_border = upper_border *100
        step = upper_border // number_of_bins
        n_bins = [x / 100.0 for x in range(0,upper_border+1,step)] 
        
        hist1,_ = np.histogram(plot_samples[:,zeta_sample_index], bins= n_bins)
        hist2,_ = np.histogram(plot_val_data[:,zeta_rec_index]    ,bins= n_bins)
        hist3,_ = np.histogram(plot_val_data[:,zeta_gen_index] ,bins= n_bins)
        hist4 = np.divide(hist1, hist3, where=hist3!=0)
        
        hep.histplot(hist1,       n_bins, ax=axs[0,0],color = "red",alpha = 0.5,      label = "Val Particle Lvl", histtype="fill")
        hep.histplot(hist2,       n_bins, ax=axs[0,0],color = "black",   label = "Val Detector Lvl") 
        hep.histplot(hist3,       n_bins, ax=axs[0,0],color = "#999999", label = "Particle Lvl"    )
        hep.histplot(hist4, n_bins, ax=axs[1,0],color = "red", alpha = 0.5)   
        
        upper_border = 300
        step = upper_border // number_of_bins
        n_bins = [x / 10000000.0 for x in range(0,upper_border+1,step)]
        
        hist1,_ = np.histogram(plot_samples[:,weight_sample_index], bins= n_bins)
        hist2,_ = np.histogram(plot_val_data[:,weight_rec_index]    ,bins= n_bins)
        hist3,_ = np.histogram(plot_val_data[:,weight_gen_index] , bins= n_bins)
        hist4 = np.divide(hist1, hist3, where=hist3!=0)
        
        hep.histplot(hist1,       n_bins, ax=axs[0,1],color = "red",alpha = 0.5,      label = "Val Particle Lvl", histtype="fill")
        hep.histplot(hist2,       n_bins, ax=axs[0,1],color = "black",   label = "Val Detector Lvl") 
        hep.histplot(hist3,       n_bins, ax=axs[0,1],color = "#999999", label = "Particle Lvl"    )
        hep.histplot(hist4, n_bins, ax=axs[1,1],color = "red", alpha = 0.5)
        
        upper_border = 7
        upper_border = upper_border *100

        step = upper_border // number_of_bins
        n_bins = [x / 100.0 for x in range(0,upper_border+1,step)] 

        hist1,_ = np.histogram(plot_samples[:,zeta_sample_index], weights= plot_samples[:,weight_sample_index], bins= n_bins)
        hist2,_ = np.histogram(plot_val_data[:,zeta_rec_index],    weights= plot_val_data[:,weight_rec_index]             , bins= n_bins)
        hist3,_ = np.histogram(plot_val_data[:,zeta_gen_index],    weights= plot_val_data[:,weight_gen_index]           , bins= n_bins)
        hist4 = np.divide(hist1, hist3, where=hist3!=0)
        
        hep.histplot(hist1,       n_bins, ax=axs[0,2],color = "red",alpha = 0.5,      label = "Val Particle Lvl", histtype="fill")
        hep.histplot(hist2,       n_bins, ax=axs[0,2],color = "black",   label = "Val Detector Lvl") 
        hep.histplot(hist3,       n_bins, ax=axs[0,2],color = "#999999", label = "Particle Lvl"    )
        hep.histplot(hist4, n_bins, ax=axs[1,2],color = "red", alpha = 0.5)
        
        axs[0,0].set_yscale("log")
        axs[0,1].set_yscale("log")
        #axs[0,2].set_yscale("log")
        
        axs[0,0].legend(frameon = False, fontsize="18")
        axs[0,1].legend(frameon = False, fontsize="18")
        axs[0,2].legend(frameon = False, fontsize="14", loc=8)

        axs[0,0].set_xlabel("$\zeta$ * pt$^2$ / 172.5$^2$")
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
        
        axs[1,0].set_ylim([0.7, +1.3])
        axs[1,1].set_ylim([0.7, +1.3])
        axs[1,2].set_ylim([0.7, +1.3])
        
        axs[0,0].set_ylim([1, 1e5])
        axs[0,1].set_ylim([1, 1e5])
        axs[0,0].set_xlim([0, 7])
        axs[0,1].set_xlim([0, 0.00003])
        axs[0,2].set_xlim([0, 7])

        
        for_plot_dir = os.path.join(plot_dir, "pt"+str(ptbin))# Zusammenpasten von plot_dir 
        if not os.path.exists( for_plot_dir ): os.makedirs( for_plot_dir )
        
        plt.savefig(for_plot_dir+"/generated_data_"+modelname+".png")
        plt.close("all")
    
    
it=[*range(len(loss_function_in))]

plt.plot(it,loss_function_in, label="Train-Loss", color="#696969") 
plt.plot(it,loss_function_out, label="Validation-Loss", color="red", alpha=0.5)
plt.xlabel("Epochs")
plt.ylabel("-log loss")
plt.legend()
plt.savefig(plot_dir+"/loss"+current_time+".png")
plt.close("all")