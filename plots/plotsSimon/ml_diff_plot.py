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
from torchdiffeq import odeint
from torch.utils.data import Dataset, DataLoader


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


import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel', action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--train',  action='store', type=str, default="NA") # ./mldata/ML_Data_train.npy
argParser.add_argument('--val',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--plot_dir',    action='store', type=str, default="MLUnfolding_tmp") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--load_model_file',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--save_model_path',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--load_model_path',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--info',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--weight_cut', action='store', type=float, default=0.0) # ./mldata/ML_Data_validate.npy
argParser.add_argument('--text_debug',    action='store', type=bool, default=True) #./mldata/ML_Data_validate.npy

args = argParser.parse_args()

gc.collect() #SH Test Garbage Collection
print(psutil.Process().memory_info().rss / (1024 * 1024))
print("Hi")


class TransformedDataset(Dataset):
    def __init__(self, transformed_data, gen_index, rec_index, device):
        self.x_data = torch.tensor(transformed_data[:, gen_index], dtype=torch.float32, device=device)
        self.y_data = torch.tensor(transformed_data[:, rec_index], dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

class CFM(nn.Module):
    def __init__(
        self,
        data_dim: int,     # number of features in the data
        hidden_dim: int,   # number of hidden layer nodes

    ):
        super().__init__()
        self.data_dim = data_dim

        # TODO: Build network to predict the velocity field with
        # 3 hidden layers with hidden_dim
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),nn.ReLU(),
            nn.Linear(hidden_dim, (data_dim)//2),
            )

    def batch_loss(
        self,
        x: torch.Tensor, # input data, shape (n_batch, data_dim/2)
        y: torch.Tensor, # conditional data, shape (n_batch, condition_dim)
    ) -> torch.Tensor:   # loss, shape (n_batch, )

        # TODO: Implement the batch_loss
        
        t = torch.rand(size=(x.shape[0], 1))
        noise = torch.randn_like(x)
        xt = (1-t)*x + t*noise
        model_pred = self.net(torch.cat((t.float(),xt.float(),y.float()), dim=1))

        v = noise - x

        return ((model_pred-v)**2).mean()

    def sample(
        self,
        y: torch.Tensor,  # Conditional data, shape (n_batch, condition_dim)
    ) -> torch.Tensor:   # Sampled data, shape (n_samples, data_dim)

        dtype = torch.float32
        n_samples = y.shape[0]  # Ensure batch size matches
        x_1 = torch.randn(n_samples, self.data_dim//2, device=device, dtype=dtype)

        # Define net_wrapper inside sample, capturing `y` (which is already batch-aligned)
        def net_wrapper(t, x_t):
            #print("Net Wrapper", x_t.shape, y.shape)
            t = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)  # Expand time dimension
            nn_input = torch.cat([t, x_t, y], dim=1)  # Concatenate condition `y`
            nn_out = self.net(nn_input)  # Pass through neural network
            return nn_out

        # Solve ODE with conditional dynamics
        x_t = odeint(
            net_wrapper,
            x_1,
            torch.tensor([1., 0.], dtype=dtype, device=device)
        )

        return x_t[-1]  # Return final sample at t=0


w_cut = args.weight_cut
text_debug= args.text_debug

plot_dir = os.path.join(plot_directory, args.plot_dir,"weight_cut_" + str(w_cut).replace('.', 'p'),"stack_with_sample_cut")# Zusammenpasten von plot_dir
if not os.path.exists( plot_dir ): os.makedirs( plot_dir )
plot_dir_data = os.path.join(plot_dir,"data")
if not os.path.exists( plot_dir_data ): os.makedirs( plot_dir_data )

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


gen_index = [zeta_gen_index,weight_gen_index,pt_gen_index]
rec_index = [zeta_rec_index,weight_rec_index,pt_rec_index]

zeta_sample_index = 0
weight_sample_index = 1
pt_sample_index = 2


sample_index = [zeta_sample_index,weight_sample_index,pt_sample_index]

print("Started")
try :
    with open(args.train, "rb") as f:
        train_data_uncut = np.load(f)
        train_data = train_data_uncut[0:1000000]
        train_data = train_data[train_data[:, weight_gen_index] > w_cut] #SH: Apply cut for weight
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
        val_data_uncut = np.load(f)
        val_data_plot = val_data_uncut[0:1000000]
        val_data = val_data_plot[val_data_plot[:, weight_gen_index] > w_cut] #SH: Apply cut for weight # val_data_plot#

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

# Get only .pt files
models = [file for file in os.listdir(args.load_model_path) 
          if os.path.isfile(os.path.join(args.load_model_path, file)) and file.endswith('.pt')]

# Sort alphanumerically, handling natural numbers correctly
models.sort()

print(type(models))

loss_function_in = []
loss_function_out=[]

indices_10th = np.arange(9, len(models), 10)
indices_last10 = np.arange(len(models) - 10, len(models))
unique_indices = np.unique(np.concatenate((indices_10th, indices_last10)))


#models = models[-10:] # only use last 10 models
models = models[-5:] # only use last model
#models = [models[i] for i in unique_indices]

print(models)

for modelname in models:
    modelpath = args.load_model_path + "/"+ modelname

    try:
        cfm =torch.load(modelpath)
        cfm.eval()
    except Exception as e:
        print("Not able to load given model " + modelpath)
        exit(0)
    
    print("Sampling from model " + modelpath)
    print(str(psutil.Process().memory_info().rss / (1024 * 1024)) + "MB")

    gc.collect() #SH Test Garbage Collection


    with torch.no_grad():
       #print(type(val_trans_cond), val_trans_cond.shape)
       samples = cfm.sample(y=val_trans_cond).view(val_trans_cond.shape[0], -1).cpu().numpy()
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
    
    plotweight = np.shape(val_data)[0] / np.shape(train_data)[0]
    plotweights = np.full((np.shape(train_data)[0]),plotweight)

    modelname = modelname.replace(".pt", "")
    modelname = modelname.replace("m2f3e", "")
    modelname = modelname.zfill(2)
    
    #_________________________________________________________________________________________________________________
    #--SH: Start Plot in Mathplotlib
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
        
    #_________________________________________________________________________________________________________________
    #--SH: Plot Zeta`
    number_of_bins = 20
    upper_border = 7
    upper_border = upper_border *100
    step = upper_border // number_of_bins
    n_bins = [x / 100.0 for x in range(0,upper_border+1,step)] 
    
    hist1,bin_edges = np.histogram(retransformed_samples[:,zeta_sample_index], bins= n_bins)
    hist2,_ = np.histogram(val_data[:,zeta_rec_index]    ,bins= n_bins)
    hist3,_ = np.histogram(val_data[:,zeta_gen_index] ,bins= n_bins)
    hist4 = np.divide(hist1, hist3, where=hist3!=0)
    
    hep.histplot(hist1,       n_bins, ax=axs[0,0],color = "red",alpha = 0.5,      label = "ML Val Particle Lvl", histtype="fill")
    hep.histplot(hist2,       n_bins, ax=axs[0,0],color = "black",   label = "Val Detector Lvl") 
    hep.histplot(hist3,       n_bins, ax=axs[0,0],color = "#999999", label = "Particle Lvl"    )
    hep.histplot(hist4, n_bins, ax=axs[1,0],color = "red", alpha = 0.5)   
    
    
    #_________________________________________________________________________________________________________________
    #--SH: Plot Weight
    upper_border = 1000 #300
    step = upper_border // number_of_bins
    n_bins = [x / 10000000.0 for x in range(0,upper_border+1,step)]
    
    
    #SH: To choose a single bin
    target_bin_index = 10
    bin_min = bin_edges[target_bin_index]
    bin_max = bin_edges[target_bin_index + 1]
    
    selected_samples = retransformed_samples[
    (retransformed_samples[:, zeta_sample_index] >= bin_min) &
    (retransformed_samples[:, zeta_sample_index] < bin_max)]   
    selected_samples = retransformed_samples 
    
    hist1,_ = np.histogram(retransformed_samples[:,weight_sample_index], bins= n_bins)
    hist2,_ = np.histogram(val_data[:,weight_rec_index]    ,bins= n_bins)
    hist3,_ = np.histogram(val_data[:,weight_gen_index] , bins= n_bins)
    hist4 = np.divide(hist1, hist3, where=hist3!=0)
    
    hep.histplot(hist1,       n_bins, ax=axs[0,1],color = "red",alpha = 0.5,      label = "ML Val Particle Lvl", histtype="fill")
    hep.histplot(hist2,       n_bins, ax=axs[0,1],color = "black",   label = "Val Detector Lvl") 
    hep.histplot(hist3,       n_bins, ax=axs[0,1],color = "#999999", label = "Particle Lvl"    )
    hep.histplot(hist4, n_bins, ax=axs[1,1],color = "red", alpha = 0.5)
    
    
    #_________________________________________________________________________________________________________________
    #--SH: Plot Weighted Zeta'
    upper_border = 7
    upper_border = upper_border *100
    step = upper_border // number_of_bins
    n_bins = [x / 100.0 for x in range(0,upper_border+1,step)] 
    
    total_events_hist3 = np.sum(val_data[:, weight_gen_index])
    total_events_hist5 = np.sum(train_data[:, weight_gen_index])
    total_events_hist6 = np.sum(val_data_plot[:, weight_gen_index])
    scaling_factor = total_events_hist3 / total_events_hist5
    scaling_factor_val_plot = total_events_hist3 / total_events_hist6
    
    hist1, bin_edges = np.histogram(selected_samples[:,zeta_sample_index] , weights= selected_samples[:,weight_sample_index] , bins= n_bins)
    hist2,_ = np.histogram(val_data[:,zeta_rec_index]  , weights= val_data[:,weight_rec_index]   , bins= n_bins)
    hist3,_ = np.histogram(val_data[:,zeta_gen_index]  , weights= val_data[:,weight_gen_index]   , bins= n_bins)
    hist4 = np.divide(hist1, hist3, where=hist3!=0)
    hist5,_ = np.histogram(train_data[:,zeta_gen_index] , weights= train_data[:,weight_gen_index] * scaling_factor  , bins= n_bins)
    hist6,_ = np.histogram(val_data_plot[:,zeta_gen_index]  , weights= val_data_plot[:,weight_gen_index] * scaling_factor_val_plot   , bins= n_bins)
    
    hist1_error, _ = np.histogram(selected_samples[:,zeta_sample_index] , weights= selected_samples[:,weight_sample_index]**2 , bins= n_bins) 
    hist1_error = np.sqrt(hist1_error)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    
    hep.histplot(hist1,       n_bins, ax=axs[0,2],color = "red",alpha = 0.5,      label = "ML Val Particle Lvl", histtype="fill")
    hep.histplot(hist2,       n_bins, ax=axs[0,2],color = "black",   label = "Val Detector Lvl") 
    hep.histplot(hist3,       n_bins, ax=axs[0,2],color = "#999999", label = "Particle Lvl"    ) # grau
    hep.histplot(hist5,       n_bins, ax=axs[0,2],color = "#0352fc", label = "Train Particle Lvl"    ) # Blue comparison Histogramm
    #hep.histplot(hist6,       n_bins, ax=axs[0,2],color = "green",alpha = 0.7, label = "Val Particle Lvl wo Filter") # Green comparison Histogramm
    hep.histplot(hist4, n_bins, ax=axs[1,2],color = "red", alpha = 0.5)
    # Add error bars
    axs[0,2].errorbar(bin_centers, hist1, yerr=hist1_error, fmt='none', ecolor='red', alpha=0.5, capsize=5, capthick=1)

    

    weight_squared = calc_squared_weights(selected_samples[:,zeta_sample_index] , weights= selected_samples[:,weight_sample_index] , bins= n_bins)
    chi2 = calculate_chisq(hist1,hist3,weight_squared)
    
    print("Chi^2: " + str(round(chi2,3)))
    
    #_________________________________________________________________________________________________________________
    #--SH: Plot Style and Axis
    axs[0,0].set_yscale("log")
    axs[0,1].set_yscale("log")
    #axs[0,2].set_yscale("log")
    
    axs[0,0].legend(frameon = False, fontsize="18")
    axs[0,1].legend(frameon = False, fontsize="18")
    axs[0,2].legend(frameon = False, fontsize="14", loc=8)

    axs[1,0].set_xlabel("$\zeta$ * pt$^2$ / 172.5$^2$")
    axs[1,1].set_xlabel("weight ($ \\prod \\frac{p_i}{p_t}$)")
    axs[1,2].set_xlabel("$\zeta$ * pt$^2$ / 172.5$^2$")
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
    axs[0,1].set_ylim([1e3, 1e5])
    axs[0,0].set_xlim([0, 7])
    axs[0,1].set_xlim([0, 0.0001])
    #axs[0,1].set_xlim([0, 0.00003])
    axs[0,2].set_xlim([0, 7])
  
    plt.savefig(plot_dir+"/generated_data_"+modelname+".png")
    plt.close("all")
    
    with open(plot_dir_data+"/"+modelname+".npy", 'wb') as f0:
        np.save(f0, retransformed_samples )

with open(plot_dir_data+"/train.npy", 'wb') as f0:
    np.save(f0, train_data_uncut )
with open(plot_dir_data+"/val.npy", 'wb') as f1:
    np.save(f1, val_data_uncut )
    
# it=[*range(len(loss_function_in))]

# plt.plot(it,loss_function_in, label="Train-Loss", color="#696969") 
# plt.plot(it,loss_function_out, label="Validation-Loss", color="red", alpha=0.5)
# plt.xlabel("Epochs")
# plt.ylabel("-log loss")
# plt.legend()
# plt.savefig(plot_dir_data+"/loss"+current_time+".png")
# plt.close("all")

