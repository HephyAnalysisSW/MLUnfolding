import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
import sys
from datetime import datetime
import os
#from __future__ import print_function

from MLUnfolding.Tools.user  import plot_directory

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel', action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--train',  action='store', type=str, default="NA") # ./mldata/ML_Data_train.npy
argParser.add_argument('--val',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--plot_dir',    action='store', type=str, default="MLUnfolding_tmp") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--load_model_file',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--save_model_path',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--text_debug',    action='store', type=bool, default=False) #./mldata/ML_Data_validate.npy

args = argParser.parse_args()


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

text_debug= False#True#False
plot_debug = False#True#False#
table_debug = False#True#False

try :
    with open(args.train, "rb") as f:
        train_data = np.load(f)
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

#Plot for visual validation
if plot_debug == True : 
    n_bins= 100
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(train_data[:,0], bins=n_bins)
    axs[1].hist(train_data[:,1], bins=n_bins)
    axs[0].set_xlabel("jet mass [GeV]")
    axs[0].set_ylabel("events")
    axs[1].set_xlabel("p_t [GeV]")
    axs[1].set_ylabel("events")
    plt.show()

## defining variable transformation und retransformation
def normalize_data(in_data, max_val, min_val):
  new_data = (in_data-min_val)/(max_val-min_val)
  mask = np.prod(((new_data < 1) & (new_data > 0 )), axis=1, dtype=bool)
  new_data = new_data[mask]
  return new_data, mask

def logit_data(in_data):
  new_data = np.log(in_data/(1-in_data))
  return new_data

def standardize_data(in_data, mean_val, std_val):
  new_data = (in_data - mean_val)/std_val
  return new_data

## defining their inverse transformations
def normalize_inverse(in_data, max_val, min_val):
  new_data = in_data*(max_val-min_val) + min_val
  return new_data

def logit_inverse(in_data):
  #in_data = in_data.astype(np.longdouble)
  new_data = (1+np.exp(-in_data))**(-1)
  return new_data

def standardize_inverse(in_data, mean_val, std_val):
  new_data = std_val*in_data + mean_val
  return new_data

## transform data and save max, min, mean, std values for the backtransformation later
max_values = np.max(train_data, keepdims=True, axis=0)*1.1
min_values = np.min(train_data, keepdims=True, axis=0)/1.1

if text_debug == True : 
    print("")
    print("Input Training\t" + str(np.shape(train_data)[0]))
## normalize
transformed_data, mask = normalize_data(train_data, max_values, min_values)

if text_debug == True : 
    print("Normalized\t" + str(np.shape(transformed_data)[0]))
## logit
transformed_data = logit_data(transformed_data)

if text_debug == True : 
    print("Logit\t\t" + str(np.shape(transformed_data)[0]))
## standardize
mean_values = np.mean(transformed_data, keepdims=True, axis=0)
std_values = np.std(transformed_data, keepdims=True, axis=0)
transformed_data = standardize_data(transformed_data, mean_values, std_values)
if text_debug == True : 
    print("Final Training\t" + str(np.shape(transformed_data)[0]))

if table_debug == True : 
    print("")
    print("Transformed Training Data:")
    print(transformed_data)

#Plot for visual validation
if plot_debug == True : 
    n_bins= 100
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(transformed_data[:,0], bins=n_bins)
    axs[1].hist(transformed_data[:,1], bins=n_bins)
    axs[0].set_xlabel("transformed jet mass [<NA>]")
    axs[0].set_ylabel("events")
    axs[1].set_xlabel("transformed p_t [<NA>]")
    axs[1].set_ylabel("events")
    plt.show()
    
#SH:  (from: https://colab.research.google.com/drive/159Uova_QyCMPi8ar-y-V2ouzWSpztUK1#scrollTo=i9xbcZsXw65-)


base_dist = StandardNormal(shape=[2])
##SH: Setup of Flow
n_features = 2
n_features_con = 2
n_layers = 6
base_dist = StandardNormal(shape=[n_features])

transforms = []
for i in range(0, n_layers):
    transforms.append(MaskedAffineAutoregressiveTransform(features=n_features, hidden_features=32, context_features=n_features_con))
    transforms.append(ReversePermutation(features=n_features))

transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist).to(device)
optimizer = optim.Adam(flow.parameters(), lr=1e-4)

## --<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>--
## Training


num_epochs = 10
batch_size = 256
model_id = 1

loss_function = []

max_batches = int(transformed_data.shape[0] / batch_size)

if args.load_model_file == "NA" :
    print("")
    print("Training ongoing:")
    for i in range(num_epochs):
        permut = np.random.permutation(transformed_data.shape[0])
        transformed_data_shuffle = transformed_data[permut]
        if i % 1 == 0: #Only for debugging reasons
            print("")
            print("Epoch "+str(i+1)+"/"+str(num_epochs), end="\t")
          
        for i_batch in range(max_batches):
            x = transformed_data_shuffle[i_batch*batch_size:(i_batch+1)*batch_size,0:2]
            x = torch.tensor(x, device=device).float()
            
            y = transformed_data_shuffle[i_batch*batch_size:(i_batch+1)*batch_size,2:4]
            y = torch.tensor(y, device=device).float()#.view(-1, 1)
            
            optimizer.zero_grad()

            nll = -flow.log_prob(x, context=y) # Feed context

            loss = nll.mean()

            if i_batch % 50 == 0:
                print(round(loss.item(),2),end ="\t")
                sys.stdout.flush()
                
            loss.backward()
            optimizer.step()
        loss_function.append(loss.item())
    if args.save_model_path != "NA":
        save_model_path = os.path.join(args.save_model_path)
        if not os.path.exists( save_model_path ): os.makedirs( save_model_path )
        save_model_file = save_model_path+"/m"+str(model_id)+"f"+str(n_features)+"e"+str(num_epochs)+".pt"
        torch.save(flow, save_model_file)
else : 
    print("")
    try:
        flow =torch.load(args.load_model_file)
        flow.eval()
    except Exception as e:
        print("Not able to load given flow " + args.load_model_file)
        exit(0)
    print("Skipping Traning | Instead using flow " +args.load_model_file)
#print(type(flow))

## --<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>--
print("\nSampling now:")

#val_data = transformed_data_shuffle[5000:10000,2:4]
#val_data = torch.tensor(val_data, device=device).float()

#SH: Redundant Exception Handling. Works though
try :
    with open(args.val, "rb") as f:
        val_data = np.load(f)
        f.close()
except FileNotFoundError :
    print("File \""+ args.val+"\" not found.")
    exit(1)
 
val_data[:,2] = 200 # Fr sample
val_data[:,0] = 200 # Fr Plot

## normalize Val Data
val_transformed_data, mask = normalize_data(val_data, max_values, min_values)
## logit Val Data
val_transformed_data = logit_data(val_transformed_data)
## standardize Val Data
val_transformed_data = standardize_data(val_transformed_data, mean_values, std_values)
val_trans_cond = torch.tensor(val_transformed_data[:,2:4], device=device).float()
val_data = val_data[mask]

#val_data[:,0] = 1

with torch.no_grad():
  samples = flow.sample(1, context=val_trans_cond).view(val_trans_cond.shape[0], -1).cpu().numpy()
## inverse standardize
retransformed_samples = standardize_inverse(samples, mean_values[:,0:2], std_values[:,0:2])
## inverse logit
retransformed_samples = logit_inverse(retransformed_samples)
## inverse normalize
retransformed_samples = normalize_inverse(retransformed_samples, max_values[:,0:2], min_values[:,0:2])

if text_debug :
    print("val data shape =",     val_trans_cond.shape)
    print("sampled data shape =", retransformed_samples.shape)

now = datetime.now()
current_time = now.strftime("%H_%M_%S")

#Save sampled data
n_bins= 100
fig, axs = plt.subplots(1, 2)
axs[0].hist(retransformed_samples[:,0], bins=n_bins, histtype="step", label="ML Generated")
axs[0].hist(val_data[1:10000,0],   bins=n_bins, histtype="step", label="Training Data")
axs[1].hist(retransformed_samples[:,1], bins=n_bins, histtype="step", label="ML Generated")
axs[1].hist(val_data[:,1],   bins=n_bins, histtype="step", label="Training Data")
axs[0].legend()
axs[1].legend()
axs[0].set_xlabel("jet mass [GeV]")
axs[0].set_ylabel("Number of Events")
axs[1].set_xlabel("p_t [GeV]")
axs[1].set_ylabel("Number of Events")
plt.savefig(plot_dir+"/generated_data"+current_time+".png")
plt.close("all")
#Save 2D Histogramm

n_bins= 100
#,norm=mpl.colors.LogNorm()
plt.hist2d(retransformed_samples[:,1],val_data[:,1],normed=True, bins=n_bins,cmap=plt.cm.jet)
plt.xlabel("ML p_t [GeV]")
plt.ylabel("Train p_t [GeV]")
plt.colorbar()
plt.xlim(300, 1000)
plt.ylim(300, 1000)
plt.savefig(plot_dir+"/2dhist_pt"+current_time+".png")
plt.close("all")

plt.hist2d(retransformed_samples[:,0],val_data[:,0],normed=True, bins=n_bins,cmap=plt.cm.jet)
plt.xlabel("ML jet mass [GeV]")
plt.ylabel("Train jet mass [GeV]")
plt.colorbar()
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.savefig(plot_dir+"/2dhist_m"+current_time+".png")
plt.close("all")

it=[*range(len(loss_function))]

plt.plot(it,loss_function, label="total of" + str(num_epochs)+ "epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss Function")
plt.legend()
plt.savefig(plot_dir+"/loss"+current_time+".png")
plt.close("all")