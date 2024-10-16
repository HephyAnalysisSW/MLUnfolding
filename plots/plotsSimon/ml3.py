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
import transformations as trf                             

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
        if np.shape(train_data)[0] > 1000000 :
            train_data = train_data[0:1000000]
        f.close()
except FileNotFoundError :
    print("File \""+ args.train+"\" (Train Data) not found.")
    exit(1)

train_data_lenght = np.shape(train_data)[0]
train_data_n_cols = np.shape(train_data)[1]
## transform data and save max, min, mean, std values for the backtransformation later
max_values = np.max(train_data, keepdims=True, axis=0)*1.1
min_values = np.min(train_data, keepdims=True, axis=0)/1.1

print(max_values)
print(min_values)

transformed_data, mask = trf.normalize_data(train_data, max_values, min_values)
transformed_data = trf.logit_data(transformed_data)
mean_values = np.mean(transformed_data, keepdims=True, axis=0)
std_values = np.std(transformed_data, keepdims=True, axis=0)
transformed_data = trf.standardize_data(transformed_data, mean_values, std_values)

try :
    with open(args.val, "rb") as f:
        val_data = np.load(f)
        if np.shape(val_data)[0] > 1000000 :
            val_data = val_data[0:1000000]
        f.close()
except FileNotFoundError :
    print("File \""+ args.val+"\" not found.")
    exit(1)

val_transformed_data, mask = trf.normalize_data(val_data, max_values, min_values)
val_transformed_data = trf.logit_data(val_transformed_data)
val_transformed_data = trf.standardize_data(val_transformed_data, mean_values, std_values)
#val_transformed_data = val_transformed_data[mask]
#val_transformed_data = np.repeat(val_transformed_data,3,axis=0)

print("Train Shape: " + str(transformed_data.shape))
print("Valid Shape: " + str(val_transformed_data.shape))
 
       
#SH:  (from: https://colab.research.google.com/drive/159Uova_QyCMPi8ar-y-V2ouzWSpztUK1#scrollTo=i9xbcZsXw65-)


base_dist = StandardNormal(shape=[2])
##SH: Setup of Flow
n_features = 2
n_features_con = 2
n_layers = 6
base_dist = StandardNormal(shape=[n_features])

transforms = []
for i in range(0, n_layers):
    #transforms.append(MaskedAffineAutoregressiveTransform(features=n_features, hidden_features=32, context_features=n_features_con))
    transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=n_features, hidden_features=32, context_features=n_features_con))
    transforms.append(ReversePermutation(features=n_features))

transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist).to(device)
optimizer = optim.Adam(flow.parameters(), lr=1e-4)

## --<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>--
## Training


num_epochs = 25
batch_size = 256
model_id = 3

loss_function_in = []
loss_function_out=[]

max_batches = int(transformed_data.shape[0] / batch_size)

if args.save_model_path != "NA": # untrained model
    save_model_path = os.path.join(args.save_model_path)
    if not os.path.exists( save_model_path ): os.makedirs( save_model_path )
    save_model_file = save_model_path+"/m"+str(model_id)+"f"+str(n_features)+"e"+"00of"+str(num_epochs)+".pt"
    torch.save(flow, save_model_file)
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
            
            max_values = torch.max(x, keepdims=True, axis=0)
            min_values = torch.min(x, keepdims=True, axis=0)

            print(max_values)
            print(min_values)
            
            
            nll = -flow.log_prob(x, context=y) # Feed context
            loss = nll.mean()

            if i_batch % 50 == 0:
                print(round(loss.item(),2),end ="\t")
                sys.stdout.flush()
                
            loss.backward()
            optimizer.step()
        #Calculate In Error
        x_train = transformed_data_shuffle[:,0:2]
        x_train = torch.tensor(x_train, device=device).float()
        y_train = transformed_data_shuffle[:,2:4]
        y_train = torch.tensor(y_train, device=device).float()#.view(-1, 1)
        
        nll_in = -flow.log_prob(x_train, context=y_train) # Feed context
        loss_in = nll_in.mean()
        loss_function_in.append(loss_in.item())
        
        #Calculate Out Error 
        x_val = val_transformed_data[:,0:2]
        x_val = torch.tensor(x_val, device=device).float()
        y_val = val_transformed_data[:,2:4]
        y_val = torch.tensor(y_val, device=device).float()
        
        nll_out = -flow.log_prob(x_val, context=y_val)
        loss_out = nll_out.mean()
        loss_function_out.append(loss_out.item())
        if args.save_model_path != "NA":
            save_model_path = os.path.join(args.save_model_path)
            if not os.path.exists( save_model_path ): os.makedirs( save_model_path )
            save_model_file = save_model_path+"/m"+str(model_id)+"f"+str(n_features)+"e"+str(i+1).zfill(2)+"of"+str(num_epochs)+".pt" # 3of50 = after the 3rd training
            torch.save(flow, save_model_file)
            
now = datetime.now()
current_time = now.strftime("%H_%M_%S")
it=[*range(len(loss_function_in))]
# plt.plot(it,loss_function_in, label="Train-Loss", color="#696969") 
# plt.plot(it,loss_function_out, label="Validation-Loss", color="red", alpha=0.5)
# plt.xlabel("Epochs")
# plt.ylabel("Loss Function")
# plt.legend()
# plt.savefig(plot_dir+"/loss"+current_time+".png")
# plt.close("all")