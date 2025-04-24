import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
import sys
import gc
import psutil
from datetime import datetime
import os
#from __future__ import print_function
import transformations as trf                             
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ExponentialLR
# use ode solver that can operate on gpu
from torchdiffeq import odeint
from torch.utils.data import Dataset, DataLoader


from MLUnfolding.Tools.user  import plot_directory

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel', action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--train',  action='store', type=str, default="NA") # ./mldata/ML_Data_train.npy
argParser.add_argument('--val',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--plot_dir',    action='store', type=str, default="MLUnfolding_tmp") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--load_model_file',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--save_model_path',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--training_weight_cut', action='store', type=float, default=0.0) # ./mldata/ML_Data_validate.npy
argParser.add_argument('--lr', action='store', type=float, default=1e-4) # ./mldata/ML_Data_validate.npy
argParser.add_argument('--text_debug',    action='store', type=bool, default=False) #./mldata/ML_Data_validate.npy
argParser.add_argument('--hidden_dim', action='store', type=int, default=64)
argParser.add_argument('--hidden_layers', action='store', type=int, default=3)

args = argParser.parse_args()
text_debug= args.text_debug

learning_rate = args.lr

print("Learning Rate=", learning_rate)

w_cut = args.training_weight_cut

plot_dir = os.path.join(plot_directory, args.plot_dir)# Zusammenpasten von plot_dir
if not os.path.exists( plot_dir ): os.makedirs( plot_dir )

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
        data_dim: int,  # Number of features in the data
        hidden_dim: int,  # Number of hidden layer nodes
        hidden_layers: int = 3,  # Number of hidden layers
    ):
        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        layers = [nn.Linear(data_dim + 1, hidden_dim), nn.ReLU()]  # Input layer

        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, data_dim // 2))  # Output layer

        self.net = nn.Sequential(*layers)


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
        x_1 = torch.randn(n_samples, self.data_dim, device=device, dtype=dtype)

        # Define net_wrapper inside sample, capturing `y` (which is already batch-aligned)
        def net_wrapper(t, x_t):
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

cuda = torch.cuda.is_available() 
if cuda:
    device = torch.device("cuda:0")       
else:
    device = torch.device("cpu")  
    
print(device)

text_debug= False#True#False
plot_debug = False#True#False#
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

print(args.train)
try :
    with open(args.train, "rb") as f:
        train_data = np.load(f)
        if np.shape(train_data)[0] > 10000000 : #SH: 1M 
            train_data = train_data[0:10000000]
        f.close()
except FileNotFoundError :
    print("File "+ args.train+" (Train Data) not found.")
    exit(1)
    
train_data = train_data[train_data[:,weight_gen_index] > w_cut]
    
print("Train Shape: " + str(train_data.shape))

train_data_lenght = np.shape(train_data)[0]
train_data_n_cols = np.shape(train_data)[1]
## transform data and save max, min, mean, std values for the backtransformation later
max_values = np.max(train_data, keepdims=True, axis=0)*1.1
min_values = np.min(train_data, keepdims=True, axis=0)/1.1

transformed_data, mask = trf.normalize_data(train_data, max_values, min_values)
transformed_data = trf.logit_data(transformed_data)
mean_values = np.mean(transformed_data, keepdims=True, axis=0)
std_values = np.std(transformed_data, keepdims=True, axis=0)
transformed_data = trf.standardize_data(transformed_data, mean_values, std_values)

try :
    with open(args.val, "rb") as f:
        val_data = np.load(f)
        if np.shape(val_data)[0] > 10000000 :
            val_data = val_data[0:10000000]
        f.close()
except FileNotFoundError :
    print("File"+ args.val+" not found.")
    exit(1)
    
#use pt cut here:
val_data = val_data[val_data[:,weight_gen_index] > w_cut]

val_transformed_data, mask = trf.normalize_data(val_data, max_values, min_values)
val_transformed_data = trf.logit_data(val_transformed_data)
val_transformed_data = trf.standardize_data(val_transformed_data, mean_values, std_values)

print("Train Shape: " + str(transformed_data.shape))
print("Valid Shape: " + str(val_transformed_data.shape))

epochs = 200
batch_size = 512 #256
model_id = 10
n_features= 3

train_dataset = TransformedDataset(transformed_data, gen_index, rec_index, device)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("Using ", args.hidden_dim, "Hidden nodes, ", args.hidden_layers)


## --<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>----<>--
## Training

cfm = CFM(
    data_dim = n_features*2,
    hidden_dim = args.hidden_dim,
    hidden_layers = args.hidden_layers
    
)

optimizer = torch.optim.Adam(cfm.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_dataloader), epochs=epochs)

loss_function_in = []
loss_function_out=[]

losses = np.zeros(epochs)

if args.save_model_path != "NA": # untrained model
    save_model_path = os.path.join(args.save_model_path)
    if not os.path.exists( save_model_path ): os.makedirs( save_model_path )
    save_model_file = save_model_path+"/m"+str(model_id)+"f"+str(n_features)+"e"+"00of"+str(epochs)+".pt"
    torch.save(cfm, save_model_file)
    
    print("")
    print("Training ongoing:")
    for epoch in range(epochs):
                    
        gc.collect() #SH Test Garbage Collection
        print("")
        print("Epoch "+str(epoch+1)+"/"+str(epochs), end="\t")
        print(psutil.Process().memory_info().rss / (1024 * 1024)) #SH: Memory Usage Output in MB
        print("")
        
        epoch_losses = []
        for i, (x, y) in enumerate(train_dataloader):
            #print(type(i),type(x),type(y))
            loss = cfm.batch_loss(x,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())
            gc.collect() #SH Test Garbage Collection
        
        epoch_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}: loss = {epoch_loss}")
        losses[epoch] = epoch_loss
        
        if args.save_model_path != "NA":
            save_model_path = os.path.join(args.save_model_path, "weight_cut_" + str(w_cut).replace('.', 'p'))
            if not os.path.exists( save_model_path ): os.makedirs( save_model_path )
            save_model_file = save_model_path+"/m"+str(model_id)+"f"+str(n_features)+"e"+str(epoch+1).zfill(3)+"of"+str(epochs)+".pt" # 3of50 = after the 3rd training
            torch.save(cfm, save_model_file)
        
now = datetime.now()
current_time = now.strftime("%H_%M_%S")

if args.save_model_path != "NA":
    save_model_path = os.path.join(args.save_model_path, "weight_cut_" + str(w_cut).replace('.', 'p'))
    if not os.path.exists( save_model_path ): os.makedirs( save_model_path )

    plt.plot(np.arange(1, epochs+1), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss Function")
    plt.legend()
    plt.savefig(save_model_path+"/loss"+current_time+".png")
    plt.close("all")