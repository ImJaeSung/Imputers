#%%
import torch
import numpy as np
# from tqdm import tqdm

import torch.nn.functional as F

from argparse import ArgumentParser
from copy import deepcopy
from importlib import import_module
from math import ceil
from os.path import exists, join
import json
from sys import stderr
import importlib
import random 
import os 


from utils import  normalization, renormalization, rounding,test_loss, generator_loss, discriminator_loss, sample_M,sample_Z
from model import Generator, Discriminator

import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

#%%
# run = wandb.init(
#     project="GAIN", # put your WANDB project name
#     # entity="", # put your WANDB username
#     tags=['Train'], # put tags of this python project
# )

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed) 


parser = ArgumentParser(description='Missing Features Multiple Imputation.')

parser.add_argument('--dataset', type=str, default='abalone', 
                    help="""
                    Dataset options: banknote, redwine, whitewine, breast, abalone
                    """)

parser.add_argument("--seed", default=0, type=int,required=True,
                    help="selcet version number ") 

parser.add_argument("--missing_type", default="MCAR", type=str,
                    help="how to generate missing: None(complete data), MCAR, MAR, MNARL, MNARQ") 

parser.add_argument("--missing_rate", default=0.3, type=float,
                    help="missing rate") 

parser.add_argument('--epochs', type=int, required=True,
                    help='Number epochs to train VAEAC.')

def main():

    args = parser.parse_args()
    config = vars(args)
    set_random_seed(config['seed'])

    model_dir = f"./assets/models/{config['dataset']}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_name = f"GAIN_{config['dataset']}_{config['missing_type']}_{config['seed']}"


    with open(f"./{model_dir}/config_{config['seed']}.json", "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)    


    base_name = f"{config['missing_type']}_{config['dataset']}"
    if config["missing_type"] != "None":
        base_name = f"{config['missing_rate']}_" + base_name
    model_dir = f"./assets/models/{base_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    ### model order has config 
    model_name = f"GAIN_{base_name}_{config['seed']}"

    ## GPU Setting
    use_gpu = use_cuda = torch.cuda.is_available() 
    if use_gpu:
        torch.cuda.set_device(0)

    ## Import Library for Dataset 
    dataset_module = importlib.import_module('datasets.imputation')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    train_dataset = CustomDataset(config)

    total_column = train_dataset.continuous_features + train_dataset.categorical_features
    one_hot_max_sizes = []
    for i,column in enumerate(total_column):
        if column in train_dataset.continuous_features:
            one_hot_max_sizes.append(1)
        elif column in train_dataset.categorical_features:
            one_hot_max_sizes.append(train_dataset.raw_data[column].nunique())

    import pandas as pd
    raw_data = train_dataset.data
    total_data = train_dataset.data
    total_data = np.array(total_data)
    total_data, parameters = normalization(total_data)

    #Parameter Setting
    # 1. Mini batch size
    batch_size = 128
    # 2. Hint rate
    p_hint = 0.9
    # 3. Loss Hyperparameters
    alpha = 10


    no = len(total_data)
    dim = len(total_data[0,:])

    # Hidden state dimensions
    H_Dim1 = dim
    H_Dim2 = dim

    ## Train / Validation Setting 
    val_size = ceil(len(total_data) * 0.2)
    val_indices = np.random.choice(len(total_data), val_size, False)
    val_indices_set = set(val_indices)
    train_indices = [i for i in range(len(total_data)) if i not in val_indices_set]
    train_data = total_data[train_indices]
    val_data = total_data[val_indices]

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            drop_last=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                drop_last=False)
    total_dataloader = DataLoader(total_data, batch_size=batch_size, shuffle=False,
                            drop_last=False)
    #%%
    ## =========================== Modeling ===================================

    D = Discriminator(dim, H_Dim1, H_Dim2).cuda() if use_cuda else Discriminator(dim, H_Dim1, H_Dim2)
    G = Generator(dim, H_Dim1, H_Dim2).cuda() if use_cuda else Generator(dim, H_Dim1, H_Dim2)

    optimizer_D = torch.optim.Adam(D.parameters())
    optimizer_G = torch.optim.Adam(G.parameters())

    ## =========================== Train ===================================
    best_val_loss_G = float('inf')  
    best_val_loss_D = float('inf')  
    best_state_G = None  
    best_state_D = None  

    for epoch in tqdm(range(config['epochs'])):
        G.train()
        D.train()

        D_loss_list = []
        G_loss_list = []
        for i, batch_data in enumerate(dataloader):
            X_mb = batch_data.float().cuda() if use_cuda else batch_data 

            get_batch = batch_data.shape[0]
            M_mb = 1 - torch.isnan(batch_data).float()
            M_mb = M_mb.cuda() if use_cuda else M_mb     

            H_mb1 = sample_M(get_batch, dim, 0.1).float() 
            H_mb1 = H_mb1.cuda() if use_cuda else H_mb1
            H_mb = M_mb * H_mb1
            H_mb = H_mb.float()
            H_mb = H_mb.cuda() if use_cuda else H_mb

            Z_mb = sample_Z(get_batch, dim).float()
            Z_mb = Z_mb.cuda() if use_cuda else Z_mb

            X_mb = torch.nan_to_num(X_mb,nan=0.0)
            New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  
            New_X_mb = New_X_mb.float()
            New_X_mb = New_X_mb.cuda() if use_cuda else New_X_mb

            optimizer_D.zero_grad()
            D_loss_curr = discriminator_loss(D, G, M_mb, New_X_mb, H_mb)
            D_loss_curr.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            G_loss_curr, _, _ = generator_loss(D,G, X_mb, M_mb, New_X_mb, H_mb, alpha)
            G_loss_curr.backward()
            optimizer_G.step()

            D_loss_list.append(D_loss_curr.item())
            G_loss_list.append(G_loss_curr.item())


        ## ==================== Validation ====================
        G.eval()
        D.eval()
        with torch.no_grad():

            loss_G = 0
            loss_D = 0
            for i, val_data in enumerate(val_dataloader):
                X_val = val_data.float().cuda() if use_cuda else val_data
                get_batch = X_val.shape[0]

                M_val = 1 - torch.isnan(X_val).float() ## 1 : observed 0 : nan
                M_val = M_val.cuda() if use_cuda else M_val

                H_val1 = sample_M(get_batch, dim, 0.1).float()  
                H_val1 = H_val1.cuda() if use_cuda else H_val1
                H_val = M_val * H_val1
                H_val = H_val.float()
                H_val = H_val.cuda() if use_cuda else H_val

                X_val = torch.nan_to_num(X_val,nan=0.0)     
                Z_val = sample_Z(get_batch, dim).float()
                Z_val = Z_val.cuda() if use_cuda else Z_val

                New_X_val = M_val * X_val + (1-M_val) * Z_val  # Missing Data Introduce
                New_X_val = New_X_val.cuda() if use_cuda else New_X_val 

                val_loss_D = discriminator_loss(D, G, M_val, New_X_val, H_val)
                val_loss_G, _, _ = generator_loss(D, G, X_val, M_val, New_X_val, H_val, alpha)

                loss_D += val_loss_D
                loss_G += val_loss_G

            loss_D /= len(val_dataloader)
            loss_G /= len(val_dataloader)

            if loss_G < best_val_loss_G:
                best_val_loss_G = loss_G
                best_state_G = G.state_dict()
                print(f"Generator model updated with val_loss_G: {best_val_loss_G:.4f}")

            if loss_D < best_val_loss_D:
                best_val_loss_D = loss_D
                best_state_D = D.state_dict()
                print(f"Discriminator model updated with val_loss_D: {best_val_loss_D:.4f}")

            # wandb.log({'val_loss_D': loss_D, 'val_loss_G': loss_G})
            print(f"Epoch {epoch+1}, Val Loss G: {loss_G:.4f}, Val Loss D: {loss_D:.4f}")



    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    D_num_params = count_parameters(D)
    G_num_params = count_parameters(G)
    num_params = D_num_params + G_num_params
    print(f"Number of Parameters: {num_params / 1000000:.1f}M")
    # wandb.log({"Number of Parameters": num_params / 1000000})
   
   

    print("Saving the final models...")
    torch.save({
        'G_state_dict': best_state_G,
        'D_state_dict': best_state_D,
    }, f"./{model_dir}/{model_name}.pth")

    
    # artifact = wandb.Artifact(
    #     "_".join(model_name.split("_")[:-1]),  
    #     type='model',
    #     metadata=config) 
    # artifact.add_file(f"./{model_dir}/{model_name}.pth")
    # wandb.log_artifact(artifact)
    # wandb.run.finish()
if __name__ == "__main__":
    main()