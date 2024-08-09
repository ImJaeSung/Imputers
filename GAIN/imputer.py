#%%
import os
import torch
import argparse
import importlib
import pandas as pd
import random
import numpy as np

from argparse import ArgumentParser
from copy import deepcopy
from importlib import import_module
from math import ceil
from os.path import exists, join
from sys import stderr
import importlib
import random 

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from evaluation_imputation import evaluate
# from VAEAC.no_datasets import compute_normalization
from utils import  normalization, renormalization, rounding, rmse_loss,binary_sampler,uniform_sampler,sample_M,sample_Z
from model import Generator, Discriminator

import wandb
import warnings
warnings.filterwarnings('ignore')

#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb


#%%
"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed) 

# %%
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def save_args_to_file(data, missing_type,seed,bias,coverage,interval, filename="output.txt"):

    with open(filename, "a") as file: 
        file.write(f"{data}, {missing_type},{seed}, {bias},{coverage},{interval}\n")  


def get_args(arg_list=None, debug=False):
    parser = argparse.ArgumentParser('parameters')
    

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

    parser.add_argument("--M", default=10, type=int,
                        help="the number of multiple imputation")
    

    if debug:
        return parser.parse_args(args = arg_list)
    else:    
        return parser.parse_args()
    
#%%

def main():

    config = vars(get_args(debug=False))

    # run = wandb.init(
    # project="GAIN", # put your WANDB project name
    # entity="", # put your WANDB username
    # tags=["inference", "imputation"]) # put tags of this python project)


    """model load"""
    base_name = f"{config['missing_rate']}_{config['missing_type']}_{config['dataset']}"
    model_name = f"GAIN_{base_name}"
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(config['seed'])


    """dataset"""
    dataset_module = importlib.import_module('datasets.imputation')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    train_dataset = CustomDataset(config)

    total_data = train_dataset.data
    total_data = np.array(total_data)
    total_data, parameters = normalization(total_data)
    no = len(total_data)
    dim = len(total_data[0,:])

    H_Dim1 = dim
    H_Dim2 = dim
    batch_size = 128
    total_dataloader = DataLoader(total_data, batch_size=batch_size, shuffle=False,
                        drop_last=False)
    
    D = Discriminator(dim, H_Dim1, H_Dim2)
    G = Generator(dim, H_Dim1, H_Dim2)

    model_path = './assets/models/{}/{}_{}.pth'.format(base_name,model_name,config['seed'])
    model_state = torch.load(model_path,map_location=torch.device('cpu'))
    G.load_state_dict(model_state['G_state_dict'])
    D.load_state_dict(model_state['D_state_dict'])

    G.eval()
    D.eval()

    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    D_num_params = count_parameters(D)
    G_num_params = count_parameters(G)
    num_params = D_num_params + G_num_params
    print(f"Number of Parameters: {num_params / 1000000:.1f}M")
    # wandb.log({"Number of Parameters": num_params / 1000000})

    
    results = evaluate(train_dataset,total_dataloader,parameters, G,D,config["M"])
    

    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        # wandb.log({f"{x}": y})
    

    # wandb.log({'Marginal Histogram': wandb.Image(fig)})
    # wandb.config.update(config, allow_val_change=True)

    # #%%
    # wandb.run.finish()
    # save_args_to_file(config['dataset'], config['missing_type'],config['seed'], results.bias, results.coverage, results.interval, filename="output.txt")

    #%% 
if __name__ == "__main__":
    main()