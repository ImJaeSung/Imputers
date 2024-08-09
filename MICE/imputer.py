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
import wandb
import warnings
warnings.filterwarnings('ignore')
#from modules.evaluation_imputation import evaluate

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


"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed) 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def save_args_to_file(data, missing_type,vv,bias,coverage,interval, filename="output.txt"):

    with open(filename, "a") as file: 
        file.write(f"{data}, {missing_type},{vv}, {bias},{coverage},{interval}\n")  


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
    


def main():

    config = vars(get_args())

    # run = wandb.init(
    # project="MICE", # put your WANDB project name
    # entity="hahaha990516", # put your WANDB username
    # tags=["inference", "imputation"]) # put tags of this python project)

    """model load"""
    base_name = f"{config['missing_rate']}_{config['missing_type']}_{config['dataset']}"
    model_name = f"MICE_{base_name}"
    set_random_seed(config['seed'])
    dataset_module = importlib.import_module('datasets.imputation')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    #from datasets.imputation import CustomDataset

    """dataset"""
    train_dataset = CustomDataset(config)

    train_dataset.raw_data[train_dataset.categorical_features]
    results = evaluate(train_dataset,config['seed'],config["M"])
    

    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")

#            wandb.log({f"{x}": y})
    

    # wandb.log({'Marginal Histogram': wandb.Image(fig)})
    # wandb.config.update(config, allow_val_change=True)

    # wandb.run.finish()
    save_args_to_file(config['dataset'], config['missing_type'],config['seed'], results.bias, results.coverage, results.interval, filename="output.txt")

    #%% 
if __name__ == "__main__":
    main()