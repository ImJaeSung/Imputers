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
from modules.imputation_networks import get_imputation_networks
from modules.train_utils import extend_batch, get_validation_iwae
from VAEAC.modules.model import VAEAC
import wandb
import warnings
warnings.filterwarnings('ignore')
#from modules.evaluation_imputation import evaluate
#%%
#%%

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

#%%
def compute_normalization(data, one_hot_max_sizes):
    """
    Compute the normalization parameters (i. e. mean to subtract and std
    to divide by) for each feature of the dataset.
    For categorical features mean is zero and std is one.
    i-th feature is denoted to be categorical if one_hot_max_sizes[i] >= 2.
    Returns two vectors: means and stds.
    """
    norm_vector_mean = torch.zeros(len(one_hot_max_sizes))
    norm_vector_std = torch.ones(len(one_hot_max_sizes))
    for i, size in enumerate(one_hot_max_sizes):
        if size >= 2:
            continue
        v = data[:, i]
        v = v[~torch.isnan(v)]
        vmin, vmax = v.min(), v.max()
        vmean = v.mean()
        vstd = v.std()
        norm_vector_mean[i] = vmean
        norm_vector_std[i] = vstd
    return norm_vector_mean, norm_vector_std 

#%%
"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 대한 시드 고정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # NumPy 시드 고정
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
    
def save_args_to_file(data, missing_type,vv,bias,coverage,interval,rate, filename="output.txt"):

    with open(filename, "a") as file: 
        file.write(f"{data}, {missing_type},{vv}, {bias},{coverage},{interval},{rate}\n")  


def get_args(arg_list=None, debug=False):
    parser = argparse.ArgumentParser('parameters')
    

    parser.add_argument('--dataset', type=str, default='abalone', 
                        help="""Dataset options: 
                        * Complete datasets: covtype, loan, kings, banknote, concrete, 
                        redwine, whitewine, yeast, breast, spam, letter, abalone
                        * Incomplete datasets: credit, adult, cabs
                        """)
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
    #%%

#%%
    config = vars(get_args(debug=True))

#%%
    # run = wandb.init(
    # project="VAEAC", # put your WANDB project name
    # entity="hahaha990516", # put your WANDB username
    # tags=["inference", "imputation"]) # put tags of this python project)


#%%
# #hahaha990516/VAEAC/VAEAC_0.3_MCAR_letter:v10
    """model load"""
    base_name = f"{config['missing_rate']}_{config['missing_type']}_{config['dataset']}"
    model_name = f"VAEAC_{base_name}"
#%%
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


    set_random_seed(config['seed'])


    print(config)
    print(config)
    dataset_module = importlib.import_module('datasets.imputation')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    #from datasets.imputation import CustomDataset
    """dataset"""
    train_dataset = CustomDataset(config,vv)

#%%

    total_column = train_dataset.continuous_features + train_dataset.categorical_features

    one_hot_max_sizes = []

    for i,column in enumerate(total_column):
        if column in train_dataset.continuous_features:
            one_hot_max_sizes.append(1)
        elif column in train_dataset.categorical_features:
            one_hot_max_sizes.append(train_dataset.raw_data[column].nunique())

    networks = get_imputation_networks(one_hot_max_sizes)



    model = VAEAC(
    networks['reconstruction_log_prob'],
    networks['proposal_network'],
    networks['prior_network'],
    networks['generative_network'])
#%%
#### =========================이거 경로 항상 조심하기 ============================================

    model_path = './assets/models/{}/{}_{}.pth'.format(base_name,model_name,vv)




#### =============================================================================================
    model_state = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    
#%%
    use_cuda = torch.cuda.is_available()
    if use_cuda: 
        model = model.cuda()


    ## 모델 불러오기 파트 
    model.eval()

    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params / 1000000:.1f}M")
    # wandb.log({"Number of Parameters": num_params / 1000000})

    results = evaluate(train_dataset, model, networks,one_hot_max_sizes,config["M"])
    

    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        # wandb.log({f"{x}": y})
    

    config['seed'] = vv
    # wandb.log({'Marginal Histogram': wandb.Image(fig)})
    # wandb.config.update(config, allow_val_change=True)

    #%%
    # wandb.run.finish()
    save_args_to_file(config['dataset'], config['missing_type'],vv, results.bias, results.coverage, results.interval, config['missing_rate'], filename="output.txt")

    #%% 
if __name__ == "__main__":
    main()