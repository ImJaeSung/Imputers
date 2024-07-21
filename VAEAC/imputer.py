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
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.utils import set_random_seed
from evaluation import evaluation_multiple # multiple imputation
from evaluation import evaluation # single imputation
#%%
import warnings
warnings.filterwarnings('ignore')
#from modules.evaluation_imputation import evaluate
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "VAEAC" # put your WANDB project name
# entity = "wotjd1410" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["imputation"], # put tags of this python project
)
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
#%%
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--ver', type=int, default=0, 
                        help='model version number')
    
    parser.add_argument('--dataset', type=str, default='loan', 
                        help="""
                        Dataset options: 
                        loan, kings, banknote, concrete, redwine, 
                        whitewine, breast, letter, abalone, anuran
                        """)
    
    parser.add_argument("--missing_type", default="MCAR", type=str,
                        help="how to generate missing: MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument("--missing_rate", default=0.3, type=float,
                        help="missing rate")
    
    parser.add_argument('--multiple', default=False, type=str2bool,
                        help="multiple imputation")
    parser.add_argument("--M", default=100, type=int,
                        help="the number of multiple imputation")
    

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
    
#%%
def main():
    #%%
    config = vars(get_args(debug=True))

    """model load"""
    base_name = f"{config['missing_type']}_{config['missing_rate']}_{config['dataset']}"
    model_name = f"VAEAC_{base_name}"
    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    #%%
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(config["seed"])
    wandb.config.update(config)

    assert config["missing_type"] != None
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    
    if config["multiple"]: 
        config["test_size"] = 0 # not using test data in multiple imputation
        train_dataset = CustomDataset(
            config,
            train=True
        )
    else:
        train_dataset = CustomDataset(
            config,
            train=True
        )
        test_dataset = CustomDataset(
            config,
            train=False,
        )
    #%%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    networks = model_module.get_imputation_networks(
        train_dataset.EncodedInfo.one_hot_max_sizes
    )

    model = model_module.VAEAC(
        config,
        networks,
        device
    ).to(device)
    
    if config["cuda"]:
        model.load_state_dict(
            torch.load(
                model_dir + "/" + model_name
            )
        )
    else:
        model.load_state_dict(
            torch.load(
                model_dir + "/" + model_name,
                map_location=torch.device("cpu"),
            )
        )
    model.eval()
    #%%
    """number of model parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params/1000000:.1f}M")
    wandb.log({"Number of Parameters": num_params / 1000000})
    #%%
    """imputation"""
    if config["multiple"]:
        results = evaluation_multiple.evaluate(
            train_dataset, model, M=config["M"]
        )
    else:
        imputed = model.impute(train_dataset, M=config["M"])
        
        # continuous: mean, discrete: mode
        cont_imputed = torch.mean(
            torch.stack(imputed)[:, :, :train_dataset.num_continuous_features],
            dim=0
        ) # [N, P(cont)]]
        disc_imputed = torch.mode(
            torch.stack(imputed)[:, :, train_dataset.num_continuous_features:],
            dim=0
        )[0] # [N, P(disc)]
        imputed = torch.cat((cont_imputed, disc_imputed), dim=1) # [M, N, P]
        imputed = pd.DataFrame(imputed, columns=train_dataset.features)

        results = evaluation.evaluate(
            imputed, train_dataset, test_dataset, config
        )
    
    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    #%% 
if __name__ == "__main__":
    main()