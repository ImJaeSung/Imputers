#%%
import os
import torch
import argparse
import importlib

from modules.gaussian_copula import GaussianCopula
from evaluation.evaluation_imputation import evaluate
from evaluation.simulation import set_random_seed
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

project="xxx", # put your WANDB project name
entity="xxx", # put your WANDB username

run = wandb.init(
    project=project,
    entity=entity,
    tags=["imputation"] # put tags of this python project
)
#%%
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
        
    parser.add_argument('--seed', type=int, default=0, 
                        help='for the repeatable result')
    
    parser.add_argument('--imputation', default=True, type=str2bool)
    parser.add_argument('--dataset', default='abalone', type=str,
                        help="Daataset options: abalone, banknote, breast, redwine, whitewine")
    parser.add_argument('--missing_type', default='MCAR', type=str,
                        help="how to generate missing: MCAR, MAR, MNARL, MNARQ")
    parser.add_argument('--missing_rate', default=0.3, type=float,
                        help="missing rate")

    parser.add_argument("--M", default=100, type=int,
                        help="the number of multiple imputation")
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(config["seed"])
    wandb.config.update(config)
    
    assert config["missing_type"] != None
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.imputation')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    
    train_dataset = CustomDataset(config)
    train_dataset_array = CustomDataset(config).data
    #%%
    """imputation evaluation"""
    model = GaussianCopula(verbose=1, random_state=config["seed"])
    model.fit(X=train_dataset_array)
    full_imputed = model.sample_imputation(train_dataset_array, num=config["M"]) # (n, p, b)
    results = evaluate(train_dataset, full_imputed, M=config["M"])
    
    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    #%%
    """number of parameters"""
    p = model.get_params()["copula_corr"].shape[1]
    num_params = p*(p-1)/2
    wandb.log({"Number of Parameters": num_params / 1000000})
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    #%% 
if __name__ == "__main__":
    main()