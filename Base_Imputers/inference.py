#%%
import os
import torch
import argparse
import importlib
import pandas as pd

from torch.utils.data import DataLoader

import modules
from modules import utils
from modules.utils import *
from evaluation.evaluation import evaluate
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

project = "baselines" # put your WANDB project name
# entity = "wotjd1410" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["imputation", "Baseline"], # put tags of this python project
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
    
    parser.add_argument('--ver', type=int, default=0, 
                        help='model version number')
    parser.add_argument("--model", type=str, default="mean",
                        help="""
                        Model options:
                        mean, median, missforest, mice, softimpute,
                        EM, sinkhorn, gain, miwae, miracle,
                        ReMasker, KNNI, complete, zero
                        """)
    parser.add_argument('--dataset', type=str, default='loan', 
                        help="""
                        Dataset options: 
                        abalone, anuran, banknote, breast, concrete,
                        kings, letter, loan, redwine, whitewine
                        """)
    parser.add_argument("--missing_type", default="MAR", type=str,
                        help="how to generate missing: MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument("--missing_rate", default=0.2, type=float,
                        help="missing rate") 
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    base_name = f"baseline_{config['missing_rate']}_{config['missing_type']}_{config['dataset']}"
    artifact = wandb.use_artifact(
        f"{project}/{base_name}:v{config['ver']}",
        type='dataset')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(config["seed"])
    wandb.config.update(config)
    
    assert config["missing_type"] != None
    #%%
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset

    """dataset"""
    train_dataset = CustomDataset(
        config,
        train=True)
    test_dataset = CustomDataset(
        config,
        scalers=train_dataset.scalers,
        train=False)
    #%%
    data_name = f"{config['model']}_{config['seed']}.csv"
    imputed = pd.read_csv(model_dir + "/" + data_name).astype(float)
    #%%
    results = evaluate(imputed, train_dataset, test_dataset, config, device)
    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
        
    # print("Marginal Distribution...")
    # data_name = data_name.replace(".csv", ".pth")
    # data_name = f"{config['missing_rate']}_{config['missing_type']}_{config['dataset']}_" + data_name
    # figs = utils.marginal_plot(train_dataset.raw_data, imputed, config, data_name)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    #%% 
if __name__ == "__main__":
    main()
#%%