#%%
import os
import torch
import argparse
import importlib
import ime

import modules
from modules import utils
from modules.utils import *

from evaluation import evaluation_multiple # multiple imputation
from evaluation import evaluation # single imputation
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

project = "not-MIWAE" # put your WANDB project name
# entity = "wotjd1410" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity,
    tags=["imputation"], # put tags of this python project
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
    parser.add_argument('--dataset', type=str, default='loan', 
                        help="""
                        Dataset options: 
                        abalone, anuran, banknote, breast, concrete,
                        kings, letter, loan, redwine, whitewine
                        """)
    
    parser.add_argument("--missing_type", default="MCAR", type=str,
                        help="how to generate missing: MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument("--missing_rate", default=0.3, type=float,
                        help="missing rate") 
    
    # multiple imputation
    parser.add_argument('--multiple', default=False, type=str2bool,
                        help="multiple imputation")
    parser.add_argument("--M", default=20, type=int,
                        help="the number of multiple imputation")
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration

    """model load"""
    model_name = f"not-MIWAE_{config['dataset']}_{config['missing_type']}_{config['missing_rate']}"

    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model'
    )
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    
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
        train=True
    )
    test_dataset = CustomDataset(
        config,
        scalers=train_dataset.scalers,
        train=False,
    )
    #%%
    """model load"""
    model_module = importlib.import_module("modules.model")
    importlib.reload(model_module)
    model = model_module.notMIWAE(
        config, 
        train_dataset.EncodedInfo, 
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
    """number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params / 1000}K")
    wandb.log({"Number of Parameters": num_params / 1000})
    #%%
    """imputation"""
    if config["multiple"]:
        results = evaluation_multiple.evaluate(train_dataset, model, config['M'])
    else:
        start_time = time.time()
        
        imputed = model.impute(
            train_dataset, M=config['M'], seed=config["seed"])
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"not-MIWAE (imputation): {elapsed_time:.4f} seconds")
        
        results = evaluation.evaluate(imputed, train_dataset, test_dataset, config, device)
    
    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    #%% 
if __name__ == "__main__":
    main()