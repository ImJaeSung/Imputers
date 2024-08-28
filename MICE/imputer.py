#%%
import os
import torch
import argparse
import importlib
import pandas as pd
import numpy as np

import torch

from modules.utils import set_random_seed, undummify
from evaluation import evaluation
from evaluation import evaluation_multiple
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "MICE" # put your WANDB project name
# entity = "wotjd1410" # put your WANDB username

run = wandb.init(
    project=project,
    # entity=entity,
    tags=['imputation'], # put tags of this python project
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
#%%
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument("--seed", type=int, default=0, 
                        help="seed for repeatable results")
    parser.add_argument('--dataset', type=str, default='loan', 
                        help="""
                        Dataset options: 
                        abalone, anuran, banknote, breast, concrete,
                        kings, letter, loan, redwine, whitewine
                        speed, nomao, musk, yeast, madelon
                        """)
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split") 
    
    parser.add_argument("--missing_type", default="MAR", type=str,
                        help="how to generate missing: MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument("--missing_rate", default=0.3, type=float,
                        help="missing rate") 
    
    parser.add_argument('--multiple', default=False, type=str2bool,
                        help="multiple imputation")
    parser.add_argument("--M", default=100, type=int,
                        help="the number of multiple imputation")
    
    parser.add_argument('--max_iter', type=int, default=10,
                        help='max iteration in MICE.')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()

#%% 
def main():
    #%%
    config = vars(get_args(debug=False))
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    #%%
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    wandb.config.update(config, allow_val_change=True)
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
              
    train_dataset = CustomDataset(
        config,
        train=True
    )

    test_dataset = CustomDataset(
        config,
        scalers=train_dataset.scalers,
        train=False
    )
    #%%
    """imputation"""
    if config["multiple"]:
        results = evaluation_multiple.evaluate(train_dataset, config, M=config["M"])

    else:
        model_module = importlib.import_module('modules.model')
        importlib.reload(model_module)
        model = model_module.MICE(
            max_iter=config["max_iter"],
            random_state=config["seed"]
        )
        imputed = pd.DataFrame(
            model.fit_transform(np.array(train_dataset.data)), 
            columns=train_dataset.data.columns
        )
        imputed = undummify(imputed, prefix_sep='###')

        # un-standardization of synthetic data
        for col, scaler in train_dataset.scalers.items():
            imputed[[col]] = scaler.inverse_transform(imputed[[col]])

        # post-process
        imputed[train_dataset.categorical_features] = imputed[train_dataset.categorical_features].astype(int)
        imputed[train_dataset.integer_features] = imputed[train_dataset.integer_features].round(0).astype(int)
        # display(imputed.head())

        results = evaluation.evaluate(imputed, train_dataset, test_dataset, config, device)
    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    # %%
    """complete data save"""
    base_name = f"MICE_{config['dataset']}_{config['missing_type']}_{config['missing_rate']}"
    data_dir = f"./assets/{config['dataset']}/{base_name}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    artifact = wandb.Artifact(
        base_name, 
        type='dataset',
        metadata=config) 
    imputed.to_csv(f"{data_dir}_{config['seed']}.csv", index=None)
    artifact.add_file(f"{data_dir}_{config['seed']}.csv")
    artifact.add_file('./modules/model.py')
    artifact.add_file('./imputer.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    # %%
if __name__ == "__main__":
    main()
# %%