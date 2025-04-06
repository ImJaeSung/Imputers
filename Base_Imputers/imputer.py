#%%
import os
import argparse
import importlib
import time

import pandas as pd
from IPython.display import display
import warnings
warnings.filterwarnings(action="ignore")

import torch
from torch.utils.data import DataLoader

# from modules.train import *
from modules.utils import set_random_seed, undummify
import wandb

import sys
sys.path.append("./remasker/")
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

project = "kdd_rebuttal1" # put your WANDB project name
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
    
    parser.add_argument('--seed', type=int, default=2, 
                        help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='yeast', 
                        help="""
                        Dataset options: 
                        abalone, anuran, banknote, breast, concrete,
                        kings, letter, loan, redwine, whitewine
                        yeast, nomao
                        """)
    
    parser.add_argument("--missing_type", default="MAR", type=str,
                        help="how to generate missing: MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument("--missing_rate", default=0.3, type=float,
                        help="missing rate (options: 0.2, 0.3, 0.4, 0.6, 0.8)") 
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split")     
    
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    set_random_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is', device)
    wandb.config.update(config)
    #%%
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    train_dataset = CustomDataset(
        config,
        train=True)
    p = train_dataset.data.shape[1]
    #%%
    imputed = []
    # """Complete"""
    # print(f"=====complete=====")
    # out = train_dataset.raw_data
    
    # # post-process
    # out[train_dataset.categorical_features] = out[train_dataset.categorical_features].astype(int)
    # out[train_dataset.integer_features] = out[train_dataset.integer_features].round(0).astype(int)
    # imputed.append(("complete", out))
    # display(out.head())
    # #%%
    # """Zero"""
    # print(f"=====zero=====")
    # out = pd.DataFrame(train_dataset.data, columns=train_dataset.features).fillna(0.)
    
    # """un-standardization of synthetic data"""
    # for col, scaler in train_dataset.scalers.items():
    #     out[[col]] = scaler.inverse_transform(out[[col]])
    
    # # post-process
    # out[train_dataset.categorical_features] = out[train_dataset.categorical_features].astype(int)
    # out[train_dataset.integer_features] = out[train_dataset.integer_features].round(0).astype(int)
    # imputed.append(("zero", out))
    # display(out.head())
    #%%
    from hyperimpute.plugins.imputers import Imputers
    imputer_list = Imputers().list()
    print(imputer_list)
    #%%
    """configuration"""
    imputer_list = [
        # ("mean", {"random_state": config["seed"]}),
        # # ("median", {"random_state": config["seed"]}),
        # ("missforest", {"random_state": config["seed"]}), 
        # ("mice", {"random_state": config["seed"]}),
        # ("softimpute", {"random_state": config["seed"]}),
        # ("EM", {"random_state": config["seed"]}),
        # ("sinkhorn", {}),
        # ("gain", {"random_state": config["seed"]}),
        # ("miwae", {"n_epochs": 2002, "batch_size": 32, "n_hidden": 128, "latent_size": 1, "random_state": config["seed"]}),
        ("miracle", {"lr":0.0005, "max_steps": 300, "n_hidden": p, "random_state": config["seed"]})
    ]
    #%%
    for method, args in imputer_list:
        print(f"====={method}=====")
        start_time = time.time()
        
        plugin = Imputers().get(method, **args)
        X = pd.DataFrame(train_dataset.data, columns=train_dataset.features)
        X = pd.get_dummies(
            X, columns=train_dataset.categorical_features, prefix_sep="###"
        ).astype(float)
        #%%        
        out = plugin.fit_transform(X.copy())
        
        if out.isnull().values.any():
            print(f"{method}: unstable result")
            continue

        out.columns = X.columns
        out = undummify(out)
        out.columns = train_dataset.features
        
        """un-standardization of synthetic data"""
        for col, scaler in train_dataset.scalers.items():
            out[[col]] = scaler.inverse_transform(out[[col]])
    
        # post-process
        out[train_dataset.categorical_features] = out[train_dataset.categorical_features].astype(int)
        out[train_dataset.integer_features] = out[train_dataset.integer_features].round(0).astype(int)
        imputed.append((method, out))
        display(out.head())
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{method} : {elapsed_time:.4f} seconds")

    #%%
    """ReMasker""" # third-party
    start_train = time.time()
    
    remasker_module = importlib.import_module('remasker.remasker_impute')
    importlib.reload(remasker_module)
    method = "ReMasker"
    print(f"====={method}=====")
    imputer = remasker_module.ReMasker()
    
    X = pd.DataFrame(train_dataset.data, columns=train_dataset.features)
    X = pd.get_dummies(
        X, columns=train_dataset.categorical_features, prefix_sep="###"
    ).astype(float)
    out = imputer.fit_transform(X)
    out = pd.DataFrame(out, columns=X.columns)
    out = undummify(out)
    
    """un-standardization of synthetic data"""
    for col, scaler in train_dataset.scalers.items():
        out[[col]] = scaler.inverse_transform(out[[col]])
    
    # post-process
    out[train_dataset.categorical_features] = out[train_dataset.categorical_features].astype(int)
    out[train_dataset.integer_features] = out[train_dataset.integer_features].round(0).astype(int)
    imputed.append((method, out))
    display(out.head())
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{method} : {elapsed_time:.4f} seconds")
    
    #%%
    """KNN Imputer"""
    start_train = time.time()
    
    from sklearn.impute import KNNImputer
    method = "KNNI"
    print(f"====={method}=====")
    knnimputer = KNNImputer(n_neighbors=5)
    
    X = pd.DataFrame(train_dataset.data, columns=train_dataset.features)
    X = pd.get_dummies(
        X, columns=train_dataset.categorical_features, prefix_sep="###"
    ).astype(float)
    out = knnimputer.fit_transform(X)
    out = pd.DataFrame(out, columns=X.columns)
    out = undummify(out)
    
    """un-standardization of synthetic data"""
    for col, scaler in train_dataset.scalers.items():
        out[[col]] = scaler.inverse_transform(out[[col]])
    
    # post-process
    out[train_dataset.categorical_features] = out[train_dataset.categorical_features].astype(int)
    out[train_dataset.integer_features] = out[train_dataset.integer_features].round(0).astype(int)
    imputed.append((method, out))
    display(out.head())
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{method} : {elapsed_time:.4f} seconds")
    #%%
    """imputed dataset save"""
    base_name = f"baseline_{config['missing_rate']}_{config['missing_type']}_{config['dataset']}"
    model_dir = f"./assets/models/{base_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    artifact = wandb.Artifact(
        base_name, 
        type='dataset',
        metadata=config) 
    for method, data in imputed:
        data.to_csv(f"{model_dir}/{method}_{config['seed']}.csv", index=None)
        artifact.add_file(f"{model_dir}/{method}_{config['seed']}.csv")
    artifact.add_file('./imputer.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%