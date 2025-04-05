#%%
import os
import torch
import argparse
import importlib
import numpy as np

import torch.nn as nn

from modules import utils
from modules.utils import set_random_seed

import warnings
warnings.filterwarnings('ignore')
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

project = "kdd_rebuttal"
# entity = ""

run = wandb.init(
    project=project, # put your WANDB project name
    # entity=entity, # put your WANDB username
    tags=["inference"], # put tags of this python project
)
#%%
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--ver', type=int, default=0, 
                        help='seed for repeatable results')
    parser.add_argument('--model', type=str, default='TabTransformer')
    
    parser.add_argument('--dataset', type=str, default='abalone', 
                        help="""
                        Dataset options: 
                        abalone, banknote, breast, concrete, covtype,
                        kings, letter, loan, redwine, whitewine
                        """)
    parser.add_argument("--missing_type", default="MAR", type=str,
                        help="""
                        how to generate missing: None(complete data), MCAR, MAR, MNARL, MNARQ
                        """) 
    parser.add_argument("--missing_rate", default=0.3, type=float,
                        help="missing rate") 
    
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split")     
    parser.add_argument('--epochs', default=1000, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=0., type=float,
                        help='parameter of AdamW')
    
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    model_name = "_".join([str(y) for x, y in config.items() if x != "ver" and x != "tau"]) 
    if config["missing_type"] != "None":
        model_name = f"{config['missing_type']}_{config['missing_rate']}_" + model_name
    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'cpu')
    set_random_seed(config["seed"])
    wandb.config.update(config)
    #%%
    """Data loader"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    train_dataset = CustomDataset(config, train=True)
    test_dataset = CustomDataset(config, train=False, scalers=train_dataset.scalers)
    #%%
    """Model"""
    dim_out = train_dataset.num_continuous_features + sum(train_dataset.num_categories)
    categories = [x + 1 for x in train_dataset.num_categories]
    mean_std_list = []
    for _, scaler in train_dataset.scalers.items():
        mean_std_list.append([scaler.mean_[0], scaler.scale_[0]])
    cont_mean_std = torch.tensor(mean_std_list).float()

    from tab_transformer_pytorch import TabTransformer
    model = TabTransformer(
        categories=tuple(categories),      # tuple containing the number of unique values within each category
        num_continuous=train_dataset.num_continuous_features,                # number of continuous values
        dim=32,                           # dimension, paper set at 32
        dim_out=dim_out,                        # binary prediction, but could be anything
        depth=6,                          # depth, paper recommended 6
        heads=8,                          # heads, paper recommends 8
        attn_dropout=0.1,                 # post-attention dropout
        ff_dropout=0.1,                   # feed forward dropout
        mlp_hidden_mults=(4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
        mlp_act=nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        continuous_mean_std=cont_mean_std, # (optional) - normalize the continuous values before layer norm
    ).to(device)
    
    model.load_state_dict(
        torch.load(
            model_dir + "/" + model_name,
            map_location=device,
        )
    )
    
    model.eval()
    # %%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params / 1000000:.3f}M")
    wandb.log({"Number of Parameters": num_params / 1000000})
    # %%
    """imputation"""
    impute_module = importlib.import_module('modules.impute')
    importlib.reload(impute_module)
    imputed = impute_module.impute(model, config, train_dataset, device)

    assert imputed.isna().sum().sum() == 0 
    #%%
    """evaluation"""    
    evaluate_module = importlib.import_module('evaluation.evaluation')
    importlib.reload(evaluate_module)
    evaluate = evaluate_module.evaluate

    results = evaluate(imputed, train_dataset, test_dataset, config, device)
    for x, y in results._asdict().items():
        print(f"{x}: {y:.4f}")
        wandb.log({f"{x}": y})
    #%%
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == "__main__":
    main()
# %%
