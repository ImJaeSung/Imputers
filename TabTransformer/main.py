#%%
import os
import argparse
import importlib
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from modules.train import *
from modules.utils import set_random_seed
import wandb
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

project = "dimvae_baselines"
# entity = ""

run = wandb.init(
    project=project, # put your WANDB project name
    # entity=entity, # put your WANDB username
    tags=["train"], # put tags of this python project
)
#%%
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=0, 
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
    parser.add_argument('--epochs', default=10000, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.0005, type=float,
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
    set_random_seed(config['seed'])
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'cpu')
    print('Current device is', device)
    wandb.config.update(config)
    #%%
    """Data loader"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    train_dataset = CustomDataset(config, train=True)
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
    #%%
    """the number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params / 1000000:.1f}M")
    #%%
    """train"""
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay'])
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    train_module.train_function(
        model,
        config,
        optimizer,
        train_dataset,
        device)

#%%
    """model save"""
    model_name = "_".join([str(y) for x, y in config.items() if x != "seed"]) 
    if config["missing_type"] != "None":
        model_name = f"{config['missing_type']}_{config['missing_rate']}_" + model_name
    model_dir = f"./assets/models/{model_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), f"./{model_dir}/{model_name}_{config['seed']}.pth")
    artifact = wandb.Artifact(
        model_name, 
        type='model',
        metadata=config) 
    artifact.add_file(f"./{model_dir}/{model_name}_{config['seed']}.pth")
    artifact.add_file('./main.py')
    artifact.add_file('./modules/train.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
