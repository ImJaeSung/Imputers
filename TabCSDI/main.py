#%%
import os
import argparse
import importlib
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
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

project = "kdd_rebuttal"
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
    parser.add_argument('--model', type=str, default='TabCSDI')
    parser.add_argument('--dataset', type=str, default='breast', 
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
    parser.add_argument('--batch_size', default=512, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='learning rate')
    
    parser.add_argument('--layers', default=4, type=int,
                    help='Number of layers')
    parser.add_argument('--channels', default=64, type=int,
                        help='Number of channels')
    parser.add_argument('--nheads', default=2, type=int,
                        help='Number of attention heads')
    
    parser.add_argument('--diffusion_embedding_dim', default=128, type=int,
                        help='Dimension of diffusion embedding')
    parser.add_argument('--beta_start', default=0.0001, type=float,
                        help='Starting beta value')
    parser.add_argument('--beta_end', default=0.5, type=float,
                        help='Ending beta value')
    parser.add_argument('--num_steps', default=150, type=int,
                        help='Number of diffusion steps')
    parser.add_argument('--schedule', default="quad", type=str,
                        help='Type of schedule to use')
    
    parser.add_argument('--mixed', default=False, type=bool,
                        help='Use mixed mode if True')
    parser.add_argument('--is_unconditional', default=0, type=int,
                        help='Flag for unconditional mode (0 or 1)')
    
    parser.add_argument('--timeemb', default=32, type=int,
                        help='Time embedding dimension')
    parser.add_argument('--featureemb', default=16, type=int,
                        help='Feature embedding dimension')
    parser.add_argument('--target_strategy', default="random", type=str,
                        help='Target strategy to employ')
    
    parser.add_argument('--weight_decay', default=1e-6, type=float,
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
    
    fixed_indices = list(range(len(train_dataset)))  # 또는 원하는 순서대로 섞은 인덱스
    sampler = dataset_module.FixedSampler(fixed_indices)
        
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler)
    mask_loader = DataLoader(train_dataset.mask, batch_size=config['batch_size'], sampler=sampler)
    #%%
    """Model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    model = model_module.TabCSDI(config, train_dataset.EncodedInfo, device).to(device)
    model.train()
    #%%
    """the number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params / 1000000:.1f}M")
    wandb.log({"Number of Parameters": num_params / 1000000})
    #%%
    """train"""
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[
            int(0.25 * config["epochs"]), 
            int(0.5 * config["epochs"]), 
            int(0.75 * config["epochs"]), 
            int(0.9 * config["epochs"])], 
        gamma=0.1
    )
    
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    train_module.train_function(
        model,
        config,
        optimizer,
        scheduler,
        train_dataloader,
        mask_loader)
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
