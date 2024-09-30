#%%
import os
import ast
import importlib
import argparse

import numpy as np

import torch
import torch.optim 
from torch.utils.data import DataLoader, RandomSampler
from modules.utils import set_random_seed
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

project = "ReMasker" # put your WANDB project name
# entity = "wotjd1410" # put your WANDB username

run = wandb.init(
    project=project,
    # entity=entity,
    tags=['train'], # put tags of this python project
)
#%%
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v
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
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='loan', 
                        help="""
                        Dataset options: 
                        abalone, anuran, banknote, breast, concrete,
                        kings, letter, loan, redwine, whitewine
                        """)
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split") 
    
    parser.add_argument("--missing_type", default="MCAR", type=str,
                        help="how to generate missing: MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument("--missing_rate", default=0.3, type=float,
                        help="missing rate") 
    parser.add_argument('--pin_mem', action='store_false')
    
    # training
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--max_epochs', default=600, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--mask_ratio', default=0.5, type=float, 
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--embed_dim', default=64, type=int, 
                         help='embedding dimensions')
    parser.add_argument('--depth', default=8, type=int, 
                        help='encoder depth')
    parser.add_argument('--decoder_depth', default=4, type=int, 
                        help='decoder depth')
    parser.add_argument('--num_heads', default=4, type=int, 
                        help='number of heads')
    parser.add_argument('--mlp_ratio', default=4., type=float, 
                        help='mlp ratio')
    parser.add_argument('--encode_func', default='linear', type=str, 
                        help='encoding function')

    parser.add_argument('--norm_field_loss', default=False,
                        help='Use (per-patch) normalized field as targets for computing loss')
    parser.set_defaults(norm_field_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, 
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', 
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', 
                        help='epochs to warmup LR')
    
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
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    
    train_dataset = CustomDataset(config, train=True)
    train_data = torch.nan_to_num(torch.FloatTensor(train_dataset.data.values))
    wandb.config.update(config, allow_val_change=True)
    #%%
    mask = train_dataset.mask.astype(bool) # 1:missing
    mask = ~mask # 0: missing
    mask = mask.astype(float)

    (n, p) = train_dataset.data.shape
    assert np.isnan(train_dataset.data.values).sum() == n*p - mask.sum() 
    #%%
    importlib.reload(dataset_module)
    MAEDataset = dataset_module.MAEDataset
    dataset = MAEDataset(train_data, mask)
    train_dataloader = DataLoader(
        dataset, 
        sampler=RandomSampler(dataset),
        batch_size=config['batch_size'],
    )
    #%%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    model = model_module.ReMasker(config, train_dataset.EncodedInfo).to(device)

    model.train()
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params/1000000:.2f}M")
    wandb.log({"Number of Parameters": num_params/1000000})
    #%%
    """train"""
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["blr"], betas=(0.9, 0.95))
    
    train_module.train_function(
        config, 
        model,
        optimizer, 
        train_dataloader,
        device
    )
    # %%
    """model save"""
    base_name = f"ReMasker_{config['dataset']}_{config['missing_type']}_{config['missing_rate']}"
    model_dir = f"./assets/models/{base_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"{base_name}_{config['seed']}"
    torch.save(model.state_dict(), f"./{model_dir}/{model_name}.pth") 
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config) 
    artifact.add_file(f"./{model_dir}/{model_name}.pth")
    artifact.add_file('./main.py')
    artifact.add_file('./modules/model.py')
    artifact.add_file('./modules/train.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    # %%
if __name__ == "__main__":
    main()
# %%