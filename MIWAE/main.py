#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import json
#%%s
import importlib
import argparse
import ast

import numpy as np

import torch
import torch.optim 
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.simulation import set_random_seed
#%%
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "xxx"  # put your WANDB project name
entity = "xxx" # put your WANDB username

run = wandb.init(
    project=project,
    entity=entity, 
    tags=["Train"], # put tags of this python project
)
#%%
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for repeatable results"
    )
    parser.add_argument('--dataset', type=str, default='kings', 
                        help='Dataset options: abalone, banknote, breast, redwine, whitewine')
    parser.add_argument("--missing_type", default="MCAR", type=str,
                        help="how to generate missing: MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument("--missing_rate", default=0.3, type=float,
                        help="missing rate") 
    

    parser.add_argument('--hidden_dim', type=int, nargs='+', default=128,
                        help='List of hidden dimensions sizes.')
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    
    parser.add_argument('--k', type=int, default=20,
                        help='# number of IS during training.')
    
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
    dataset_module = importlib.import_module('datasets.imputation')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    train_dataset = CustomDataset(config)

    ### zero imputation function ###
    train_dataset0 = train_dataset.data.copy()
    train_dataset0[np.isnan(train_dataset.data)] = 0
    
    (n, p) = train_dataset.data.shape
    config["input_dim"] = p
    config["latent_dim"] = p-1
    ## mask
    mask = train_dataset.mask # 1:mask
    mask = ~mask

    # checking masking
    assert np.isnan(train_dataset.data).sum() == n*p - mask.sum() 
    
    CustomMask = dataset_module.CustomMask
    mask = CustomMask(mask)
    

    train_dataloader = DataLoader(
        train_dataset0, batch_size=config['batch_size']
    )
    mask_loader = DataLoader(
        mask, batch_size=config["batch_size"]
    )

    #%%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    model = model_module.MIWAE(config, train_dataset.EncodedInfo, device).to(device)
    
    print(model.train())
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params/1000000}M")
    wandb.log({"Number of Parameters": num_params/1000000})
    #%%
    """train"""
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config["epochs"]):
        logs = train_module.train_function(
            model, 
            optimizer, 
            train_dataloader,
            mask_loader,
            device
        )
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += "".join(
            [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
        )
        print(print_input)

        """update log"""
        wandb.log({x: np.mean(y) for x, y in logs.items()})
    # %%
    """model save"""
    model_dir = f"./assets/models/{config['dataset']}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"MIWAE_{config['dataset']}_{config['missing_type']}_{config['seed']}"

    torch.save(model.state_dict(), f"./{model_dir}/{model_name}.pth")

    with open(f"./{model_dir}/config_{config['seed']}.json", "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)    

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