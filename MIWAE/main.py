#%%
import os
import importlib
import argparse
import ast

import numpy as np

import torch
import torch.optim 
from torch.utils.data import DataLoader

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

project = "dimvae_baselines"  # put your WANDB project name
# entity = "wotjd1410" # put your WANDB username

run = wandb.init(
    project=project,
    # entity=entity, |
    tags=["train"], # put tags of this python project
)
#%%
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v
#%%
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument("--model", type=str, default="MIWAE")
    parser.add_argument("--seed", type=int, default=2, 
                        help="seed for repeatable results")
    parser.add_argument('--dataset', type=str, default='concrete', 
                        help="""
                        Dataset options: 
                        abalone, anuran, banknote, breast, concrete,
                        kings, letter, loan, redwine, whitewine
                        """)
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split") 
    
    parser.add_argument("--missing_type", default="MAR", type=str,
                        help="how to generate missing: MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument("--missing_rate", default=0.9, type=float,
                        help="missing rate") 

    parser.add_argument('--hidden_dim', type=int, nargs='+', default=64,
                        help='List of hidden dimensions sizes.')
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=2002,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    
    parser.add_argument('--k', type=int, default=10,
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    train_dataset = CustomDataset(config)

    train_dataset0 = train_dataset.data.copy()
    train_dataset0[np.isnan(train_dataset.data)] = 0 # zero imputation
    train_dataloader = DataLoader(
        train_dataset0, batch_size=config['batch_size']
    )

    config["input_dim"] = train_dataset.EncodedInfo.num_features
    config["latent_dim"] = 1
    #%%
    """missing mask"""
    mask = train_dataset.mask.astype(bool) # 1:missing
    mask = ~mask # 0:missing

    (n, p) = train_dataset.data.shape
    assert np.isnan(train_dataset.data).sum() == n*p - mask.sum() # checking missing
    
    CustomMask = dataset_module.CustomMask
    mask = CustomMask(mask)
    mask_loader = DataLoader(
        mask, batch_size=config["batch_size"]
    )
    #%%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    model = model_module.MIWAE(config, train_dataset.EncodedInfo, device).to(device)
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
    base_name = f"MIWAE_{config['missing_type']}_{config['missing_rate']}_{config['dataset']}"
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