#%%
import os
import argparse
import importlib
#%%
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
#%%
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

project = "dimvae_baselines2" # put your WANDB project name
# entity = "wotjd1410" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["train"], # put tags of this python project
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
    parser.add_argument("--model", default="gain", type=str)
    parser.add_argument("--seed", default=0, type=int,
                        help="seed for repeatable results") 
    
    parser.add_argument('--dataset', type=str, default='loan', 
                        help="""
                        Dataset options: 
                        loan, kings, banknote, concrete, redwine, 
                        whitewine, breast, letter, abalone, anuran
                        """)

    parser.add_argument("--missing_type", default="MAR", type=str,
                        help="how to generate missing: MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument("--missing_rate", default=0.3, type=float,
                        help="missing rate")
    
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split") 
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')  
    parser.add_argument('--epochs', default=5000, type=int,
                        help='Number epochs to train GAIN.')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate to train GAIN')
    
    parser.add_argument('--hint_rate', default=0.9, type=float,
                         help='hint probability')
    parser.add_argument('--alpha', default=100, type=float,
                        help='hyperparameter')
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

    assert config["missing_type"] != None
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    train_dataset = CustomDataset(
        config, 
        train=True
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"],
        drop_last=False 
    )
    config["dim"] = train_dataset.EncodedInfo.num_features
    #%%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    D = model_module.Discriminator(config).to(device)
    G = model_module.Generator(config).to(device)
    #%%
    D.train()
    G.train()

    optimizer_D = optim.Adam(D.parameters(), lr=config["lr"])
    optimizer_G = optim.Adam(G.parameters(), lr=config["lr"])
    #%%
    """number of model parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    D_params = count_parameters(D)
    G_params = count_parameters(G)
    print(f"Number of Parameters: {D_params/1000:.1f}K")
    print(f"Number of Parameters: {G_params/1000:.1f}K")
    #%%
    """train"""
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    train_module.train_function(
        D, 
        G, 
        config, 
        optimizer_D,
        optimizer_G,
        train_dataloader, 
        device
    )
    #%%
    """model save"""
    base_name = f"{config['model']}_{config['missing_type']}_{config['missing_rate']}_{config['hint_rate']}_{config['dataset']}"
    model_dir = f"./assets/models/{base_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"GAIN_{base_name}_{config['seed']}"

    torch.save(G.state_dict(), f"./{model_dir}/{model_name}.pth")
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
#%%
if __name__ == '__main__':
    main()
#%%