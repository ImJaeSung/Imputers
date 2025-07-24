#%%
import os
import torch
import argparse
import importlib

import torch
from torch.utils.data import DataLoader

from evaluation import evaluation
from evaluation import evaluation_multiple
from modules.utils import set_random_seed
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

project = "dimvae_baselines2" # put your WANDB project name
# entity = "wotjd1410" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["imputation"], # put tags of this python project
)
# %%
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
def get_args(debug=False):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--ver', type=int, default=0, 
                        help='model version number')
    parser.add_argument("--model", default="gain", type=str)
    
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
    parser.add_argument('--hint_rate', default=0.9, type=float,
                         help='hint probability')

    parser.add_argument("--M", default=100, type=int,
                        help="the number of multiple imputation")

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
    
#%%
def main():
    #%%
    config = vars(get_args(debug=False))
    #%%
    """model load"""
    base_name = f"{config['model']}_{config['missing_type']}_{config['missing_rate']}_{config['hint_rate']}_{config['dataset']}"
    model_name = f"GAIN_{base_name}"
    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    #%%
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(config["seed"])
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
    test_dataset = CustomDataset(
        config,
        train=False,
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
    G = model_module.Generator(config).to(device)
    
    if config["cuda"]:
        G.load_state_dict(
            torch.load(
                model_dir + "/" + model_name
            )
        )
    else:
        G.load_state_dict(
            torch.load(
                model_dir + "/" + model_name,
                map_location=torch.device("cpu"),
            )
        )
    G.eval()
    #%%
    """number of model parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    G_params = count_parameters(G)
    print(f"Number of Parameters: {G_params/1000:.1f}M")
    wandb.log({"Number of Parameters": G_params / 1000})
    #%%
    """imputation"""
    imputed = G.impute(train_dataset, config, device)
    results = evaluation.evaluate(imputed, train_dataset, test_dataset, config, device)
    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    #%%
    results = evaluation_multiple.evaluate(
        train_dataset, G, config, device, M=config["M"]
    )
    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%% 
if __name__ == "__main__":
    main()  
#%% 