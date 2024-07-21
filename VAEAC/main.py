#%%
import os
import argparse
import importlib

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from modules.utils import set_random_seed
#%%
import warnings
warnings.filterwarnings('ignore')
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

project = "VAEAC" # put your WANDB project name
# entity = "wotjd1410" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["train"], # put tags of this python project
)
#%%
class ArgParseRange:
    """
    List with this element restricts the argument to be
    in range [start, end].
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __repr__(self):
        return '{0}...{1}'.format(self.start, self.end)
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

    parser.add_argument('--dataset', type=str, default='loan', 
                        help="""
                        Dataset options: 
                        loan, kings, banknote, concrete, redwine, 
                        whitewine, breast, letter, abalone, anuran
                        """)

    parser.add_argument("--seed", default=0, type=int,
                        help="seed for repeatable results") 

    parser.add_argument("--missing_type", default="MCAR", type=str,
                        help="how to generate missing: MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument("--missing_rate", default=0.3, type=float,
                        help="missing rate")
     
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split") 
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')  
    parser.add_argument('--epochs', default=5, type=int,
                        help='Number epochs to train VAEAC.')
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='learning rate to train VAEAC')
    
    parser.add_argument('--validation_ratio', default=0.25, type=float,
                        choices=[ArgParseRange(0, 1)],
                        help='The proportion of objects ' +
                            'to include in the validation set.')

    parser.add_argument('--validation_iwae_num_samples', default=25, 
                        type=int, action='store', 
                        help='Number of samples per object to estimate IWAE ' +
                            'on the validation set. Default: 25.'
                        )

    parser.add_argument('--validations_per_epoch', default=1,
                        type=int, action='store',
                        help='Number of IWAE estimations on the validation set ' +
                            'per one epoch on the training set. Default: 1.'
                        )

    parser.add_argument('--use_last_checkpoint', action='store_true',
                        default=False,
                        help='By default the model with the best ' +
                            'validation IWAE is used to generate ' +
                            'imputations. This flag forces the last model ' +
                            'to be used.'
                        )
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

    # train-validation split
    train_data, valid_data = train_test_split(
        train_dataset.data, 
        test_size=config["validation_ratio"], 
        random_state=config["seed"]
    )

    train_dataloader = DataLoader(
        train_data, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=0, 
        drop_last=False
    )
    valid_dataloader = DataLoader(
        valid_data, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=0, 
        drop_last=False
    )
    #%%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    networks = model_module.get_imputation_networks(
        train_dataset.EncodedInfo.one_hot_max_sizes
    )

    model = model_module.VAEAC(
        config,
        networks,
        device
    ).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    #%%
    """number of model parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params/1000000:.1f}M")
    #%%
    """train"""
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    best_state = train_module.train_function(
        model,
        networks,
        config,
        optimizer,
        train_dataloader,
        valid_dataloader,
        device,
        verbose=True
    )
    #%%
    """model save"""
    base_name = f"{config['missing_type']}_{config['missing_rate']}_{config['dataset']}"
    model_dir = f"./assets/models/{base_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"VAEAC_{base_name}_{config['seed']}"

    torch.save(best_state['model_state_dict'], f"./{model_dir}/{model_name}.pth")
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config
    ) 
    artifact.add_file(f"./{model_dir}/{model_name}.pth")
    artifact.add_file('./main.py')
    artifact.add_file('./modules/model.py')
    artifact.add_file('./modules/train.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == "__main__":
    main()
# %%
