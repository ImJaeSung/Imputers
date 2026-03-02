#%%
import os
import argparse
import importlib
import time

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

project = "GRAPE" # put your WANDB project name
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

    parser.add_argument("--model", default='GRAPE', type=str) 
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument("--seed", default=0, type=int,
                        help="seed for repeatable results") 
    parser.add_argument('--dataset', type=str, default='concrete', 
                        help="""
                        Dataset options: 
                        loan, kings, banknote, concrete, redwine, 
                        whitewine, breast, letter, abalone, anuran,
                        spam, diabetes, dna, ncbirths
                        """)

    # parser.add_argument('--train_edge', type=float, default=0.7) # 1 - missing rate
    parser.add_argument('--split_sample', type=float, default=0.)
    parser.add_argument('--split_by', type=str, default='y') # 'y', 'random'
    parser.add_argument('--split_train', action='store_true', default=False)
    parser.add_argument('--split_test', action='store_true', default=False)
    # parser.add_argument('--train_y', type=float, default=0.7)
    parser.add_argument('--node_mode', type=int, default=0) # 0: feature onehot, sample all 1; 1: all onehot

    parser.add_argument("--missing_type", default="MAR", type=str,
                        help="how to generate missing: MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument("--missing_rate", default=0.3, type=float,
                        help="missing rate")
    
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight; 1: as input to mlp
    
    parser.add_argument('--impute_hiddens', type=str, default='64')
     
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split") 
    # parser.add_argument('--batch_size', default=256, type=int,
    #                     help='batch size')  
    parser.add_argument('--epochs', default=20000, type=int,
                        help='Number epochs to train GRAPE.')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate to train GRAPE')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--loss_mode', type=int, default=0) # 0: loss on all train edge, 1: loss only on unknown train edge
    
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    
    parser.add_argument('--valid', type=float, default=0.)
    parser.add_argument('--known', type=float, default=0.7) # 1 - edge dropout rate
    parser.add_argument('--auto_known', action='store_true', default=False)
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--impute_activation', type=str, default='relu')
    parser.add_argument('--post_hiddens', type=str, default=None,) # default to be 1 hidden of node_dim
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None,) # default to be all true
    parser.add_argument('--aggr', type=str, default='mean',)
    
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=True)) # default configuration
    set_random_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is', device)
    wandb.config.update(config)

    assert config["missing_type"] != None
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    data, train_data, test_data = dataset_module.load_data(config)
    #%%
    """model"""
    gnn_module = importlib.import_module('modules.gnn_model')
    importlib.reload(gnn_module)    
    gnn = gnn_module.get_gnn(data, config).to(device)
    gnn.train()
    #%%
    impute_module = importlib.import_module('modules.prediction_model')
    importlib.reload(impute_module)   
    impute_model = impute_module.MLPNet(gnn,config, data).to(device)
    #%%
    optimizer = optim.Adam(
        filter(lambda p : p.requires_grad,list(gnn.parameters())+ list(impute_model.parameters())), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    scheduler = None
    #%%
    """number of model parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_gnn = count_parameters(gnn)
    num_params_impute = count_parameters(impute_model)
    print(f"Number of Parameters: {num_params_gnn/1000:.1f}K")
    print(f"Number of Parameters: {num_params_impute/1000:.1f}K")
    #%%
    """train"""
    start_time = time.time()
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    train_module.train_function(
        gnn, impute_model, config, data, optimizer, scheduler, 'tmp/', device
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"GRAPE (train): {elapsed_time:.4f} seconds")
    #%%
    """model save"""
    base_name = f"{config['model']}_{config['missing_type']}_{config['missing_rate']}_{config['dataset']}"
    model_dir = f"assets/models/{base_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"GRAPE_{base_name}_{config['seed']}"
    #%%
    torch.save(gnn.state_dict(), f"./{model_dir}/{model_name}_GNN.pth")
    torch.save(impute_model.state_dict(), f"./{model_dir}/{model_name}_Impute.pth")
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config
    ) 
    artifact.add_file(f"./{model_dir}/{model_name}_GNN.pth")
    artifact.add_file(f"./{model_dir}/{model_name}_Impute.pth")
    artifact.add_file('./main.py')
    artifact.add_file('./modules/gnn_model.py')
    artifact.add_file('./modules/prediction_model.py')
    artifact.add_file('./modules/train.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == "__main__":
    main()
# %%
