#%%
from argparse import ArgumentParser
from copy import deepcopy
from importlib import import_module
from math import ceil
from os.path import exists, join
from sys import stderr
import sys
import importlib
import random 

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# from VAEAC.no_datasets import compute_normalization
from modules.imputation_networks import get_imputation_networks
from modules.train_utils import extend_batch, get_validation_iwae
from modules.VAEAC import VAEAC
import wandb
import warnings
warnings.filterwarnings('ignore')
#%%
# import subprocess
# try:
#     import wandb
# except:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
#     with open("wandb_api.txt", "r") as f:
#         key = f.readlines()
#         wandb.login(key=[key[0]])
#     subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
#     import wandb

# run = wandb.init(
#     project="VAEAC", # put your WANDB project name
#     # entity="", # put your WANDB username
#     tags=['Train'], # put tags of this python project
# )


#%%
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 대한 시드 고정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # NumPy 시드 고정
    np.random.seed(seed)
    random.seed(seed) 

def compute_normalization(data, one_hot_max_sizes):
    """
    Compute the normalization parameters (i. e. mean to subtract and std
    to divide by) for each feature of the dataset.
    For categorical features mean is zero and std is one.
    i-th feature is denoted to be categorical if one_hot_max_sizes[i] >= 2.
    Returns two vectors: means and stds.
    """
    norm_vector_mean = torch.zeros(len(one_hot_max_sizes))
    norm_vector_std = torch.ones(len(one_hot_max_sizes))
    for i, size in enumerate(one_hot_max_sizes):
        if size >= 2:
            continue
        v = data[:, i]
        v = v[~torch.isnan(v)]
        vmin, vmax = v.min(), v.max()
        vmean = v.mean()
        vstd = v.std()
        norm_vector_mean[i] = vmean
        norm_vector_std[i] = vstd
    return norm_vector_mean, norm_vector_std 
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


parser = ArgumentParser(description='Missing Features Multiple Imputation.')


parser.add_argument('--dataset', type=str, default='abalone', 
                    help="""
                    Dataset options: covtype, loan, kings, banknote, concrete, 
                    redwine, whitewine, yeast, breast, spam, letter, abalone
                    """)

parser.add_argument("--seed", default=0, type=int,required=True,
                    help="selcet version number ") 

parser.add_argument("--missing_type", default="MCAR", type=str,
                    help="how to generate missing: None(complete data), MCAR, MAR, MNARL, MNARQ") 

parser.add_argument("--missing_rate", default=0.3, type=float,
                    help="missing rate") 

parser.add_argument('--epochs', type=int, required=True,
                    help='Number epochs to train VAEAC.')

parser.add_argument('--validation_ratio', type=float,
                    choices=[ArgParseRange(0, 1)], required=True,
                    help='The proportion of objects ' +
                         'to include in the validation set.')

parser.add_argument('--validation_iwae_num_samples', type=int, action='store',
                    default=25,
                    help='Number of samples per object to estimate IWAE ' +
                         'on the validation set. Default: 25.')

parser.add_argument('--validations_per_epoch', type=int, action='store',
                    default=1,
                    help='Number of IWAE estimations on the validation set ' +
                         'per one epoch on the training set. Default: 1.')

parser.add_argument('--use_last_checkpoint', action='store_true',
                    default=False,
                    help='By default the model with the best ' +
                         'validation IWAE is used to generate ' +
                         'imputations. This flag forces the last model ' +
                         'to be used.')
#%%

def main():
    
    args = parser.parse_args()
    config = vars(args)
    ### ==============For Debug======================
    #%%
    # wandb.config.update(config, allow_val_change=True)

    model_dir = f"./assets/models/{config['dataset']}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_name = f"VAEAC_{config['dataset']}_{config['missing_type']}_{config['seed']}"

    with open(f"./{model_dir}/config_{config['seed']}.json", "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)    

    base_name = f"{config['missing_type']}_{config['dataset']}"
    if config["missing_type"] != "None":
        base_name = f"{config['missing_rate']}_" + base_name

    model_dir = f"./assets/models/{base_name}/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    ### model order has config 
    model_name = f"VAEAC_{base_name}_{config['seed']}"



    set_random_seed(config['seed'])

    ## =============== 데이터셋 ============================
    dataset_module = importlib.import_module('datasets.imputation')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    #from datasets.imputation import CustomDataset
    """dataset"""
    train_dataset = CustomDataset(config,config['seed'])

    total_column = train_dataset.continuous_features + train_dataset.categorical_features
    one_hot_max_sizes = [] # Decide that which features are continuous or categorical by nunique()

    for i,column in enumerate(total_column):
        if column in train_dataset.continuous_features:
            one_hot_max_sizes.append(1)
        elif column in train_dataset.categorical_features:
            one_hot_max_sizes.append(train_dataset.raw_data[column].nunique())

    raw_data = np.array(train_dataset.data)
    raw_data = torch.from_numpy(raw_data).float()
    norm_mean, norm_std = compute_normalization(raw_data, one_hot_max_sizes)
    norm_std = torch.max(norm_std, torch.tensor(1e-9))
    data = (raw_data - norm_mean[None]) / norm_std[None]
    use_cuda = torch.cuda.is_available()
    verbose = False
    num_workers = 0

    # Non-zero number of workers cause nasty warnings because of some bug in
    # multiprocess library. It might be fixed now, but anyway there is no need
    # to have a lot of workers for dataloader over in-memory tabular data.

    ## 모델링
    # design all necessary networks and learning parameters for the dataset
    ## feature에 따라 각기 다른 파라미터 설정
    networks = get_imputation_networks(one_hot_max_sizes)
    # build VAEAC on top of returned network, optimizer on top of VAEAC,
    # extract optimization parameters and mask generator
    model = VAEAC(
        networks['reconstruction_log_prob'],
        networks['proposal_network'],
        networks['prior_network'],
        networks['generative_network']
    )

    if use_cuda:
        model = model.cuda()
    optimizer = networks['optimizer'](model.parameters())
    batch_size = networks['batch_size']
    mask_generator = networks['mask_generator']
    vlb_scale_factor = networks.get('vlb_scale_factor', 1)


    # train-validation split
    val_size = ceil(len(data) * args.validation_ratio)
    val_indices = np.random.choice(len(data), val_size, False)
    val_indices_set = set(val_indices)
    train_indices = [i for i in range(len(data)) if i not in val_indices_set]
    train_data = data[train_indices]
    val_data = data[val_indices]


    # initialize dataloaders
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=False)
    # number of batches after which it is time to do validation#
    validation_batches = ceil(len(dataloader) / args.validations_per_epoch)
    # a list of validation IWAE estimates
    validation_iwae = []

    # a list of running variational lower bounds on the train set
    train_vlb = []

    # best model state according to the validation IWAE
    best_state = None


    ## ================== Train ====================
    for epoch in range(args.epochs):

        iterator = dataloader
        avg_vlb = 0
        if verbose:
            print('Epoch %d...' % (epoch + 1), file=stderr, flush=True)
            iterator = tqdm(iterator)

        # one epoch
        for i, batch in enumerate(iterator):

            if any([
                        i == 0 and epoch == 0,
                        i % validation_batches == validation_batches - 1,
                        i + 1 == len(dataloader)
                    ]):
                val_iwae = get_validation_iwae(val_dataloader, mask_generator,
                                            batch_size, model,
                                            args.validation_iwae_num_samples,
                                            verbose)
                validation_iwae.append(val_iwae)
                train_vlb.append(avg_vlb)
                # wandb.log({'Avg_vlb': avg_vlb, 'Val_iwae':val_iwae})

                if max(validation_iwae[::-1]) <= val_iwae:
                    best_state = deepcopy({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'validation_iwae': validation_iwae,
                        'train_vlb': train_vlb,
                    })

                if verbose:
                    print(file=stderr)
                    print(file=stderr)


            batch = extend_batch(batch, dataloader, batch_size)

            # generate mask and do an optimizer step over the mask and the batch
            mask = mask_generator(batch)
            optimizer.zero_grad()
            if use_cuda:
                batch = batch.cuda()
                mask = mask.cuda()
            vlb = model.batch_vlb(batch, mask).mean()
            (-vlb / vlb_scale_factor).backward()
            optimizer.step()


            # update running variational lower bound average
            avg_vlb += (float(vlb) - avg_vlb) / (i + 1)

            # if verbose:
            #     iterator.set_description('Train VLB: %g' % avg_vlb)

            
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params/1000000}M")
    # wandb.log({"Number of Parameters": num_params/1000000})


    if not args.use_last_checkpoint:
        model.load_state_dict(best_state['model_state_dict'])
        torch.save(model.state_dict(), f"./{model_dir}/{model_name}.pth")
        print("모델 상태가 'model_checkpoint.pth'로 저장되었습니다.")


if __name__ == "__main__":
    main()