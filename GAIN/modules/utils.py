"""Reference:
[1] https://github.com/jsyoon0823/GAIN/blob/master/utils.py
"""
#%%
from tqdm import tqdm

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random
import os

#%%
"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 대한 시드 고정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # NumPy 시드 고정
    np.random.seed(seed)
    random.seed(seed)   

#%%
def marginal_plot(train_dataset, imputed, config):
    train = train_dataset.raw_data
    if not os.path.exists(f"./assets/figs/{config['dataset']}/seed{config['seed']}/"):
        os.makedirs(f"./assets/figs/{config['dataset']}/seed{config['seed']}/")
    
    figs = []
    for idx, feature in tqdm(enumerate(train.columns), desc="Plotting Histograms..."):
        fig = plt.figure(figsize=(7, 4))
        fig, ax = plt.subplots(1, 1)
        sns.histplot(
            data=imputed,
            x=imputed[feature],
            stat='density',
            label='synthetic',
            ax=ax,
            bins=int(np.sqrt(len(imputed)))) 
        sns.histplot(
            data=train,
            x=train[feature],
            stat='density',
            label='train',
            ax=ax,
            bins=int(np.sqrt(len(train)))) 
        ax.legend()
        ax.set_title(f'{feature}', fontsize=15)
        plt.tight_layout()
        plt.savefig(f"./assets/figs/{config['dataset']}/seed{config['seed']}/hist_{feature}.png")
        # plt.show()
        plt.close()
        figs.append(fig)
    return figs
#%%
def get_frequency(
    X_gt: pd.DataFrame, X_synth: pd.DataFrame, n_histogram_bins: int = 10
):
    """
    Reference:
    [1] https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics/_utils.py
    
    Get percentual frequencies for each possible real categorical value.

    Returns:
        The observed and expected frequencies (as a percent).
    """
    res = {}
    for col in X_gt.columns:
        local_bins = min(n_histogram_bins, len(X_gt[col].unique()))

        if len(X_gt[col].unique()) < 5:  # categorical
            gt = (X_gt[col].value_counts() / len(X_gt)).to_dict()
            synth = (X_synth[col].value_counts() / len(X_synth)).to_dict()
        else:
            gt_vals, bins = np.histogram(X_gt[col], bins=local_bins)
            synth_vals, _ = np.histogram(X_synth[col], bins=bins)
            gt = {k: v / (sum(gt_vals) + 1e-8) for k, v in zip(bins, gt_vals)}
            synth = {k: v / (sum(synth_vals) + 1e-8) for k, v in zip(bins, synth_vals)}

        for val in gt:
            if val not in synth or synth[val] == 0:
                synth[val] = 1e-11
        for val in synth:
            if val not in gt or gt[val] == 0:
                gt[val] = 1e-11

        if gt.keys() != synth.keys():
            raise ValueError(f"Invalid features. {gt.keys()}. syn = {synth.keys()}")
        res[col] = (list(gt.values()), list(synth.values()))

    return res

#%%
def sample_binary(m, n, p):
    '''Sample binary random variables.
  
    Args:
        - p: probability of 1
        - rows: the number of rows
        - cols: the number of columns
        
    Returns:
        - binary_random_matrix: generated binary random matrix.
    '''
    random_matrix = np.random.uniform(0., 1., size = [m, n])
    binary_matrix = random_matrix < p
    result = 1.*binary_matrix
    return torch.tensor(result)
#%%
def sample_uniform(m, n):
    '''Sample uniform random variables.
  
    Args:
        - low: low limit
        - high: high limit
        - rows: the number of rows
        - cols: the number of columns
        
    Returns:
        - uniform_random_matrix: generated uniform random matrix.
    '''
    z = np.random.uniform(0., 0.01, size = [m, n])  
    result = torch.tensor(z)
    return result
#%%
def discriminator_loss(D, G, M, New_X, H):
    G_sample = G(New_X, M)

    Hat_New_X = New_X * M + G_sample * (1 - M)
    D_prob = D(Hat_New_X, H)
    
    D_loss = -torch.mean(
        M * torch.log(D_prob + 1e-8) + (1 - M) * torch.log(1. - D_prob + 1e-8)
    )

    return D_loss
#%%
def generator_loss(D, G, X, M, New_X, H, alpha):
    G_sample = G(New_X, M)

    Hat_New_X = New_X * M + G_sample * (1 - M)
    D_prob = D(Hat_New_X, H)
    G_loss1 = -torch.mean((1 - M) * torch.log(D_prob + 1e-8))
    MSE_train_loss = torch.mean((M * New_X - M * G_sample) ** 2) / torch.mean(M)
    
    G_loss = G_loss1 + alpha * MSE_train_loss
    MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
    
    return G_loss, MSE_train_loss, MSE_test_loss
# %%

#%%
def renormalization(norm_data, norm_parameters):
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data
