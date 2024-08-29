"""Reference:
[1] https://github.com/jsyoon0823/GAIN/blob/master/gain.py
"""
#%%
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from modules.utils import set_random_seed
from modules.utils import renormalization, sample_uniform
#%%
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        d = config["dim"]
        d_hidden = int(d)

        self.layer1 = nn.Linear(d*2, d_hidden) # [data, hint]
        self.layer2 = nn.Linear(d_hidden, d_hidden)
        self.layer3 = nn.Linear(d_hidden, d)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)

    def forward(self, new_x, h):
        inputs = torch.cat((new_x, h), dim=1)
        D_h1 = F.relu(self.layer1(inputs))
        D_h2 = F.relu(self.layer2(D_h1))
        D_logit = self.layer3(D_h2)
        D_prob = torch.sigmoid(D_logit)
        return D_prob
    
#%%
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        d = config["dim"]
        d_hidden = int(d)
        
        self.layer1 = nn.Linear(d*2, d_hidden)
        self.layer2 = nn.Linear(d_hidden, d_hidden)
        self.layer3 = nn.Linear(d_hidden, d)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)

    def forward(self, new_x, m):
        inputs = torch.cat((new_x, m), dim=1)
        G_h1 = F.relu(self.layer1(inputs))
        G_h2 = F.relu(self.layer2(G_h1))
        G_logit = self.layer3(G_h2)
        G_prob = torch.sigmoid(G_logit)
        return G_prob
    
    def impute(self, train_dataset, config, device):
        set_random_seed(config["seed"])

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"],
            drop_last=False 
        )

        imputed_data_ = []
        for batch in tqdm(train_dataloader, desc="imputation..."):
            with torch.no_grad():
                X = batch.float()
                get_batch = X.shape[0]
                dim = X.shape[1]

                M = 1 - torch.isnan(X).float().to(device) #1: observed, 0: nan
                
                X = torch.nan_to_num(X,nan=0.0).to(device)     
                Z = sample_uniform(get_batch, dim).float().to(device)
                New_X = M * X + (1-M) * Z  
                _, Sample = self.test_loss(X, M, New_X)

                imputed_data_batch = M * X + (1-M) * Sample
                imputed_data_.append(imputed_data_batch)
                    
            imputed_data = torch.cat(imputed_data_, dim=0)

        """post-processing"""
        imputed_data = renormalization(
            np.array(imputed_data.cpu().detach()), 
            train_dataset.EncodedInfo.norm_parameters
        )

        imputed_data = pd.DataFrame(imputed_data, columns=train_dataset.features)
        imputed_data[train_dataset.categorical_features] = imputed_data[train_dataset.categorical_features].astype(int)
        imputed_data[train_dataset.integer_features] = imputed_data[train_dataset.integer_features].round(0).astype(int)

        # evaluation
        data = imputed_data*train_dataset.mask + train_dataset.raw_data*(1. - train_dataset.mask)
        return data
    #%%
    def test_loss(self, X, M, New_X):
        # Generator
        G_sample = self(New_X, M)

        MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
        return MSE_test_loss, G_sample
# %%
