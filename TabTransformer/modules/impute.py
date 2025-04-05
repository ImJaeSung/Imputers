#%%
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

from torch.distributions.categorical import Categorical

#%%
def impute(
    model,
    config,
    train_dataset,
    device):
    
    C = train_dataset.num_continuous_features
    x_categ = torch.from_numpy(train_dataset.data[:, C:]).float()
    x_cont = torch.from_numpy(train_dataset.data[:, :C].astype(np.float32))
    batch_size = config["batch_size"]
    
    data = []
    for i in tqdm(range(0, x_categ.shape[0], config["batch_size"]), desc="imputation..."):
        st = i
        ed = st + batch_size
        
        x_cat = x_categ[st: ed].to(device)
        x_con = x_cont[st: ed].to(device)
        
        batch = torch.cat([x_con, x_cat], dim=1)
            
        nan_mask = batch.isnan()
        
        masked_batch = batch.clone()
        masked_batch[nan_mask] = 0. # [MASKED] token
        
        masked_x_con = masked_batch[:, :C].float()
        masked_x_cat = masked_batch[:, C:].int()
        with torch.no_grad():
            pred = model(masked_x_cat, masked_x_con)     

        imputed_con = pred[:, :C]
        st_ = 0
        imputed_cat_ = []
        for idx, j in enumerate(train_dataset.num_categories):
            pred_cat = pred[:, C:]
            
            imputed_cat_tmp = Categorical(logits=pred_cat[:, st_:st_+j]).sample().float() + 1
            st_ = j
            imputed_cat_.append(imputed_cat_tmp)

        imputed_cat = torch.stack((imputed_cat_), dim=1)
    
        data_ = torch.cat([imputed_con, imputed_cat], dim=1)
    
        data.append(data_)
    
    data = torch.cat(data, dim=0)
    data = pd.DataFrame(data.cpu().numpy(), columns=train_dataset.features)
        
    for col, scaler in train_dataset.scalers.items():
        data[[col]] = scaler.inverse_transform(data[[col]]).astype(np.float32)
        
    data[train_dataset.categorical_features] = data[train_dataset.categorical_features].astype(int)
    data[train_dataset.integer_features] = data[train_dataset.integer_features].round(0).astype(int)
    
    imputed = pd.DataFrame(
        data.values * train_dataset.mask + train_dataset.raw_data.values * (1. - train_dataset.mask),
        columns=train_dataset.raw_data.columns)

    return imputed
# %%
