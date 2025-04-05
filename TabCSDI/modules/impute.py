"""Reference:
[1]https://github.com/pfnet-research/TabCSDI/blob/main/src/utils_table.py
"""
##%
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch

#%%
def impute(
    model,
    train_dataset,
    train_dataloader,
    mask_loader
    ):  
    
    data = []
    with tqdm(zip(train_dataloader, mask_loader), 
            total=min(len(train_dataloader), len(mask_loader)), 
            mininterval=5.0, 
            maxinterval=50.0, desc="imputation...") as pbar:
    
        for train_batch, mask_batch in pbar:
            (
                observed_data,
                observed_mask,
                observed_tp,
                gt_mask,
                _,
                cut_length,
            ) = model.process_data(train_batch, mask_batch)
            
            with torch.no_grad():
                cond_mask = gt_mask
                target_mask = observed_mask - cond_mask
                side_info = model.get_side_info(observed_tp, cond_mask)
                
                samples = model.impute(observed_data, cond_mask, side_info, n_samples=100)  
                
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)

                # take the median from samples.
                imputed_ = samples.median(dim=1).values.squeeze(2)
                data.append(imputed_)
                
    data = torch.cat(data, dim=0)
    data = pd.DataFrame(data.cpu().numpy(), columns=train_dataset.features)

    for col, scaler in train_dataset.scalers.items():
        data[[col]] = scaler.inverse_transform(data[[col]]).astype(np.float32)
    
    
    data[train_dataset.categorical_features] = data[train_dataset.categorical_features].astype(int)
    
    for idx, num_category in enumerate(train_dataset.num_categories):
        col = train_dataset.categorical_features[idx]
        data[col] = data[col].round(0).astype(int).clip(lower=0, upper=num_category)
        
    
    imputed = pd.DataFrame(
        data.values * train_dataset.mask + train_dataset.raw_data.values * (1. - train_dataset.mask),
        columns=train_dataset.raw_data.columns)

    return imputed
# %%
