#%%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from modules.missing import generate_mask
#%%
from datasets.raw_data import load_raw_data
#%%
EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['num_features', 'num_continuous_features', 'num_categories', 'one_hot_max_sizes'])
#%%
class CustomDataset(Dataset):
    def __init__(self, config, train=True):
        
        self.config = config
        data, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(config["dataset"])
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.ClfTarget = ClfTarget

        self.features = continuous_features + categorical_features
        self.col_2_idx = {col : i + 1 for i, col in enumerate(data[self.features].columns.to_list())}
        self.num_continuous_features = len(continuous_features)
        
        # categorical (discrete) encoding
        data[categorical_features] = data[categorical_features].apply(
            lambda col: col.astype('category').cat.codes)
        self.num_categories = data[categorical_features].nunique(axis=0).to_list()

        data = data[self.features] # select features for training
        
        train_data, test_data = train_test_split(
            data, test_size=config["test_size"], random_state=config["seed"]
        )
        
        data = train_data if train else test_data        
        data = data.reset_index(drop=True)
        self.raw_data = data[self.features]
        
        # Decide that which features are continuous or categorical by nunique()
        self.one_hot_max_sizes = [1]*self.num_continuous_features + self.num_categories
        
        # Genrating missing values
        if config["missing_type"] != "None":
            mask = generate_mask(
                torch.from_numpy(data.values).float(), 
                config["missing_rate"], 
                config["missing_type"], 
                seed=config["seed"]
            )
            data.mask(mask.astype(bool), np.nan, inplace=True)

        if train:
            self.norm_mean, self.norm_std = self.compute_normalization(
                data, 
                self.one_hot_max_sizes
            )
            data = self.transform(
                data, 
                self.norm_mean, 
                self.norm_std
            )
        
        self.data = data 
        self.mask = None if config["missing_type"] == "None" else mask.astype(bool)

        self.EncodedInfo = EncodedInfo(
            len(self.features), self.num_continuous_features, self.num_categories, self.one_hot_max_sizes
        )
    #%%
    def compute_normalization(self, data, one_hot_max_sizes):
        """
        Compute the normalization parameters (i. e. mean to subtract and std
        to divide by) for each feature of the dataset.
        For categorical features mean is zero and std is one.
        i-th feature is denoted to be categorical if one_hot_max_sizes[i] >= 2.
        Returns two vectors: means and stds.
        """
        data = torch.from_numpy(np.array(data)).float()
        
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
    def transform(self, data, norm_mean, norm_std):
        data = torch.from_numpy(np.array(data)).float()

        norm_std = torch.max(norm_std, torch.tensor(1e-9))
        
        data -= norm_mean[None] 
        data /= norm_std[None]

        return data
    #%%
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
#%%
