#%%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from tqdm import tqdm

from modules.missing import generate_mask
from datasets.raw_data import load_raw_data

EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['num_features', 'num_continuous_features', 'num_categories', 'mean', 'std']
)
#%%
"""
Data Source: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
"""

#%%
class CustomDataset(Dataset):
    def __init__(self, config):
        
        self.config = config
        data, continuous_features, categorical_features, integer_features = load_raw_data(config["dataset"])
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        
        self.features = self.continuous_features + self.categorical_features
        self.col_2_idx = {col : i + 1 for i, col in enumerate(data[self.features].columns.to_list())}
        self.num_continuous_features = len(self.continuous_features)
        
        # encoding categorical dataset
        data[self.categorical_features] = data[self.categorical_features].apply(
            lambda col: col.astype('category').cat.codes + 1)
        self.num_categories = data[self.categorical_features].nunique(axis=0).to_list()

        data = data[self.features] # select features for training
        data = data.reset_index(drop=True)
        self.raw_data = data[self.features]

        # Missing 처리
        if config["missing_type"] != "None":
            mask = generate_mask(
                torch.from_numpy(data.values).float(), 
                config["missing_rate"], 
                config["missing_type"],
                seed=config["seed"])
            data.mask(mask.astype(bool), np.nan, inplace=True)
        
        # standardizaton
        continuous_data, mean, std = self.standardize(data)
        data = pd.concat(
            [continuous_data, data[self.categorical_features]], axis=1
        )
        
        self.data = data.values
        self.mask = None if config["missing_type"] == "None" else mask.astype(bool)
        
        self.EncodedInfo = EncodedInfo(
            len(self.features), len(self.continuous_features), self.num_categories, mean, std
        )
    
    def standardize(self, data):
        """standardization for continuous features"""
        continuous_data =  data[self.continuous_features]
        mean = continuous_data.mean(axis=0)
        std = continuous_data.std(axis=0)
    
        continuous_data = (continuous_data - mean) / std # scaled
        
        return continuous_data, mean, std

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
#%%
class CustomMask(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
