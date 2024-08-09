#%%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from tqdm import tqdm

from modules.missing import generate_mask
#%%
from datasets.raw_data import load_raw_data
#%%
EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['num_features', 'num_continuous_features', 'num_categories'])

#%%
class CustomDataset(Dataset):
    def __init__(self, config):
        
        self.config = config
        data, continuous_features, categorical_features, integer_features = load_raw_data(config["dataset"])
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.seed = self.config['seed']
        self.features = self.continuous_features + self.categorical_features
        self.col_2_idx = {col : i + 1 for i, col in enumerate(data[self.features].columns.to_list())}
        self.num_continuous_features = len(self.continuous_features)
        
        # 범주형 데이터 인코딩
        data[self.categorical_features] = data[self.categorical_features].apply(
            lambda col: col.astype('category').cat.codes)
        self.num_categories = data[self.categorical_features].nunique(axis=0).to_list()

        # 필요한 컬럼만 정렬
        data = data[self.features] # select features for training
        data = data.reset_index(drop=True)
        self.raw_data = data[self.features]
        
        # Missing 처리
        if config["missing_type"] != "None":
            mask = generate_mask(
                torch.from_numpy(data.values).float(), 
                config["missing_rate"], 
                config["missing_type"], seed=self.seed)
            data.mask(mask.astype(bool), np.nan, inplace=True)

        self.data = data
        self.EncodedInfo = EncodedInfo(
            len(self.features), self.num_continuous_features, self.num_categories)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
#%%
