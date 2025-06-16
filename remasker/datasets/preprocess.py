#%%
from tqdm import tqdm
import copy

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import namedtuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from modules.missing import generate_mask
#%%
from datasets.raw_data import load_raw_data
#%%
EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['Encoded_dim', 'num_features', 'num_continuous_features', 'num_categories'])

#%%
class CustomDataset(Dataset):
    def __init__(
        self, 
        config,
        scalers=None,
        train=True):
        #%%
        self.config = config
        self.train = train
        data, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(config["dataset"])

        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.ClfTarget = ClfTarget
        
        self.features = continuous_features + categorical_features
        self.num_continuous_features = len(continuous_features)
        #%%
        # category to number
        data[categorical_features] = data[categorical_features].apply(
            lambda col: col.astype('category').cat.codes)
        self.num_categories = data[categorical_features].nunique(axis=0).to_list()
        
        data = data[self.features] # select features for training
    
        train_data, test_data = train_test_split(
            data, test_size=config["test_size"], random_state=config["seed"]
        )
        data = train_data if train else test_data
        data = data.reset_index(drop=True)

        # ground truth data for test
        self.raw_data = copy.deepcopy(data)
        #%%  
        # generating missing value
        if train:
            if config["missing_type"] != "None":
                self.mask = generate_mask(
                    torch.from_numpy(data.values).float(), 
                    config["missing_rate"], 
                    config["missing_type"],
                    seed=config["seed"])
                data.mask(self.mask.astype(bool), np.nan, inplace=True)        
        #%%
        self.scalers = {} if train else scalers
        cont_transformed = []
        for continuous_feature in tqdm(continuous_features, desc="Tranform Continuous Features..."):
            cont_transformed.append(self.transform_continuous(data, continuous_feature))
            
        cont_transformed = pd.DataFrame(
            np.hstack(cont_transformed), columns=continuous_features
        )
        
        self.data = pd.concat(
            [cont_transformed, data[categorical_features]], axis=1
        )

        Encoded_dim = self.data.shape[1]
        self.EncodedInfo = EncodedInfo(
            Encoded_dim, len(self.features), len(self.continuous_features), self.num_categories
        )

    def transform_continuous(self, data, col):
        nan_value = data[[col]].to_numpy().astype(float)
        nan_mask = np.isnan(nan_value)
        feature = nan_value[~nan_mask].reshape(-1, 1)
        
        if self.train:
            scaler = MinMaxScaler().fit(feature)
            self.scalers[col] = scaler
        else:
            scaler = self.scalers[col]
            
        nan_value[~nan_mask] = scaler.transform(feature)[:, 0]
        return nan_value
        #%%    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self):
        return torch.FloatTensor(self.data.values)
#%%
class MAEDataset(Dataset):

    def __init__(self, X, M):        
         self.X = X
         self.M = M

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.M[idx]