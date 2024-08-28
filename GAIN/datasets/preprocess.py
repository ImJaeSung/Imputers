#%%
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import namedtuple

from sklearn.model_selection import train_test_split
from modules.missing import generate_mask
#%%
from datasets.raw_data import load_raw_data
#%%
EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['norm_parameters', 'num_features', 'num_continuous_features', 'num_categories'])

#%%
class CustomDataset(Dataset):
    def __init__(self, config, train=True):
        self.config = config
        data, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(config["dataset"])
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.features = continuous_features + categorical_features
        
        self.ClfTarget = ClfTarget
        self.num_continuous_features = len(continuous_features)
        
        # encoding for categorical variable
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
        
        # generate missing values
        if config["missing_type"] != "None":
            self.mask = generate_mask(
                torch.from_numpy(data.values).float(), 
                config)
            data.mask(self.mask.astype(bool), np.nan, inplace=True)

        norm_data, norm_parameters = self.normalization(data)

        self.data = norm_data
        self.norm_parameters = norm_parameters
        self.EncodedInfo = EncodedInfo(
            self.norm_parameters, len(self.features), self.num_continuous_features, self.num_categories)
    #%%
    def normalization(self, data, parameters=None):
        '''Normalize data in [0, 1] range.
        
        Args:
            - data: original data
        
        Returns:
            - norm_data: normalized data
            - norm_parameters: min_val, max_val for each feature for renormalization
        '''

        # Parameters
        _, dim = data.shape
        norm_data = np.array(data.copy())
        
        if parameters is None:
        
            # MixMax normalization
            min_val = np.zeros(dim)
            max_val = np.zeros(dim)
            
            # For each dimension
            for i in range(dim):
                min_val[i] = np.nanmin(norm_data[:,i])
                norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
                max_val[i] = np.nanmax(norm_data[:,i])
                norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
                
            # Return norm_parameters for renormalization
            norm_parameters = {'min_val': min_val,
                            'max_val': max_val}

        else:
            min_val = parameters['min_val']
            max_val = parameters['max_val']
            
            # For each dimension
            for i in range(dim):
                norm_data[:,i] = norm_data[:,i] - min_val[i]
                norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
                
            norm_parameters = parameters     
            
        return norm_data, norm_parameters
    #%%
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
#%%
