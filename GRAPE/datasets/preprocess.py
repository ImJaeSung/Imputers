"""
Reference:
[1] https://github.com/maxiaoba/GRAPE/blob/master/uci/uci_data.py
[2] https://github.com/G-AILab/IGRM/blob/main/uci/uci_data.py
"""
#%%
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from collections import namedtuple

from datasets.raw_data import load_raw_data
from modules.missing import generate_mask
from modules.utils import mask_edge, create_edge, create_edge_attr, create_node
# %%
EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['norm_parameters', 'num_features', 'num_continuous_features', 'num_categories'])

#%%
class CustomDataset(Dataset):
    def __init__(self, config, train=True):
        #%%
        self.config = config
        data, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(config["dataset"])
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.features = continuous_features + categorical_features
        
        self.ClfTarget = ClfTarget
        self.num_continuous_features = len(continuous_features)
        #%%
        # encoding for categorical variable
        data[categorical_features] = data[categorical_features].apply(
            lambda col: col.astype('category').cat.codes)
        self.num_categories = data[categorical_features].nunique(axis=0).to_list()

        data = data[self.features] # select features for training

        train_data, test_data = train_test_split(
            data, test_size=config["test_size"], random_state=config["seed"]
        )
        
        raw_data = train_data if train else test_data
        self.raw_data = raw_data.reset_index(drop=True)
        
        # y_train = train_data[[ClfTarget]].to_numpy()
        # X_train = train_data.drop(columns=ClfTarget)
        #%%
        """Generate missing values"""
        assert config["missing_type"] != "None"
        
        mask = generate_mask(
            # torch.from_numpy(train_data.values).float(), 
            train_data.values,
            config['missing_rate'],
            config['missing_rate'],
            config['seed']
        ) # 1:missing
        self.mask = mask
        #%%
        observed_edge_mask = torch.from_numpy(~mask.astype(bool).reshape(-1)) # [n*p] 0:missing
        double_observed_edge_mask = torch.cat(
            (observed_edge_mask, observed_edge_mask), # [2*n*p]
            dim=0
        ) 
        #%%
        """min-max scaling"""
        self.X_train_scaled, self.norm_parameters = self.normalization(train_data)
        X_train_scaled = pd.DataFrame(self.X_train_scaled)
        
        self.missing_data = X_train_scaled.mask(mask.astype(bool), np.nan)
        #%%
        """UNdirected bipartite graph"""
        edge_start, edge_end = create_edge(X_train_scaled) # 2*n*p
        edge_index = torch.tensor([edge_start, edge_end], dtype=int) # [2, 2*n*p]
        edge_attr = torch.tensor(
            create_edge_attr(X_train_scaled), # [2*n*p, 1]
            dtype=torch.float
        ) 

        node_init = create_node(X_train_scaled, config['node_mode']) # [(n+p)*p]
        #%%
        """masking edges"""
        observed_edge_index, observed_edge_attr = mask_edge(
            edge_index, 
            edge_attr,
            mask=double_observed_edge_mask, 
            remove_edge=True
        ) # train_edge_index is known
        observed_labels = observed_edge_attr[:int(observed_edge_attr.shape[0]/2), 0]
        #%%
        missing_edge_index, missing_edge_attr = mask_edge(
            edge_index, 
            edge_attr,
            mask=~double_observed_edge_mask, 
            remove_edge=True
        ) # test_edge_index in unknown, i.e. missing
        missing_labels = missing_edge_attr[:int(missing_edge_attr.shape[0]/2),0]
        #%%
        # """masking the y-values during training"""
        # # i.e. how we split the training and test sets
        # train_y_mask = (
        #     torch.FloatTensor(
        #         torch.tensor(y_train, dtype=torch.float).shape[0], 1
        #     ).uniform_() < config['missing_rate']
        # ).view(-1)
        # test_y_mask = ~train_y_mask
        #%%
        self.data = Data(
            x=torch.tensor(node_init, dtype=torch.float), # [n+p, p], 
            # y=torch.tensor(y_train, dtype=torch.float), # [n], 
            edge_index=edge_index, 
            edge_attr=edge_attr,
            # train_y_mask=train_y_mask,
            # test_y_mask=test_y_mask,
            observed_edge_index=observed_edge_index,
            observed_edge_attr=observed_edge_attr,
            observed_edge_mask=observed_edge_mask,
            observed_labels=observed_labels,
            missing_edge_index=missing_edge_index,
            missing_edge_attr=missing_edge_attr,
            missing_edge_mask=~observed_edge_mask,
            missing_labels=missing_labels, 
            df_X=X_train_scaled,
            # df_y=y_train,
            edge_attr_dim=missing_edge_attr.shape[-1],
            user_num=train_data.shape[0]
        )
        #%%
        # EncodedInfo = EncodedInfo(norm_parameters, len(features), num_continuous_features, num_categories)
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
            norm_parameters = {'min_val': min_val, 'max_val': max_val}

        else:
            min_val = parameters['min_val']
            max_val = parameters['max_val']
            
            # For each dimension
            for i in range(dim):
                norm_data[:,i] = norm_data[:,i] - min_val[i]
                norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6) # stable result
                
            norm_parameters = parameters     
            
        return norm_data, norm_parameters
    #%%
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self):
        return self.data
#%%
