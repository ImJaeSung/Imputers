"""
Reference:
[1] https://github.com/maxiaoba/GRAPE/blob/master/uci/uci_data.py
[2] https://github.com/G-AILab/IGRM/blob/main/uci/uci_data.py
"""
#%%
import pandas as pd
import os.path as osp
import inspect
from torch_geometric.data import Data
from sklearn import preprocessing

import torch
import random
import numpy as np
import pdb

from sklearn.model_selection import train_test_split
from datasets.raw_data import load_raw_data
from modules.missing import generate_mask
from modules.utils import mask_edge, create_edge, create_edge_attr, create_node
# %%
def load_data(config, normalize=True):
    #%%
    data, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(config["dataset"])
    train_data, test_data = train_test_split(
        data, test_size=config["test_size"], random_state=config["seed"]
    )
    
    y_train = train_data[[ClfTarget]].to_numpy()
    X_train = train_data.drop(columns=ClfTarget)
    
    if normalize:
        x = X_train.values
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_scaled = min_max_scaler.fit_transform(x)
        X_train = pd.DataFrame(X_train_scaled)
        
    edge_start, edge_end = create_edge(X_train) # 2*n*p
    edge_index = torch.tensor([edge_start, edge_end], dtype=int) # [2, 2*n*p]
    edge_attr = torch.tensor(create_edge_attr(X_train), dtype=torch.float) # [2*n*p]
    node_init = create_node(X_train, config['node_mode']) # sample node (const.) + feature node (one-hot)
    
    x = torch.tensor(node_init, dtype=torch.float) # [n+p, p]
    y = torch.tensor(y_train, dtype=torch.float) # [n]
    #%%
    """Generate missing values"""
    M = generate_mask(
        torch.from_numpy(X_train.values).float(), 
        config['missing_rate'],
        config['missing_rate'],
        config['seed']
    )
    train_mask = ~M.astype(bool) # [n*p] 0:missing
    train_edge_mask = torch.from_numpy(train_mask.reshape(-1))
    double_train_edge_mask = torch.cat((train_edge_mask, train_edge_mask), dim=0) # [2*n*p]
    #%%
    """masking edges"""
    train_edge_index, train_edge_attr = mask_edge(
        edge_index, 
        edge_attr,
        mask=double_train_edge_mask, 
        remove_edge=True) # train_edge_index is known
    train_labels = train_edge_attr[:int(train_edge_attr.shape[0]/2), 0]
    
    test_edge_index, test_edge_attr = mask_edge(
        edge_index, 
        edge_attr,
        mask=~double_train_edge_mask, 
        remove_edge=True) # test_edge_index in unknown, i.e. missing
    test_labels = test_edge_attr[:int(test_edge_attr.shape[0]/2),0]
    #%%
    """masking the y-values during training"""
    # i.e. how we split the training and test sets
    train_y_mask = (torch.FloatTensor(y.shape[0], 1).uniform_() < config['missing_rate']).view(-1)
    test_y_mask = ~train_y_mask
    #%%
    data = Data(
        x=x, 
        y=y, 
        edge_index=edge_index, 
        edge_attr=edge_attr,
        train_y_mask=train_y_mask, 
        test_y_mask=test_y_mask,
        train_edge_index=train_edge_index,
        train_edge_attr=train_edge_attr,
        train_edge_mask=train_edge_mask,
        train_labels=train_labels,
        test_edge_index=test_edge_index,
        test_edge_attr=test_edge_attr,
        test_edge_mask=~train_edge_mask,
        test_labels=test_labels, 
        df_X=X_train,
        df_y=y_train,
        edge_attr_dim=train_edge_attr.shape[-1],
        user_num=X_train.shape[0]
    )
    #%%
    if config['split_sample'] > 0.:
        if config['split_by'] == 'y':
            sorted_y, sorted_y_index = torch.sort(torch.reshape(y,(-1,)))
        elif config['split_by'] == 'random':
            sorted_y_index = torch.randperm(y.shape[0])
        
        lower_y_index = sorted_y_index[:int(np.floor(y.shape[0]*config['split_sample']))]
        higher_y_index = sorted_y_index[int(np.floor(y.shape[0]*config['split_sample'])):]
        
        # here we don't split x, only split edge
        """train"""
        half_train_edge_index = train_edge_index[:,:int(train_edge_index.shape[1]/2)]
        lower_train_edge_mask = []
        for node_index in half_train_edge_index[0]:
            if node_index in lower_y_index:
                lower_train_edge_mask.append(True)
            else:
                lower_train_edge_mask.append(False)
        lower_train_edge_mask = torch.tensor(lower_train_edge_mask)
        double_lower_train_edge_mask = torch.cat((lower_train_edge_mask, lower_train_edge_mask), dim=0)
        lower_train_edge_index, lower_train_edge_attr = mask_edge(
            train_edge_index, 
            train_edge_attr,
            double_lower_train_edge_mask, 
            True)
        lower_train_labels = lower_train_edge_attr[:int(lower_train_edge_attr.shape[0]/2),0]
        higher_train_edge_index, higher_train_edge_attr = mask_edge(
            train_edge_index, 
            train_edge_attr,
            ~double_lower_train_edge_mask,
            True)
        higher_train_labels = higher_train_edge_attr[:int(higher_train_edge_attr.shape[0]/2),0]
        
        """test"""
        half_test_edge_index = test_edge_index[:,:int(test_edge_index.shape[1]/2)]
        lower_test_edge_mask = []
        for node_index in half_test_edge_index[0]:
            if node_index in lower_y_index:
                lower_test_edge_mask.append(True)
            else:
                lower_test_edge_mask.append(False)
        lower_test_edge_mask = torch.tensor(lower_test_edge_mask)
        double_lower_test_edge_mask = torch.cat(
            (lower_test_edge_mask, lower_test_edge_mask), dim=0)
        
        lower_test_edge_index, lower_test_edge_attr = mask_edge(
            test_edge_index, 
            test_edge_attr,
            double_lower_test_edge_mask,
            True)
        lower_test_labels = lower_test_edge_attr[:int(lower_test_edge_attr.shape[0]/2), 0]
        
        higher_test_edge_index, higher_test_edge_attr = mask_edge(
            test_edge_index, 
            test_edge_attr,
            ~double_lower_test_edge_mask,
            True)
        higher_test_labels = higher_test_edge_attr[:int(higher_test_edge_attr.shape[0]/2),0]

        data.lower_y_index = lower_y_index
        data.higher_y_index = higher_y_index
        data.lower_train_edge_index = lower_train_edge_index
        data.lower_train_edge_attr = lower_train_edge_attr
        data.lower_train_labels = lower_train_labels
        data.higher_train_edge_index = higher_train_edge_index
        data.higher_train_edge_attr = higher_train_edge_attr
        data.higher_train_labels = higher_train_labels
        data.lower_test_edge_index = lower_test_edge_index
        data.lower_test_edge_attr = lower_test_edge_attr
        data.lower_test_labels = lower_train_labels
        data.higher_test_edge_index = higher_test_edge_index
        data.higher_test_edge_attr = higher_test_edge_attr
        data.higher_test_labels = higher_test_labels
        
    return data, train_data, test_data