"""
Reference:
[1] https://github.com/maxiaoba/GRAPE/blob/master/models/prediction_model.py
"""
#%%
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import get_activation, renormalization
#%%
class MLPNet(nn.Module):
    def __init__(self, gnn, config, data):
        super(MLPNet, self).__init__()
        self.config = config
        self.data = data
        
        if config['impute_hiddens'] == '':
            impute_hiddens = []
        else:
            impute_hiddens = list(
				map(int, config['impute_hiddens'].split('_'))
			)
        
        if config['concat_states']:
            input_dim = config['node_dim'] * len(gnn.convs) * 2
        else:
            input_dim = config['node_dim'] * 2 # 
            
        if hasattr(config,'ce_loss') and config['ce_loss']:
            output_dim = len(data.class_values)
        else:
            output_dim = 1
        
        layers = nn.ModuleList()
        for layer_size in impute_hiddens:
            hidden_dim = layer_size
            layer = nn.Sequential(
				nn.Linear(input_dim, hidden_dim),
				get_activation(config['impute_activation']),
				nn.Dropout(config['dropout']),
			)
            layers.append(layer)
            input_dim = hidden_dim

        layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
			get_activation(None),
		)
        layers.append(layer)
        self.layers = layers

    def forward(self, inputs):
        inputs = [inputs] if torch.is_tensor(inputs) else inputs # list
        input_var = torch.cat(inputs, -1) # [train_edge_num, 128]
        for layer in self.layers:
            input_var = layer(input_var)
        
        return input_var
    
    def impute(self, train_dataset, gnn, device, seed=0):
        #%%
        torch.random.manual_seed(seed)
        #%%
        with torch.no_grad():
            #%%
            all_observed_edge_index = train_dataset.data.observed_edge_index.clone().detach().to(device)
            all_observed_edge_attr = train_dataset.data.observed_edge_attr.clone().detach().to(device)
            all_observed_labels = train_dataset.data.observed_labels.clone().detach().to(device)
            
            missing_input_edge_index = all_observed_edge_index
            missing_input_edge_attr = all_observed_edge_attr
            missing_edge_index = train_dataset.data.missing_edge_index.clone().detach().to(device)
            missing_edge_attr = train_dataset.data.missing_edge_attr.clone().detach().to(device)
            missing_labels = train_dataset.data.missing_labels.clone().detach().to(device)
            #%%
            x = train_dataset.data.x.clone().detach().to(device)
            x_embed = gnn(x, missing_input_edge_attr, missing_input_edge_index) # [n+p, 64]
            pred_ = self(
                [x_embed[missing_edge_index[0], :], x_embed[missing_edge_index[1], :]]
            ) # 
            #%%
            if hasattr(train_dataset.data,'class_values'):
                class_values = train_dataset.data.class_values.clone().detach().to(device)
            #%%
            if hasattr(self.config, 'ce_loss') and self.config['ce_loss']:
                pred_test = class_values[
                    pred_[:int(missing_edge_attr.shape[0] / 2)].max(1)[1]]
                label_test = class_values[missing_labels]
            elif hasattr(self.config,'norm_label') and self.config['norm_label']:
                pred_test = pred_[:int(missing_edge_attr.shape[0] / 2), 0]
                pred_test = pred_test * max(class_values)
                label_test = missing_labels
                label_test = label_test * max(class_values)
            else:
                pred_test = pred_[:int(missing_edge_attr.shape[0] / 2), 0]
                label_test = missing_labels
            #%%
            """imputation"""
            imputed_data = train_dataset.missing_data.to_numpy().copy()

            src = missing_edge_index[0].detach().cpu().numpy()
            dst = missing_edge_index[1].detach().cpu().numpy()

            i = src[:src.shape[0]//2]  
            j = dst[:src.shape[0]//2] - imputed_data.shape[0]
                
            imputed_values = pred_test.detach().cpu().numpy().astype(np.float32)

            assert len(imputed_values) == np.isnan(imputed_data).sum().sum()
        
            imputed_data[i, j] = imputed_values
            #%%
            """post-processing"""
            imputed_data = renormalization(
                imputed_data, 
                train_dataset.norm_parameters
            )
            
            imputed_data = pd.DataFrame(imputed_data, columns=train_dataset.features)
            imputed_data[train_dataset.categorical_features] = imputed_data[train_dataset.categorical_features].astype(int)
            imputed_data[train_dataset.integer_features] = imputed_data[train_dataset.integer_features].round(0).astype(int)
            data = imputed_data*train_dataset.mask + train_dataset.raw_data*(1. - train_dataset.mask)
            #%%
        return data
            
# %%
