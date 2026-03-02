"""
Reference:
[1] https://github.com/maxiaoba/GRAPE/blob/master/models/prediction_model.py
"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.utils import get_activation
#%%
class MLPNet(nn.Module):
    def __init__(self, gnn,config, data):
        super(MLPNet, self).__init__()
        
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