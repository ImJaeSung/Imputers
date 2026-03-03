"""
Reference:
[1] https://github.com/maxiaoba/GRAPE/blob/master/training/gnn_mdi.py
"""
#%%
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

import wandb
# from utils.plot_utils import plot_curve, plot_sample
from modules.utils import mask_edge
#%%
def train_function(
    gnn, 
    impute_model, 
    config, 
    data, 
    optimizer, 
    scheduler, 
    device):
    #%%
    """Data"""
    # data = Dataset.data # for debugging
    x = data.x.clone().detach().to(device)

    all_observed_edge_index = data.observed_edge_index.clone().detach().to(device)
    all_observed_edge_attr = data.observed_edge_attr.clone().detach().to(device)
    all_observed_labels = data.observed_labels.clone().detach().to(device)
    
    missing_input_edge_index = all_observed_edge_index
    missing_input_edge_attr = all_observed_edge_attr
    missing_edge_index = data.missing_edge_index.clone().detach().to(device)
    missing_edge_attr = data.missing_edge_attr.clone().detach().to(device)
    missing_labels = data.missing_labels.clone().detach().to(device)
    #%%
    if hasattr(data,'class_values'):
        class_values = data.class_values.clone().detach().to(device)
    #%%
    observed_edge_index = all_observed_edge_index 
    observed_edge_attr = all_observed_edge_attr
    observed_labels = all_observed_labels
     
    print(
        "train edge num is {}, test edge num is input {}, output {}".format(
            observed_edge_attr.shape[0],
            missing_input_edge_attr.shape[0], 
            missing_edge_attr.shape[0]
        )
    )
    #%%
    for epoch in tqdm(range(config['epochs']), desc="training..."):
        #%%
        logs = {
            "Train_loss":[],
            # "Test_rmse":[],
            # "Test_l1":[],
            "Lr":[]
        }
        #%%
        gnn.train()
        impute_model.train()
        #%%
        known_mask = (
            torch.FloatTensor(int(observed_edge_attr.shape[0] / 2), 1).uniform_() < config['known']
        ).view(-1).to(device) 
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(
            observed_edge_index, 
            observed_edge_attr,
            double_known_mask, 
            remove_edge=True
        )
        #%%
        optimizer.zero_grad()
        x_embed = gnn(x, known_edge_attr, known_edge_index) # [n+p, 64]
        pred_ = impute_model(
            [x_embed[observed_edge_index[0]], x_embed[observed_edge_index[1]]]
        ) # equal to observed_edge_attr.shape
        #%%
        if hasattr(config,'ce_loss') and config['ce_loss']:
            pred = pred_[:int(observed_edge_attr.shape[0] / 2)]
        else:
            pred = pred_[:int(observed_edge_attr.shape[0] / 2),0]
            
        if config['loss_mode'] == 1:
            pred[known_mask] = observed_labels[known_mask]
        
        # label_train = observed_labels

        if hasattr(config,'ce_loss') and config['ce_loss']:
            loss = F.cross_entropy(pred, observed_labels)
        else:
            loss = F.mse_loss(pred, observed_labels)
        
        #%%
        loss.backward()
        optimizer.step()
        logs['Train_loss'] = loss.item()
        
        if scheduler is not None:
            scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            logs['Lr'] = param_group['lr']
        #%%
        if epoch % 100 == 0:
            print_input = "[epoch {:03d}]".format(epoch + 1)
            print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
            print(print_input)
        #%%
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
# %%
