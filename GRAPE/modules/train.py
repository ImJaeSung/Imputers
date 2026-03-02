"""
Reference:
[1] https://github.com/maxiaoba/GRAPE/blob/master/training/gnn_mdi.py
"""
#%%
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pickle

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
    log_path, 
    device):
    #%%
    """Data"""
    x = data.x.clone().detach().to(device)
    if hasattr(config,'split_sample') and config['split_sample'] > 0.:
        if config['split_train']:
            all_train_edge_index = data.lower_train_edge_index.clone().detach().to(device)
            all_train_edge_attr = data.lower_train_edge_attr.clone().detach().to(device)
            all_train_labels = data.lower_train_labels.clone().detach().to(device)
        else:
            all_train_edge_index = data.train_edge_index.clone().detach().to(device)
            all_train_edge_attr = data.train_edge_attr.clone().detach().to(device)
            all_train_labels = data.train_labels.clone().detach().to(device)
        if config['split_test']:
            test_input_edge_index = data.higher_train_edge_index.clone().detach().to(device)
            test_input_edge_attr = data.higher_train_edge_attr.clone().detach().to(device)
        else:
            test_input_edge_index = data.train_edge_index.clone().detach().to(device)
            test_input_edge_attr = data.train_edge_attr.clone().detach().to(device)
        test_edge_index = data.higher_test_edge_index.clone().detach().to(device)
        test_edge_attr = data.higher_test_edge_attr.clone().detach().to(device)
        test_labels = data.higher_test_labels.clone().detach().to(device)
    else:
        all_train_edge_index = data.train_edge_index.clone().detach().to(device)
        all_train_edge_attr = data.train_edge_attr.clone().detach().to(device)
        all_train_labels = data.train_labels.clone().detach().to(device)
        test_input_edge_index = all_train_edge_index
        test_input_edge_attr = all_train_edge_attr
        test_edge_index = data.test_edge_index.clone().detach().to(device)
        test_edge_attr = data.test_edge_attr.clone().detach().to(device)
        test_labels = data.test_labels.clone().detach().to(device)
    #%%
    if hasattr(data,'class_values'):
        class_values = data.class_values.clone().detach().to(device)
    #%%
    if config['valid'] > 0.:
        valid_mask = (
            torch.FloatTensor(int(all_train_edge_attr.shape[0] / 2), 1).uniform_() < config['valid']
        ).view(-1).to(device)  
        print("valid mask sum: ",torch.sum(valid_mask))
    
        train_labels = all_train_labels[~valid_mask]
        valid_labels = all_train_labels[valid_mask]
        double_valid_mask = torch.cat((valid_mask, valid_mask), dim=0)
        valid_edge_index, valid_edge_attr = mask_edge(all_train_edge_index, all_train_edge_attr, double_valid_mask, True)
        train_edge_index, train_edge_attr = mask_edge(all_train_edge_index, all_train_edge_attr, ~double_valid_mask, True)
        print(
            "train edge num is {}, valid edge num is {}, test edge num is input {} output {}".format(
                train_edge_attr.shape[0], 
                valid_edge_attr.shape[0],
                test_input_edge_attr.shape[0], 
                test_edge_attr.shape[0]
            )
        )
    
        Valid_rmse = []
        Valid_l1 = []
        best_valid_rmse = np.inf
        best_valid_rmse_epoch = 0
        best_valid_l1 = np.inf
        best_valid_l1_epoch = 0
    else:
        train_edge_index, train_edge_attr, train_labels = all_train_edge_index, all_train_edge_attr, all_train_labels
        print(
            "train edge num is {}, test edge num is input {}, output {}".format(
                train_edge_attr.shape[0],
                test_input_edge_attr.shape[0], 
                test_edge_attr.shape[0]
            )
        )
    #%%
    if config['auto_known']:
        known = float(all_train_labels.shape[0])
        known /= float(all_train_labels.shape[0]+test_labels.shape[0])
        config['known'] = known
        print(
            "auto calculating known is {}/{} = {:.3g}".format(
                all_train_labels.shape[0],
                all_train_labels.shape[0] + test_labels.shape[0],
                config['known']
            )
        )
    #%%
    for epoch in tqdm(range(config['epochs']), desc="training..."):
        #%%
        if config['valid'] > 0.:
            logs = {
                "Train_loss":[],
                "Valid_rmse":[],
                "Valid_l1":[],
                "Test_rmse":[],
                "Test_l1":[],
                "Lr":[]
            }
        else:
            logs = {
                "Train_loss":[],
                "Test_rmse":[],
                "Test_l1":[],
                "Lr":[]
            }
        #%%
        gnn.train()
        impute_model.train()

        known_mask = (
            torch.FloatTensor(int(train_edge_attr.shape[0] / 2), 1).uniform_() < config['known']
        ).view(-1).to(device)  
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(
            train_edge_index, train_edge_attr, double_known_mask, True
        )
        #%%
        optimizer.zero_grad()
        x_embed = gnn(x, known_edge_attr, known_edge_index) # [n+p, 64]
        pred = impute_model(
            [x_embed[train_edge_index[0]], x_embed[train_edge_index[1]]]
        ) # [train_edge_num,1]
        #%%
        if hasattr(config,'ce_loss') and config['ce_loss']:
            pred_train = pred[:int(train_edge_attr.shape[0] / 2)]
        else:
            pred_train = pred[:int(train_edge_attr.shape[0] / 2),0]
        if config['loss_mode'] == 1:
            pred_train[known_mask] = train_labels[known_mask]
        label_train = train_labels

        if hasattr(config,'ce_loss') and config['ce_loss']:
            loss = F.cross_entropy(pred_train,train_labels)
        else:
            loss = F.mse_loss(pred_train, label_train)
        
        loss.backward()
        optimizer.step()
        logs['Train_loss'] = loss.item()
        
        if scheduler is not None:
            scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            logs['Lr'] = param_group['lr']

        gnn.eval()
        impute_model.eval()
        #%%
        """Validation or Test"""
        with torch.no_grad():
            #%%
            if config['valid'] > 0.:
                x_embed = gnn(x, train_edge_attr, train_edge_index)
                pred = impute_model(
                    [x_embed[valid_edge_index[0], :], x_embed[valid_edge_index[1], :]]
                )
                if hasattr(config,'ce_loss') and config['ce_loss']:
                    pred_valid = class_values[pred[:int(valid_edge_attr.shape[0] / 2)].max(1)[1]]
                    label_valid = class_values[valid_labels]
                elif hasattr(config,'norm_label') and config['norm_label']:
                    pred_valid = pred[:int(valid_edge_attr.shape[0] / 2),0]
                    pred_valid = pred_valid * max(class_values)
                    label_valid = valid_labels
                    label_valid = label_valid * max(class_values)
                else:
                    pred_valid = pred[:int(valid_edge_attr.shape[0] / 2),0]
                    label_valid = valid_labels
                    
                mse = F.mse_loss(pred_valid, label_valid)
                valid_rmse = np.sqrt(mse.item())
                l1 = F.l1_loss(pred_valid, label_valid)
                valid_l1 = l1.item()
                
                logs['Valid_rmse'] = valid_rmse
                logs['Valid_l1'] = valid_l1
                
                if valid_l1 < best_valid_l1:
                    best_valid_l1 = valid_l1
                    best_valid_l1_epoch = epoch
                    if config['save_model']:
                        torch.save(gnn, log_path + 'model_best_valid_l1.pt')
                        torch.save(impute_model, log_path + 'impute_model_best_valid_l1.pt')
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_valid_rmse_epoch = epoch
                    if config['save_model']:
                        torch.save(gnn, log_path + 'model_best_valid_rmse.pt')
                        torch.save(impute_model, log_path + 'impute_model_best_valid_rmse.pt')
                Valid_rmse.append(valid_rmse)
                Valid_l1.append(valid_l1)
            #%%
            x_embed = gnn(x, test_input_edge_attr, test_input_edge_index)
            pred = impute_model(
                [x_embed[test_edge_index[0], :], x_embed[test_edge_index[1], :]]
            )
            
            if hasattr(config, 'ce_loss') and config['ce_loss']:
                pred_test = class_values[pred[:int(test_edge_attr.shape[0] / 2)].max(1)[1]]
                label_test = class_values[test_labels]
            elif hasattr(config,'norm_label') and config['norm_label']:
                pred_test = pred[:int(test_edge_attr.shape[0] / 2),0]
                pred_test = pred_test * max(class_values)
                label_test = test_labels
                label_test = label_test * max(class_values)
            else:
                pred_test = pred[:int(test_edge_attr.shape[0] / 2),0]
                label_test = test_labels
            #%%    
            test_mse = F.mse_loss(pred_test, label_test)
            test_rmse = np.sqrt(test_mse.item())
            test_l1 = F.l1_loss(pred_test, label_test)
            test_l1 = test_l1.item()
            #%%
            logs['Test_rmse'] = test_rmse
            logs['Test_l1'] = test_l1            
        
        if epoch % 100 == 0:
            print_input = "[epoch {:03d}]".format(epoch + 1)
            print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
            print(print_input)
        #%%
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
# %%
