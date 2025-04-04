#%%
from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F

import wandb
#%%
def train_function(
    model,
    config,
    optimizer,
    train_dataset,
    device):
    
    C = train_dataset.num_continuous_features
    x_categ = torch.from_numpy(train_dataset.data[:, C:]).float()
    x_cont = torch.from_numpy(train_dataset.data[:, :C].astype(np.float32))
    
    for epoch in tqdm(range(config["epochs"]), desc="Training..."):
        logs = {}
        batch_size = config["batch_size"]
        for i in tqdm(range(0, x_categ.shape[0], config["batch_size"])):
            st = i
            ed = st + batch_size
            
            x_cat = x_categ[st: ed].to(device)
            x_con = x_cont[st: ed].to(device)

            batch = torch.cat([x_con, x_cat], dim=1)
            nan_mask = batch.isnan()
            
            mask1 = torch.rand(batch.shape) < 0.15
            mask1 = mask1.to(device)
            
            mask = mask1 | nan_mask
            loss_mask = mask1 & ~nan_mask
            
            masked_batch = batch.clone()
            masked_batch[mask] = 0. # [MASKED] token
            
            masked_x_con = masked_batch[:, :C].float()
            masked_x_cat = masked_batch[:, C:].int()
            
            loss_ = []
            
            optimizer.zero_grad()
            masked_x_cat
            pred = model(masked_x_cat, masked_x_con)     
            
            cat_loss = 0
            st_ = 0
            for idx, j in enumerate(train_dataset.num_categories):
                pred_cat = pred[:, C:]
                target = torch.where(
                    masked_x_cat[:, idx].long() == 0,
                    masked_x_cat[:, idx].long(),
                    masked_x_cat[:, idx].long() - 1
                )
                
                cat_loss_ = F.cross_entropy(
                    pred_cat[:, st_:st_+j], 
                    target, 
                    reduction='none')[loss_mask[:, idx]].mean()
                st_ = j
                
                cat_loss += cat_loss_

            con_loss = F.mse_loss(
                pred[:, :C], 
                masked_x_con[:, :C], 
                reduction='none')[loss_mask[:, :C]].mean()
            
            loss = con_loss + cat_loss
            
            loss_.append(('cat_loss', cat_loss))
            loss_.append(('con_loss', con_loss))
            loss_.append(('loss', loss))
                    
            loss.backward()
            optimizer.step()
            
            for x, y in loss_:
                try:
                    logs[x] = logs.get(x) + [y.item()]
                except:
                    logs[x] = []
                    logs[x] = logs.get(x) + [y.item()]
        
        if epoch % 10 == 0:
            print_input = "[epoch {:03d}]".format(epoch + 1)
            print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
            print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        

    return