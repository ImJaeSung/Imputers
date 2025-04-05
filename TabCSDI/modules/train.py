"""Reference:
[1] https://github.com/pfnet-research/TabCSDI/blob/main/src/utils_table.py
"""
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
    scheduler,
    train_dataloader,
    mask_loader
    ):  

    for epoch in range(config["epochs"]):
        logs = {}
        
        with tqdm(zip(train_dataloader, mask_loader), 
                total=min(len(train_dataloader), len(mask_loader)), 
                mininterval=5.0, 
                maxinterval=50.0) as pbar:
            
            loss_ = []
            for train_batch, mask_batch in pbar:
                optimizer.zero_grad()
                # The forward method returns loss.
                loss = model(train_batch, mask_batch)
                loss_.append(('loss', loss))
                loss.backward()
                optimizer.step()
                
            scheduler.step()
                
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
# %%
