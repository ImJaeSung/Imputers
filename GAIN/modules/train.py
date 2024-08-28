"""Reference:
[1] https://github.com/jsyoon0823/GAIN/blob/master/gain.py
"""
#%%
from tqdm import tqdm
import numpy as np
import torch

import wandb

from modules.utils import sample_binary, sample_uniform, discriminator_loss, generator_loss
#%%
def train_function(
        D, 
        G, 
        config, 
        optimizer_D,
        optimizer_G,
        train_dataloader, 
        device):

    for epoch in range(config['epochs']):
        logs = {
            'D_loss': [],
            'G_loss': []
        }

        for i, batch in tqdm(enumerate(train_dataloader), desc="inner loop..."):
            loss_ = []
            
            get_batch = batch.shape[0]
            
            X = batch.float().to(device) 
            M = 1 - torch.isnan(batch).float().to(device)

            H1 = sample_binary(
                get_batch, 
                config["dim"], 
                config["hint_rate"]
            ).float().to(device) 
            
            H = M * H1
            H = H.float().to(device)

            Z = sample_uniform(
                get_batch, 
                config["dim"]
            ).float().to(device)

            X = torch.nan_to_num(X,nan=0.0)
            New_X = M * X + (1-M) * Z  
            New_X = New_X.float().to(device)

            optimizer_D.zero_grad()
            D_loss = discriminator_loss(D, G, M, New_X, H)
            D_loss.backward()
            optimizer_D.step()
        
            loss_.append(('D_loss', D_loss))
        
            optimizer_G.zero_grad()
            G_loss, _, _ = generator_loss(D,G, X, M, New_X, H, config["alpha"])
            G_loss.backward()
            optimizer_G.step()

            loss_.append(('G_loss', G_loss))

            """accumulate losses"""
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()]

        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})

    return