"""Reference:
[1] https://github.com/yburda/iwae/blob/master/train.py
[2] https://github.com/xqding/Importance_Weighted_Autoencoders/blob/master/MNIST/script/main_train.py
"""
#%%
from tqdm import tqdm
import torch
#%%
def train_function(
        model, 
        optimizer, 
        # batches_data,
        # batches_mask,
        train_dataloader,
        mask_loader,
        device):
    
    logs = {
        "NLL":[],
        "Disc_loss":[],
        "Total_loss":[]
    }

    for x, mask in tqdm(zip(train_dataloader, mask_loader), desc="inner loop..."):
        loss_ = []
        x, mask = x.float().to(device), mask.to(device) # 0:missing

        optimizer.zero_grad()
        # x = torch.from_numpy(batches_data[i]).float().to(device)
        # mask = torch.from_numpy(batches_mask[i]).float().to(device)
        
        neg_bound, disc_loss = model.miwae_loss(x, mask)
        loss = neg_bound + disc_loss
        loss_.append(("NLL", neg_bound))
        loss_.append(("Disc_loss", disc_loss))
        loss_.append(("Total_loss", loss))
        loss.backward()
        optimizer.step()

        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
        
    return logs
# %%
