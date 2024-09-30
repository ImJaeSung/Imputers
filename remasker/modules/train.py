#%%
import math
from tqdm import tqdm

import torch
import wandb

from modules.utils import adjust_learning_rate, NativeScaler
#%%
def train_function(
        config, 
        model,
        optimizer, 
        train_dataloader,
        device):
    
    if config["lr"] is None:
        eff_batch_size = config["batch_size"] * config["accum_iter"]
        config["lr"] = config["blr"] * eff_batch_size / 64
    
    loss_scaler = NativeScaler()

    for epoch in range(config["max_epochs"]):
        optimizer.zero_grad()
        total_loss = 0

        iter = 0
        for iter, (samples, masks) in tqdm(enumerate(train_dataloader), desc="inner loop..."):
            loss_ = []
   
            # we use a per iteration (instead of per epoch) lr scheduler
            if iter % config["accum_iter"] == 0:
                adjust_learning_rate(
                    optimizer, 
                    iter / len(train_dataloader) + epoch, 
                    config["lr"], 
                    config["min_lr"],
                    config["max_epochs"], 
                    config["warmup_epochs"]
                )
            
            samples = samples.unsqueeze(dim=1)
            samples = samples.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                loss, _, _, _ = model(samples, masks)
                loss_value = loss.item()

                loss_.append(('loss', loss))

                if math.isfinite(loss_value):
                    total_loss += loss_value

            loss /= config["accum_iter"]
            loss_scaler(
                loss, 
                optimizer,
                parameters=model.parameters(),
                update_grad=(iter + 1) % config["accum_iter"] == 0
            )

            if (iter + 1) % config["accum_iter"] == 0:
                optimizer.zero_grad()

        total_loss = (total_loss / (iter + 1)) ** 0.5
        
        print_input = f"[epoch {epoch+1:03d}/{config['max_epochs']}]"
        print_input += f" total Loss: {total_loss:.4f}"
        print(print_input)
        
        """updatae log"""
        wandb.log({"Loss" : total_loss})

    return 