"""
Reference:
[1] https://github.com/tigvarts/vaeac/blob/master/train.py
"""
#%%
from copy import deepcopy
from sys import stderr
from tqdm import tqdm
from math import ceil

import wandb

import numpy as np
import torch

import modules
from modules import utils
from modules.utils import extend_batch, get_validation_iwae
#%%
def train_function(
        model,
        networks,
        config,
        optimizer,
        train_dataloader,
        valid_dataloader,
        device,
        verbose=True):
    mask_generator = networks['mask_generator']
    vlb_scale_factor = networks.get('vlb_scale_factor', 1)

    # number of batches after which it is time to do validation
    valid_batch = ceil(
        len(train_dataloader) / config["validations_per_epoch"]
    )

    # a list of validation IWAE estimates
    validation_iwae = []
    # a list of running variational lower bounds on the train set
    train_vlb = []
    # the length of two lists above is the same because the new
    # values are inserted into them at the validation checkpoints only

    # best model state according to the validation IWAE
    best_state = None
    
    # main train loop
    for epoch in range(config["epochs"]):
        logs = {
            'loss': [], 
        }
        # one epoch
        for i, batch in tqdm(enumerate(train_dataloader), desc="inner loop..."):

            # the time to do a checkpoint is at start and end of the training
            # and after processing validation_batches batches
            if any([
                        i == 0 and epoch == 0,
                        i % valid_batch == valid_batch - 1,
                        i + 1 == len(train_dataloader)
                    ]):
                val_iwae = get_validation_iwae(
                    valid_dataloader, 
                    mask_generator,
                    config["batch_size"], 
                    model,
                    config["validation_iwae_num_samples"],
                    verbose
                )
                validation_iwae.append(val_iwae)

                # if current model validation IWAE is the best validation IWAE
                # over the history of training, the current state
                # is saved to best_state variable
                if max(validation_iwae[::-1]) <= val_iwae:
                    best_state = deepcopy({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'validation_iwae': validation_iwae,
                        'train_vlb': train_vlb,
                    })

                if verbose:
                    print(file=stderr)
                    print(file=stderr)

            loss_ = []

            # if batch size is less than batch_size, extend it with objects
            # from the beginning of the dataset
            batch = extend_batch(batch, train_dataloader, config["batch_size"])

            # generate mask and do an optimizer step over the mask and the batch
            mask = mask_generator(batch)

            batch = batch.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            vlb = model.batch_vlb(batch, mask).mean()
            loss = -vlb / vlb_scale_factor
            
            loss.backward()
            optimizer.step()

            loss_.append(('loss', loss))

            """accumulate losses"""
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()]
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})

    return best_state