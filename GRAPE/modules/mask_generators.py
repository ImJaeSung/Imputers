import numpy as np
import torch
from torchvision import transforms
from PIL import Image


# Mask generator for missing feature imputation

class MCARGenerator:
    """
    Returned mask is sampled from component-wise independent Bernoulli
    distribution with probability of component to be unobserved p.
    Such mask induces the type of missingness which is called
    in literature "missing completely at random" (MCAR).

    If some value in batch is missed, it automatically becomes unobserved.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, batch):
        nan_mask = torch.isnan(batch).float()  # missed values
        bernoulli_mask_numpy = np.random.choice(2, size=batch.shape,
                                                p=[1 - self.p, self.p])
        bernoulli_mask = torch.from_numpy(bernoulli_mask_numpy).float()
        mask = torch.max(bernoulli_mask, nan_mask)  # logical or
        return mask