"""Reference:
[1] https://github.com/pamattei/miwae/blob/master/Pytorch%20notebooks/MIWAE_Pytorch_exercises_demo_ProbAI.ipynb
"""
import torch
#%%
def generate_prior(config, device):
    loc = torch.zeros(config['latent_dim']).to(device)
    scale = torch.ones(config['latent_dim']).to(device)
    p_z = torch.distributions.Independent(
        torch.distributions.Normal(loc=loc, scale = scale), 1
    )
    
    return p_z