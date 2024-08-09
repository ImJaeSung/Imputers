"""Reference:
[1] https://github.com/pamattei/miwae/blob/master/Pytorch%20notebooks/MIWAE_Pytorch_exercises_demo_ProbAI.ipynb
"""
#%%
import os
import sys
#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical

from modules.prior import generate_prior
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.simulation import set_random_seed

#%%
class MIWAE(nn.Module):
    def __init__(self, config, EncodedInfo, device):
        super(MIWAE, self).__init__()
        self.config = config
        self.device = device
        
        self.input_dim = config["input_dim"] # num_features
        self.hidden_dim = config["hidden_dim"] # 128
        self.latent_dim = config["latent_dim"] # 1
        self.k = config["k"] # default 20
        self.num_continuous_features = EncodedInfo.num_continuous_features
        self.num_categories = EncodedInfo.num_categories
        
        self.encoder_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2*self.latent_dim), # Encoder will output both (1)the mean and (2)the diagonal covariance
        )

        self.decoder_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3*self.num_continuous_features+sum(self.num_categories)), # Decoder will output (1)the mean, (2)the scale, and (3) the number of degree
            # nn.Sigmoid(),
        )

        self.apply(self.weights_init)
    
    @staticmethod
    def weights_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight)
            
    def encoder(self, x):
        out_encoder = self.encoder_layer(x)
        
        return out_encoder
    
    def decoder(self, z):
        out_decoder = self.decoder_layer(z)

        return out_decoder

    def miwae_loss(self, x, mask):
        """masking"""
        # x = next(iter(train_dataloader))
        # x.size()
        # x = xhat
        # assert x.isnan().sum() == x.size(0)*x.size(1)-mask.sum() # checking masking
        batch_size = len(x)

        out_encoder = self.encoder(x)
        q_zgivenxobs = torch.distributions.Independent(
            torch.distributions.Normal(
                loc=out_encoder[..., :self.latent_dim],
                scale=nn.Softplus()(out_encoder[..., self.latent_dim:(2*self.latent_dim)])+ 0.001
            ), 1
        )
    
        zgivenx = q_zgivenxobs.rsample([self.k]) # (K, b, 1)
        zgivenx_flat = zgivenx.reshape([batch_size*self.k, self.latent_dim])
        
        
        out_decoder = self.decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :self.num_continuous_features] # (b*K, num_continuous_features)
        all_scales_obs_model = nn.Softplus()(
            out_decoder[..., self.num_continuous_features:(2*self.num_continuous_features)]
        ) + 0.001 # (b*K, num_continuous_features)

        all_degfreedom_obs_model = nn.Softplus()(
            out_decoder[..., (2*self.num_continuous_features):(3*self.num_continuous_features)]
        ) + 3 # (b*K, num_continuous_features)
        logit = out_decoder[:,(3*self.num_continuous_features):]
        
        data_flat = torch.Tensor.repeat(
            x[:, :self.num_continuous_features],[self.k, 1]).reshape(-1, 1) # (b*K*num_continuous_features, 1)
        tiledmask = torch.Tensor.repeat(
            mask[:, :self.num_continuous_features],[self.k, 1]) # (b*K, num_continuous_features)

        all_log_pxgivenz_flat = torch.distributions.StudentT(
            loc=all_means_obs_model.reshape(-1, 1),
            scale=all_scales_obs_model.reshape(-1, 1),
            df=all_degfreedom_obs_model.reshape(-1, 1)
        ).log_prob(data_flat) # (b*K*num_continuous_features, 1)

        all_log_pxgivenz = all_log_pxgivenz_flat.reshape(batch_size*self.k, self.num_continuous_features)
        
        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask, axis=1).reshape(self.k, batch_size)
        
        prior = generate_prior(self.config, self.device)
        logpz = prior.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)
        
        #Category Var loss
        cat_mask = mask[:, self.num_continuous_features:].to(torch.bool)
        cat_mask = torch.Tensor.repeat(cat_mask, [self.k, 1])
        disc_loss = 0
        st = 0
        end = 0
        for i, n in enumerate(self.num_categories):
            end += n
            target = x[:,self.num_continuous_features+i].to(torch.long)
            target = torch.Tensor.repeat(target,[self.k, 1]).reshape(-1)
            disc_loss += torch.nn.functional.cross_entropy(
                logit[:,st:end][cat_mask[:,i]], target[cat_mask[:,i]]-1
            )
            st = end
        
        neg_bound = -torch.mean(
            torch.logsumexp(logpxobsgivenz + logpz - logq, axis=0)
        )
        
        return neg_bound, disc_loss
    
    def impute(self, train_dataset, M, seed=0):
        set_random_seed(seed)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False)
        
        imputed = []
        for batch in train_dataloader:
            batch_size = len(batch)
            mask = (~torch.isnan(batch)).to(torch.float32).to(self.device)
            batch = batch.to(self.device)
            batch[torch.isnan(batch)] = 0
        
            out_encoder = self.encoder(batch)
            q_zgivenxobs = torch.distributions.Independent(
                torch.distributions.Normal(
                    loc=out_encoder[..., :self.latent_dim],
                    scale=nn.Softplus()(out_encoder[..., self.latent_dim:(2*self.latent_dim)])+ 0.001
                ), 1
            )
        
            zgivenx = q_zgivenxobs.rsample([M]) # (K, b, 1)
            zgivenx_flat = zgivenx.reshape([-1, self.latent_dim])
            
            
            out_decoder = self.decoder(zgivenx_flat)
            all_means_obs_model = out_decoder[..., :self.num_continuous_features] # (b*K, num_continuous_features)
            all_scales_obs_model = nn.Softplus()(
                out_decoder[..., self.num_continuous_features:(2*self.num_continuous_features)]
            ) + 0.001 # (b*K, num_continuous_features)

            all_degfreedom_obs_model = nn.Softplus()(
                out_decoder[..., (2*self.num_continuous_features):(3*self.num_continuous_features)]
            ) + 3 
            logit = out_decoder[:,(3*self.num_continuous_features):]
            
            data_flat = torch.Tensor.repeat(batch[:, :self.num_continuous_features],[M, 1]).reshape(-1, 1) # (b*K*num_continuous_features, 1)
            tiledmask = torch.Tensor.repeat(mask[:, :self.num_continuous_features],[M, 1]) # (b*K, num_continuous_features)

            all_log_pxgivenz_flat = torch.distributions.StudentT(
                loc=all_means_obs_model.reshape(-1, 1),
                scale=all_scales_obs_model.reshape(-1, 1),
                df=all_degfreedom_obs_model.reshape(-1, 1)
            ).log_prob(data_flat) # (b*K*num_continuous_features, 1)

            all_log_pxgivenz = all_log_pxgivenz_flat.reshape(batch_size*M, self.num_continuous_features)
        
            logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask, axis=1).reshape(M, batch_size)
            
            prior = generate_prior(self.config, self.device)
            logpz = prior.log_prob(zgivenx)
            logq = q_zgivenxobs.log_prob(zgivenx)
            
            xgivenz = torch.distributions.Independent(
                torch.distributions.StudentT(
                    loc=all_means_obs_model,
                    scale=all_scales_obs_model,
                    df=all_degfreedom_obs_model),1)
            
            disc_loss = []
            xm_cat = []
            st = 0
            end = 0
            for i, n in enumerate(self.num_categories):
                end += n
                target = batch[:,self.num_continuous_features+i].to(torch.long)
                target = torch.Tensor.repeat(target,[M, 1]).reshape(-1)
                target = F.one_hot(target, num_classes=n+1)[:,1:].float()
                
                loss = torch.sum(torch.log_softmax(logit[:,st:end], dim=-1)*target, dim=-1).reshape(M,-1)
                disc_loss.append(loss)
                
                xm_cat.append(Categorical(logits=logit[:,st:end]).sample())
                
                st = end
                
            # cross_entropy = torch.sum(torch.stack(disc_loss), dim=0)
            xm_cats = torch.stack(xm_cat).T.reshape([M,batch_size,-1])

            # imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq-cross_entropy,dim=0) # these are w_1,....,w_L for all observations in the batch
            xms = xgivenz.sample().reshape([M,batch_size,self.num_continuous_features])
            
            # label = Categorical(probs=imp_weights.T).sample_n(M)
            # xm = xms[label,torch.arange(batch_size),:]
            # xm_cat = xm_cats[label,torch.arange(batch_size),:]
            # xm = torch.cat([xm, xm_cat], dim=-1)
            xm = torch.cat([xms, xm_cats], dim=-1)
            
            batch = torch.Tensor.repeat(batch, [M,1,1])
            mask = torch.Tensor.repeat(mask, [M,1,1])
            mask = ~mask.to(torch.bool)
            batch[mask] = xm[mask]
            imputed.append(batch)
        
        full_data = torch.cat(imputed, dim=1)
        
        data_list = []
        for data in full_data: 
            train_dataset.EncodedInfo.mean
            data = pd.DataFrame(data.cpu().numpy(), columns=train_dataset.features)
        
            data[train_dataset.continuous_features] = data[train_dataset.continuous_features]*train_dataset.EncodedInfo.std + train_dataset.EncodedInfo.mean
            
            data[train_dataset.categorical_features] = data[train_dataset.categorical_features].astype(int)
            data[train_dataset.integer_features] = data[train_dataset.integer_features].round(0).astype(int)
            data_list.append(data)
        
        return data_list
# %%
