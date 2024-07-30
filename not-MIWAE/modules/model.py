#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Normal, Bernoulli, Categorical

import numpy as np
import pandas as pd

import modules
from modules.prior import generate_prior
from modules.utils import set_random_seed
#%%
class notMIWAE(nn.Module):
    def __init__(self, config, EncodedInfo, device):
        super(notMIWAE, self).__init__()
        self.config = config
        self.device = device
        
        self.input_dim = config["input_dim"] 
        self.hidden_dim = config["hidden_dim"] 
        self.latent_dim = config["latent_dim"] 
        
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
            nn.Linear(
                self.hidden_dim, 2*self.num_continuous_features+sum(self.num_categories)
            ), # Decoder will output (1)the mean, (2)the scale, and (3) the number of degree
            # nn.Sigmoid(),
        )
        
        self.mask_layer = nn.Linear(
            self.num_continuous_features + sum(self.num_categories), self.input_dim
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

    def loss_function(self, x, mask):
        """
        x: batch
        mask: True(Complete), False(Missing)
        """
        batch_size = len(x)

        l = self.latent_dim
        p = self.num_continuous_features
        k = self.k

        # Encoder
        out_encoder = self.encoder(x)
        qz_given_xo = torch.distributions.Independent(
            Normal(
                loc=out_encoder[..., :l],
                scale=nn.Softplus()(out_encoder[..., l:(2*l)])+ 0.001
            ), 
            1,
        )

        # z sampling k
        z_given_x = qz_given_xo.rsample([k]) # (K, b, 1)
        z_given_x_flat = z_given_x.reshape([k*batch_size, l])

        # Decoder
        out_decoder = self.decoder(z_given_x_flat)
        all_means_obs_model = out_decoder[..., :p] # (b*K, num_continuous_features)
        all_scales_obs_model = (
            nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
        ) # (b*K, num_continuous_features)

        # data, mask repeat
        data_flat = torch.Tensor.repeat(x[:, :p],[k, 1]).reshape(-1, 1) # (b*K*num_continuous_features, 1)
        tiledmask = torch.Tensor.repeat(mask[:, :p],[k, 1]) # (b*K, num_continuous_features)

        # continuous
        # x^m ~ p(x^m|z)
        pz = torch.distributions.Independent(
            torch.distributions.Normal(
                loc=torch.zeros(
                    len(all_means_obs_model.reshape(-1, 1))
                ).to(self.device), 
                scale = torch.ones(
                    len(all_means_obs_model.reshape(-1, 1))
                ).to(self.device)), 
            1,
        )
        
        z = pz.sample([1])
        cont_recon_x = all_means_obs_model.reshape(-1, 1) + z.reshape(-1, 1)*all_scales_obs_model.reshape(-1, 1)
        
        # discrete
        logit = out_decoder[:,(2*p):]
        
        st = 0
        end = 0
        disc_recon_x = []
        for i, n in enumerate(self.num_categories):
            end += n
            disc_x = F.one_hot(
                x[:, p+i].to(torch.long),
                num_classes=n
            ).to(torch.float32)
            
            disc_x = torch.Tensor.repeat(
                disc_x, [k, 1]
            )
            disc_x_out = F.gumbel_softmax(
                logit[:, st:end], hard=True
            ) # reconstruction
            disc_mask = torch.Tensor.repeat(
                mask[:, p+i],[k, ]
            )
            
            disc_x[~disc_mask] = disc_x_out[~disc_mask]
            disc_recon_x.append(disc_x)
            st = end
        
        disc_recon_x = torch.cat(disc_recon_x, dim=-1)
        
        # concatenation x^o, x^m 
        mask_flat = torch.Tensor.repeat(mask[:, :p], [k, 1]).reshape(-1, 1)
    
        recon_x = data_flat.clone()
        recon_x[~mask_flat] = cont_recon_x[~mask_flat] # imputing masking value in continuous
        recon_x = recon_x.reshape(-1, p)

        recon_x = torch.cat([recon_x, disc_recon_x], dim=-1) # x^m & x^o
        
        mask_logit = self.mask_layer(recon_x)
        
        # p(s|x^o,x^m)
        all_mask = torch.Tensor.repeat(mask, [k, 1])

        ps_given_xo_xm = Bernoulli(logits=mask_logit)
        log_ps_given_xo_xm = torch.sum(
            ps_given_xo_xm.log_prob(all_mask.to(torch.float32)),
            dim=-1
        ).reshape(k, batch_size)
        
        # log p(x^o|z)
        all_log_pxo_given_z_flat = Normal(
            loc=all_means_obs_model.reshape(-1, 1),
            scale=all_scales_obs_model.reshape(-1, 1),
        ).log_prob(data_flat) # (b*K*num_continuous_features, 1)
        
        all_log_pxo_given_z = all_log_pxo_given_z_flat.reshape(batch_size*k, p)
        
        log_pxo_given_z = torch.sum(
            all_log_pxo_given_z*tiledmask, axis=1
        ).reshape(k, batch_size)
        
        prior = generate_prior(self.config, self.device)
        log_pz = prior.log_prob(z_given_x)
        log_q = qz_given_xo.log_prob(z_given_x)
        
        neg_bound = -torch.mean(
            torch.logsumexp(
                log_ps_given_xo_xm + log_pxo_given_z + log_pz - log_q, axis=0
            )
        )
        
        disc_mask = mask[:, p:]
        disc_mask = torch.Tensor.repeat(disc_mask, [k, 1])
        disc_loss = torch.tensor(0.).to(self.device)
        
        st = 0
        end = 0
        for i, n in enumerate(self.num_categories):
            end += n
            target = x[: ,p+i].to(torch.long)
            target = torch.Tensor.repeat(target,[k, 1]).reshape(-1)
            
            if sum(disc_mask[:,i]) == 0: continue 
            disc_loss += F.cross_entropy(
                logit[:,st:end][~disc_mask[:, i]], target[~disc_mask[:, i]] ## complete?
            )
            st = end
        
        return neg_bound, disc_loss
    
    def impute(self, train_dataset, M, seed=0):
        set_random_seed(seed)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False)
        
        p = self.num_continuous_features
        l = self.latent_dim
        M = self.config["M"]
        
        imputed = []
        for batch in train_dataloader:
            with torch.no_grad():
                batch_size = len(batch)
                mask = (~torch.isnan(batch)).to(torch.float32).to(self.device)
                batch = batch.to(self.device)
                batch[torch.isnan(batch)] = 0
            
                out_encoder = self.encoder(batch)
                qz_given_xo = torch.distributions.Independent(
                    Normal(
                        loc=out_encoder[..., :l],
                        scale=nn.Softplus()(out_encoder[..., l:(2*l)])+ 0.001
                    ), 
                    1,
                )
            
                z_given_x = qz_given_xo.rsample([M]) # (K, b, 1)
                z_given_x_flat = z_given_x.reshape([-1, l])
                
                out_decoder = self.decoder(z_given_x_flat)
                all_means_obs_model = out_decoder[..., :p] # (b*K, num_continuous_features)
                all_scales_obs_model = (
                    nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
                ) # (b*K, num_continuous_features)

                logit = out_decoder[:, (2*p):]
                
                x_given_z = torch.distributions.Independent(
                    Normal(
                        loc=all_means_obs_model,
                        scale=all_scales_obs_model),
                    1,
                )
                
                disc_xms = []
                st = 0
                end = 0
                for i, n in enumerate(self.num_categories):
                    end += n
                    disc_xms.append(
                        Categorical(logits=logit[:,st:end]).sample()
                    )
                    st = end
                    
                disc_xms = torch.stack(disc_xms).T.reshape([M, batch_size, -1])
                cont_xms = x_given_z.sample().reshape([M, batch_size, p])
                
                xms = torch.cat([cont_xms, disc_xms], dim=-1)
                
                batch = torch.Tensor.repeat(batch, [M, 1, 1])
                mask = torch.Tensor.repeat(mask, [M, 1, 1])
                mask = ~mask.to(torch.bool) # missing : 1
                
                batch[mask] = xms[mask]
                imputed.append(batch)
        
        M_data = torch.cat(imputed, dim=1)

        # multiple imputation
        if self.config["multiple"]:
            imputed = []
            for data in M_data: 
                data = pd.DataFrame(data.cpu().numpy(), columns=train_dataset.features)
            
                """un-standardization of synthetic data"""
                for col, scaler in train_dataset.scalers.items():
                    data[[col]] = scaler.inverse_transform(data[[col]])
                
                """post-process"""
                data[train_dataset.categorical_features] = data[train_dataset.categorical_features].astype(int)
                data[train_dataset.integer_features] = data[train_dataset.integer_features].round(0).astype(int)
                imputed.append(data)

        # single imputation
        else:
            cont_imputed = torch.mean(
                M_data[:, :, : train_dataset.EncodedInfo.num_continuous_features], dim=0
            )
            disc_imputed = torch.mode(
                M_data[:, :, train_dataset.EncodedInfo.num_continuous_features:], dim=0
            )[0]
            imputed = torch.cat([cont_imputed, disc_imputed], dim=1)
            imputed= pd.DataFrame(imputed.cpu().numpy(), columns=train_dataset.features)
            
            """un-standardization of synthetic data"""
            for col, scaler in train_dataset.scalers.items():
                imputed[[col]] = scaler.inverse_transform(imputed[[col]])
            
            """post-process"""
            imputed[train_dataset.categorical_features] = imputed[train_dataset.categorical_features].astype(int)
            imputed[train_dataset.integer_features] = imputed[train_dataset.integer_features].round(0).astype(int)
        
        return imputed
# %%