"""
Reference:
[1] https://github.com/tigvarts/vaeac/blob/master/VAEAC.py
[2] https://github.com/tigvarts/vaeac/blob/master/imputation_networks.py
[3] https://github.com/tigvarts/vaeac/blob/master/impute.py
"""
#%%
from tqdm import tqdm
import math

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions import kl_divergence
#%%
from modules.mask_generators import MCARGenerator
from modules.utils import (
    MemoryLayer, 
    extend_batch,
    SkipConnection,
    normal_parse_params,
    CategoricalToOneHotLayer, 
    GaussianCategoricalLoss,
    GaussianCategoricalSampler, 
    SetGaussianSigmasToOne
)
#%%
class VAEAC(nn.Module):
    """
    Variational Autoencoder with Arbitrary Conditioning core model.
    It is rather flexible, but have several assumptions:
    + The batch of objects and the mask of unobserved features
      have the same shape.
    + The prior and proposal distributions in the latent space
      are component-wise independent Gaussians.
    The constructor takes
    + Prior and proposal network which take as an input the concatenation
      of the batch of objects and the mask of unobserved features
      and return the parameters of Gaussians in the latent space.
      The range of neural networks outputs should not be restricted.
    + Generative network takes latent representation as an input
      and returns the parameters of generative distribution
      p_theta(x_b | z, x_{1 - b}, b), where b is the mask
      of unobserved features. The information about x_{1 - b} and b
      can be transmitted to generative network from prior network
      through nn_utils.MemoryLayer. It is guaranteed that for every batch
      prior network is always executed before generative network.
    + Reconstruction log probability. rec_log_prob is a callable
      which takes (groundtruth, distr_params, mask) as an input
      and return vector of differentiable log probabilities
      p_theta(x_b | z, x_{1 - b}, b) for each object of the batch.
    + Sigma_mu and sigma_sigma are the coefficient of the regularization
      in the hidden space. The default values correspond to a very weak,
      almost disappearing regularization, which is suitable for all
      experimental setups the model was tested on.
    """
    def __init__(
            self,
            config,
            networks,
            device, 
            sigma_mu=1e4, 
            sigma_sigma=1e-4):
        super().__init__()
        self.config = config
        self.device = device    

        self.networks = networks
        self.rec_log_prob = networks['reconstruction_log_prob']
        self.proposal_network = networks['proposal_network']
        self.prior_network = networks['prior_network']
        self.generative_network = networks['generative_network']
        self.sigma_mu = sigma_mu
        self.sigma_sigma = sigma_sigma

    def make_observed(self, batch, mask):
        """
        Copy batch of objects and zero unobserved features.
        """
        observed = torch.tensor(batch)
        observed[mask.byte()] = 0
        return observed

    def make_latent_distributions(self, batch, mask, no_proposal=False):
        """
        Make latent distributions for the given batch and mask.
        No no_proposal is True, return None instead of proposal distribution.
        """

        observed = self.make_observed(batch, mask)
        if no_proposal:
            proposal = None
        else:

            full_info = torch.cat([batch, mask], 1)
            proposal_params = self.proposal_network(full_info)
            proposal = normal_parse_params(proposal_params, 1e-3)

        prior_params = self.prior_network(torch.cat([observed, mask], 1))
        prior = normal_parse_params(prior_params, 1e-3)

        return proposal, prior

    def prior_regularization(self, prior): ### regulariazation 굳이 없어도 ?
        """
        The prior distribution regularization in the latent space.
        Though it saves prior distribution parameters from going to infinity,
        the model usually doesn't diverge even without this regularization.
        It almost doesn't affect learning process near zero with default
        regularization parameters which are recommended to be used.
        """
        num_objects = prior.mean.shape[0]
        mu = prior.mean.view(num_objects, -1)
        sigma = prior.scale.view(num_objects, -1)
        mu_regularizer = -(mu ** 2).sum(-1) / 2 / (self.sigma_mu ** 2)
        sigma_regularizer = (sigma.log() - sigma).sum(-1) * self.sigma_sigma
        return mu_regularizer + sigma_regularizer

    def batch_vlb(self, batch, mask):
        """
        Compute differentiable lower bound for the given batch of objects
        and mask.
        """
        proposal, prior = self.make_latent_distributions(batch, mask)
        prior_regularization = self.prior_regularization(prior)
        latent = proposal.rsample()
        rec_params = self.generative_network(latent)
        rec_loss = self.rec_log_prob(batch, rec_params, mask)
        kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).sum(-1)
        return rec_loss - kl + prior_regularization

    def batch_iwae(self, batch, mask, K):
        """
        Compute IWAE log likelihood estimate with K samples per object.
        Technically, it is differentiable, but it is recommended to use it
        for evaluation purposes inside torch.no_grad in order to save memory.
        With torch.no_grad the method almost doesn't require extra memory
        for very large K.
        The method makes K independent passes through generator network,
        so the batch size is the same as for training with batch_vlb.
        """
        proposal, prior = self.make_latent_distributions(batch, mask)
        estimates = []
        for i in range(K):
            latent = proposal.rsample()

            rec_params = self.generative_network(latent)
            rec_loss = self.rec_log_prob(batch, rec_params, mask)

            prior_log_prob = prior.log_prob(latent)
            prior_log_prob = prior_log_prob.view(batch.shape[0], -1)
            prior_log_prob = prior_log_prob.sum(-1)

            proposal_log_prob = proposal.log_prob(latent)
            proposal_log_prob = proposal_log_prob.view(batch.shape[0], -1)
            proposal_log_prob = proposal_log_prob.sum(-1)

            estimate = rec_loss + prior_log_prob - proposal_log_prob
            estimates.append(estimate[:, None])

        return torch.logsumexp(torch.cat(estimates, 1), 1) - math.log(K)

    def generate_samples_params(self, batch, mask, K=1):
        """
        Generate parameters of generative distributions for samples
        from the given batch.
        It makes K latent representation for each object from the batch
        and generate samples from them.
        The second axis is used to index samples for an object, i. e.
        if the batch shape is [n x D1 x D2], then the result shape is
        [n x K x D1 x D2].
        It is better to use it inside torch.no_grad in order to save memory.
        With torch.no_grad the method doesn't require extra memory
        except the memory for the result.
        """
        _, prior = self.make_latent_distributions(batch, mask)
        samples_params = []
        for i in range(K):
            latent = prior.rsample()
            sample_params = self.generative_network(latent)
            samples_params.append(sample_params.unsqueeze(1))
        return torch.cat(samples_params, 1)

    def generate_reconstructions_params(self, batch, mask, K=1):
        """
        Generate parameters of generative distributions for reconstructions
        from the given batch.
        It makes K latent representation for each object from the batch
        and generate samples from them.
        The second axis is used to index samples for an object, i. e.
        if the batch shape is [n x D1 x D2], then the result shape is
        [n x K x D1 x D2].
        It is better to use it inside torch.no_grad in order to save memory.
        With torch.no_grad the method doesn't require extra memory
        except the memory for the result.
        """
        _, prior = self.make_latent_distributions(batch, mask)
        reconstructions_params = []
        for i in range(K):
            latent = prior.rsample()
            rec_params = self.generative_network(latent)
            reconstructions_params.append(rec_params.unsqueeze(1))
        return torch.cat(reconstructions_params, 1)
    
    # impute missing values for all input data
    def impute(self, train_dataset, M, seed=0):
        torch.random.manual_seed(seed)

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config["batch_size"],
            shuffle=False, 
            num_workers=0,
            drop_last=False
        )

        # prepare the store for the imputations
        results = []
        for _ in range(M):
            results.append([])

        # impute missing values for all input data
        for batch in tqdm(train_dataloader, desc="imputing..."):
            # if batch size is less than batch_size, extend it with objects
            # from the beginning of the dataset
            batch_extended = torch.tensor(batch)
            batch_extended = extend_batch(
                batch_extended, train_dataloader, self.config["batch_size"]
            )
            
            batch = batch.to(self.device)
            batch_extended = batch_extended.to(self.device)
            mask_extended = torch.isnan(batch_extended).float().to(self.device)

            with torch.no_grad():
                samples_params = self.generate_samples_params(
                    batch_extended, mask_extended, M
                ) # [B, M, D]

                samples_params = samples_params[:batch.shape[0]]
                
            # make a copy of batch with zeroed missing values
            mask = torch.isnan(batch)
            batch_zeroed_nans = torch.tensor(batch)
            batch_zeroed_nans[mask] = 0
            
            mask = mask.to(self.device)
            batch_zeroed_nans = batch_zeroed_nans.to(self.device)
            
            # impute samples from the generative distributions into the data
            # and save it to the results
            
            for i in range(M):
                sample_params = samples_params[:, i]
                sample = self.networks['sampler'](sample_params)
                sample[(1. - mask.long()).byte()] = 0
                sample += batch_zeroed_nans
                results[i].append(torch.tensor(sample, device='cpu'))
        
        # concatenate all batches into one [n x M x D] tensor,
        # where n in the number of objects, M is the number of imputations
        # and D is the dimensionality of one object
        for i in range(len(results)):
            results[i] = torch.cat(results[i]).unsqueeze(1)
        result = torch.cat(results, 1)
        result.shape

        imputed = []
        for i in range(M):
            imputed_sample = result[:, i, :]
            imputed_sample *= train_dataset.norm_std[None] 
            imputed_sample += train_dataset.norm_mean[None]
            # imputed_sample = pd.DataFrame(imputed_sample, columns=train_dataset.features)

            imputed_sample = pd.DataFrame(imputed_sample, columns=train_dataset.features)
            imputed_sample[train_dataset.categorical_features] = imputed_sample[train_dataset.categorical_features].astype(int)
            imputed_sample[train_dataset.integer_features] = imputed_sample[train_dataset.integer_features].round(0).astype(int)
            
            imputed.append(
                torch.tensor(imputed_sample.values, dtype=torch.float32)
            )

        assert len(imputed) == M
        
        return imputed
#%%
def get_imputation_networks(one_hot_max_sizes):
    """
    This function builds neural networks for imputation given
    the list of one-hot max sizes of the dataset features.
    
    It returns a dictionary with those neural networks together with
    reconstruction log probability function, optimizer constructor,
    sampler from the generator output, mask generator, batch size,
    and scale factor for the stability of the variational lower bound
    optimization.
    """

    #TODO: model size 
    width = 256
    depth = 10
    latent_dim = 64
    
    # Proposal network
    proposal_layers = [
        CategoricalToOneHotLayer(
            one_hot_max_sizes + [0] * len(one_hot_max_sizes),
            list(range(len(one_hot_max_sizes)))
        ),
        
        nn.Linear(
            sum(max(1, x) for x in one_hot_max_sizes) + len(one_hot_max_sizes) * 2,
            width
        ),

        nn.LeakyReLU(),
    ]

    for i in range(depth):
        proposal_layers.append(
            SkipConnection(
                nn.Linear(width, width),
                nn.LeakyReLU(),
            )
        )
    proposal_layers.append(
        nn.Linear(width, latent_dim * 2)
    )
    proposal_network = nn.Sequential(*proposal_layers)

    # Prior network
    prior_layers = [
        CategoricalToOneHotLayer(
            one_hot_max_sizes + [0] * len(one_hot_max_sizes)
        ),
        
        MemoryLayer('#input'),
        
        nn.Linear(
            sum(max(1, x) for x in one_hot_max_sizes) + len(one_hot_max_sizes),
            width
        ),

        nn.LeakyReLU(),
    ]

    for i in range(depth):
        prior_layers.append(
            SkipConnection(
                # skip-connection from prior network to generative network
                MemoryLayer('#%d' % i),
                nn.Linear(width, width),
                nn.LeakyReLU(),
            )
        )

    prior_layers.extend(
        [MemoryLayer('#%d' % depth), nn.Linear(width, latent_dim * 2),]
    )
    prior_network = nn.Sequential(*prior_layers)

    # Generative network
    generative_layers = [
        nn.Linear(64, 256),
        nn.LeakyReLU(),
    ]
    for i in range(depth + 1):
        generative_layers.append(
            SkipConnection(
                # skip-connection from prior network to generative network
                MemoryLayer('#%d' % (depth - i), True),
                nn.Linear(width * 2, width),
                nn.LeakyReLU(),
            )
        )

    generative_layers.extend(
        [MemoryLayer('#input', True),
        nn.Linear(
            width + sum(max(1, x) for x in one_hot_max_sizes) + len(one_hot_max_sizes), sum(max(2, x) for x in one_hot_max_sizes)
        ),
        SetGaussianSigmasToOne(one_hot_max_sizes),]
    )
    generative_network = nn.Sequential(*generative_layers)

    return {
        # 'batch_size': 64,

        'reconstruction_log_prob': GaussianCategoricalLoss(one_hot_max_sizes),

        'sampler': GaussianCategoricalSampler(one_hot_max_sizes,
                                              sample_most_probable=True),

        'vlb_scale_factor': 1 / len(one_hot_max_sizes),

        # 'optimizer': lambda parameters: Adam(parameters, lr=3e-4),

        'mask_generator': MCARGenerator(0.2),

        'proposal_network': proposal_network,

        'prior_network': prior_network,

        'generative_network': generative_network,
    }