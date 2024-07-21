"""
Reference:
[1] https://github.com/tigvarts/vaeac/blob/master/train_utils.py
[2] https://github.com/tigvarts/vaeac/blob/master/nn_utils.py
[3] https://github.com/tigvarts/vaeac/blob/master/prob_utils.py
"""
#%%
import os
import re
import random
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from torch.nn.functional import softplus, softmax
#%%
"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 대한 시드 고정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # NumPy 시드 고정
    np.random.seed(seed)
    random.seed(seed) 
#%%
def extend_batch(batch, dataloader, batch_size):
    """
    If the batch size is less than batch_size, extends it with
    data from the dataloader until it reaches the required size.
    Here batch is a tensor.
    Returns the extended batch.
    """
    while batch.shape[0] != batch_size:
        dataloader_iterator = iter(dataloader)
        nw_batch = next(dataloader_iterator)
        if nw_batch.shape[0] + batch.shape[0] > batch_size:
            nw_batch = nw_batch[:batch_size - batch.shape[0]]
        batch = torch.cat([batch, nw_batch], 0)
    return batch


def extend_batch_tuple(batch, dataloader, batch_size):
    """
    The same as extend_batch, but here the batch is a list of tensors
    to be extended. All tensors are assumed to have the same first dimension.
    Returns the extended batch (i. e. list of extended tensors).
    """
    while batch[0].shape[0] != batch_size:
        dataloader_iterator = iter(dataloader)
        nw_batch = next(dataloader_iterator)
        if nw_batch[0].shape[0] + batch[0].shape[0] > batch_size:
            nw_batch = [nw_t[:batch_size - batch[0].shape[0]]
                        for nw_t in nw_batch]
        batch = [torch.cat([t, nw_t], 0) for t, nw_t in zip(batch, nw_batch)]
    return batch


def get_validation_iwae(val_dataloader, mask_generator, batch_size,
                        model, num_samples, verbose=False):
    """
    Compute mean IWAE log likelihood estimation of the validation set.
    Takes validation dataloader, mask generator, batch size, model (VAEAC)
    and number of IWAE latent samples per object.
    Returns one float - the estimation.
    """
    cum_size = 0
    avg_iwae = 0
    iterator = val_dataloader
    if verbose:
        iterator = tqdm(iterator)
    for batch in iterator:
        init_size = batch.shape[0]
        batch = extend_batch(batch, val_dataloader, batch_size)
        mask = mask_generator(batch)

        if next(model.parameters()).is_cuda:
            batch = batch.cuda()
            mask = mask.cuda()
        with torch.no_grad():
 
            iwae = model.batch_iwae(batch, mask, num_samples)[:init_size]
            avg_iwae = (avg_iwae * (cum_size / (cum_size + iwae.shape[0])) +
                        iwae.sum() / (cum_size + iwae.shape[0]))
            cum_size += iwae.shape[0]
        if verbose:
            iterator.set_description('Validation IWAE: %g' % avg_iwae)
    return float(avg_iwae)

#%%
class ResBlock(nn.Module):
    """
    Usual full pre-activation ResNet bottleneck block.
    For more information see
    He, K., Zhang, X., Ren, S., & Sun, J. (2016, October).
    Identity mappings in deep residual networks.
    European Conference on Computer Vision (pp. 630-645).
    Springer, Cham.
    ArXiv link: https://arxiv.org/abs/1603.05027
    """

    def __init__(self, outer_dim, inner_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(outer_dim),
            nn.LeakyReLU(),
            nn.Conv2d(outer_dim, inner_dim, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, outer_dim, 1),
        )

    def forward(self, input):
        return input + self.net(input)


class SkipConnection(nn.Module):
    """
    Skip-connection over the sequence of layers in the constructor.
    The module passes input data sequentially through these layers
    and then adds original data to the result.
    """
    def __init__(self, *args):
        super().__init__()
        self.inner_net = nn.Sequential(*args)

    def forward(self, input):
        return input + self.inner_net(input)


class MemoryLayer(nn.Module):
    """
    If output=False, this layer stores its input in a static class dictionary
    `storage` with the key `id` and then passes the input to the next layer.
    If output=True, this layer takes stored tensor from a static storage.
    If add=True, it returns sum of the stored vector and an input,
    otherwise it returns their concatenation.
    If the tensor with specified `id` is not in `storage` when the layer
    with output=True is called, it would cause an exception.

    The layer is used to make skip-connections inside nn.Sequential network
    or between several nn.Sequential networks without unnecessary code
    complication.
    The usage pattern is
    ```
        net1 = nn.Sequential(
            MemoryLayer('#1'),
            MemoryLayer('#0.1'),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            MemoryLayer('#0.1', output=True, add=False),
            # here add cannot be True because the dimensions mismatch
            nn.Linear(768, 256),
            # the dimension after the concatenation with skip-connection
            # is 512 + 256 = 768
        )
        net2 = nn.Sequential(
            nn.Linear(512, 512),
            MemoryLayer('#1', output=True, add=True),
            ...
        )
        b = net1(a)
        d = net2(c)
        # net2 must be called after net1,
        # otherwise tensor '#1' will not be in `storage`
    ```
    """

    storage = {}

    def __init__(self, id, output=False, add=False):
        super().__init__()
        self.id = id
        self.output = output
        self.add = add

    def forward(self, input):
        if not self.output:
            self.storage[self.id] = input
            return input
        else:
            if self.id not in self.storage:
                err = 'MemoryLayer: id \'%s\' is not initialized. '
                err += 'You must execute MemoryLayer with the same id '
                err += 'and output=False before this layer.'
                raise ValueError(err)
            stored = self.storage[self.id]
            if not self.add:
                data = torch.cat([input, stored], 1)
            else:
                data = input + stored
            return data
#%%
def normal_parse_params(params, min_sigma=0):
    """
    Take a Tensor (e. g. neural network output) and return
    torch.distributions.Normal distribution.
    This Normal distribution is component-wise independent,
    and its dimensionality depends on the input shape.
    First half of channels is mean of the distribution,
    the softplus of the second half is std (sigma), so there is
    no restrictions on the input tensor.

    min_sigma is the minimal value of sigma. I. e. if the above
    softplus is less than min_sigma, then sigma is clipped
    from below with value min_sigma. This regularization
    is required for the numerical stability and may be considered
    as a neural network architecture choice without any change
    to the probabilistic model.
    """
    n = params.shape[0]
    d = params.shape[1]
    mu = params[:, :d // 2]
    sigma_params = params[:, d // 2:]
    sigma = softplus(sigma_params)
    sigma = sigma.clamp(min=min_sigma)
    distr = Normal(mu, sigma)
    return distr


def categorical_parse_params_column(params, min_prob=0):
    """
    Take a Tensor (e. g. a part of neural network output) and return
    torch.distributions.Categorical distribution.
    The input tensor after applying softmax over the last axis contains
    a batch of the categorical probabilities. So there are no restrictions
    on the input tensor.

    Technically, this function treats the last axis as the categorical
    probabilities, but Categorical takes only 2D input where
    the first axis is the batch axis and the second one corresponds
    to the probabilities, so practically the function requires 2D input
    with the batch of probabilities for one categorical feature.

    min_prob is the minimal probability for each class.
    After clipping the probabilities from below they are renormalized
    in order to be a valid distribution. This regularization
    is required for the numerical stability and may be considered
    as a neural network architecture choice without any change
    to the probabilistic model.
    """
    params = softmax(params, -1)
    params = params.clamp(min_prob)
    params = params / params.sum(-1, keepdim=True)
    distr = Categorical(probs=params)
    return distr


class GaussianLoss(nn.Module):
    """
    Compute reconstruction log probability of groundtruth given
    a tensor of Gaussian distribution parameters and a mask.
    Gaussian distribution parameters are output of a neural network
    without any restrictions, the minimal sigma value is clipped
    from below to min_sigma (default: 1e-2) in order not to overfit
    network on some exact pixels.

    The first half of channels corresponds to mean, the second half
    corresponds to std. See normal_parse_parameters for more info.
    This layer doesn't work with NaNs in the data, it is used for
    inpainting. Roughly speaking, this loss is similar to L2 loss.
    Returns a vector of log probabilities for each object of the batch.
    """
    def __init__(self, min_sigma=1e-2):
        super().__init__()
        self.min_sigma = min_sigma

    def forward(self, groundtruth, distr_params, mask):
        distr = normal_parse_params(distr_params, self.min_sigma)
        log_probs = distr.log_prob(groundtruth) * mask
        return log_probs.view(groundtruth.shape[0], -1).sum(-1)


class GaussianCategoricalLoss(nn.Module):
    """
    This layer computes log probability of groundtruth for each object
    given the mask and the distribution parameters.
    This layer works for the cases when the dataset contains both
    real-valued and categorical features.

    one_hot_max_sizes[i] is the one-hot max size of i-th feature,
    if i-th feature is categorical, and 0 or 1 if i-th feature is real-valued.
    In the first case the distribution over feature is categorical,
    in the second case it is Gaussian.

    For example, if one_hot_max_sizes is [4, 1, 1, 2], then the distribution
    parameters for one object is the vector
    [p_00, p_01, p_02, p_03, mu_1, sigma_1, mu_2, sigma_2, p_30, p_31],
    where Softmax([p_00, p_01, p_02, p_03]) and Softmax([p_30, p_31])
    are probabilities of the first and the fourth feature categories
    respectively in the model generative distribution, and
    Gaussian(mu_1, sigma_1 ^ 2) and Gaussian(mu_2, sigma_2 ^ 2) are
    the model generative distributions on the second and the third features.

    For the definitions of min_sigma and min_prob see normal_parse_params
    and categorical_parse_params docs.

    This layer works correctly with missing values in groundtruth
    which are represented by NaNs.

    This layer works with 2D inputs only.
    """
    def __init__(self, one_hot_max_sizes, min_sigma=1e-4, min_prob=1e-4):
        super().__init__()
        self.one_hot_max_sizes = one_hot_max_sizes
        self.min_sigma = min_sigma
        self.min_prob = min_prob

    def forward(self, groundtruth, distr_params, mask):
        cur_distr_col = 0
        log_prob = []
        for i, size in enumerate(self.one_hot_max_sizes):
            # for i-th feature
            if size <= 1:
                # Gaussian distribution

                # select groundtruth, mask and distr_params for i-th feature
                groundtruth_col = groundtruth[:, i: i + 1]
                mask_col = mask[:, i: i + 1]
                params = distr_params[:, cur_distr_col: cur_distr_col + 2]
                cur_distr_col += 2

                # generative model distribution for the feature
                distr = normal_parse_params(params, self.min_sigma)

                # copy groundtruth column, so that zeroing nans will not
                # affect the original data
                gt_col_nansafe = torch.tensor(groundtruth_col)
                nan_mask = torch.isnan(groundtruth_col) ## real nan 
                gt_col_nansafe[nan_mask] = 0

                # compute the mask of the values
                # which we consider in the log probability
                mask_col = mask_col * (1 - nan_mask.float())

                col_log_prob = distr.log_prob(gt_col_nansafe) * mask_col
            else:
                # categorical distribution

                # select groundtruth, mask and distr_params for i-th feature
                groundtruth_col = groundtruth[:, i]
                mask_col = mask[:, i]
                params = distr_params[:, cur_distr_col: cur_distr_col + size]
                cur_distr_col += size

                # generative model distribution for the feature
                distr = categorical_parse_params_column(params, self.min_prob)

                # copy groundtruth column, so that zeroing nans will not
                # affect the original data
                gt_col_nansafe = torch.tensor(groundtruth_col)
                nan_mask = torch.isnan(groundtruth_col)
                gt_col_nansafe[nan_mask] = 0

                # compute the mask of the values
                # which we consider in the log probability
                mask_col = mask_col * (1 - nan_mask.float())

                col_log_prob = distr.log_prob(gt_col_nansafe) * mask_col
                col_log_prob = col_log_prob[:, None]

            # append the column of log probabilities for the i-th feature
            # (or zeros if the values is missed or masked) into log_prob list
            log_prob.append(col_log_prob)

        return torch.cat(log_prob, 1).sum(-1)


class CategoricalToOneHotLayer(nn.Module):
    """
    This layer expands categorical features into one-hot vectors, because
    multi-layer perceptrons are known to work better with this data
    representation. It also replaces NaNs with zeros in order so that
    further layers may work correctly.

    one_hot_max_sizes[i] is the one-hot max size of i-th feature,
    if i-th feature is categorical, and 0 or 1 if i-th feature is real-valued.

    add_nan_maps_for_columns is an optional list which contains
    indices of columns which isnan masks are to be appended
    to the result tensor. This option is necessary for proposal
    network to distinguish whether value is to be reconstructed or not.
    """
    def __init__(self, one_hot_max_sizes, add_nans_map_for_columns=[]):
        super().__init__()
        self.one_hot_max_sizes = one_hot_max_sizes
        self.add_nans_map_for_columns = add_nans_map_for_columns

    def forward(self, input):
        out_cols = []
        for i, size in enumerate(self.one_hot_max_sizes):
            if size <= 1:
                # real-valued feature
                # just copy it and replace NaNs with zeros
                out_col = input[:, i: i + 1]
                nan_mask = torch.isnan(out_col)
                out_col[nan_mask] = 0
            else:
                # categorical feature
                # replace NaNs with zeros
                cat_idx = torch.tensor(input[:, i])
                nan_mask = torch.isnan(cat_idx)
                cat_idx[nan_mask] = 0

                # one-hot encoding
                n = input.shape[0]
                out_col = torch.zeros(n, size, device=input.device)

                out_col[torch.arange(n).long(), cat_idx.long()] = 1

                # set NaNs to be zero vectors
                out_col[nan_mask] = 0

                # reshape nan_mask to be a column
                nan_mask = nan_mask[:, None]

            # append this feature column to the result
            out_cols.append(out_col)

            # if necessary, append isnan mask of this feature to the result
            if i in self.add_nans_map_for_columns:
                out_cols.append(nan_mask.float())

        return torch.cat(out_cols, 1)


class GaussianCategoricalSampler(nn.Module):
    """
    Generates a sample from the generative distribution defined by
    the output of the neural network.

    one_hot_max_sizes[i] is the one-hot max size of i-th feature,
    if i-th feature is categorical, and 0 or 1 if i-th feature is real-valued.

    The distribution parameters format, min_sigma and min_prob are described
    in docs for GaussianCategoricalLoss.

    If sample_most_probable is True, then the layer will return
    mean for Gaussians and the most probable class for categorical features.
    Otherwise the fair sampling procedure from Gaussian and categorical
    distributions is performed.
    """
    def __init__(self, one_hot_max_sizes, sample_most_probable=False,
                 min_sigma=1e-4, min_prob=1e-4):
        super().__init__()
        self.one_hot_max_sizes = one_hot_max_sizes
        self.sample_most_probable = sample_most_probable
        self.min_sigma = min_sigma
        self.min_prob = min_prob

    def forward(self, distr_params):
        cur_distr_col = 0
        sample = []
        for i, size in enumerate(self.one_hot_max_sizes):
            if size <= 1:
                # Gaussian distribution
                params = distr_params[:, cur_distr_col: cur_distr_col + 2]
                cur_distr_col += 2

                # generative model distribution for the feature
                distr = normal_parse_params(params, self.min_sigma)

                if self.sample_most_probable:
                    col_sample = distr.mean
                else:
                    col_sample = distr.sample()
            else:
                # categorical distribution
                params = distr_params[:, cur_distr_col: cur_distr_col + size]
                cur_distr_col += size

                # generative model distribution for the feature
                distr = categorical_parse_params_column(params, self.min_prob)

                if self.sample_most_probable:
                    col_sample = torch.max(distr.probs, 1)[1][:, None].float()
                else:
                    col_sample = distr.sample()[:, None].float()

            sample.append(col_sample)

        return torch.cat(sample, 1)


class SetGaussianSigmasToOne(nn.Module):
    """
    This layer is used in missing features imputation. Because the target
    metric for this problem is NRMSE, we set all sigma to one, so that
    the optimized metric is L2 loss without any disbalance between features,
    which probably increases (we are not sure about this) NRMSE score.

    one_hot_max_sizes[i] is the one-hot max size of i-th feature,
    if i-th feature is categorical, and 0 or 1 if i-th feature is real-valued.

    The distribution parameters format is described in docs
    for GaussianCategoricalLoss.

    Because the output of the network is passed through softplus
    for real-valued features, this layer replaces all corresponding
    columns with softplus^{-1}(1), where softplus^{-1} is the inverse
    softplus function.
    """
    def __init__(self, one_hot_max_sizes):
        super().__init__()
        self.one_hot_max_sizes = one_hot_max_sizes

    def forward(self, distr_params):
        cur_distr_col = 0
        # mask shows which columns must save their values
        mask = torch.ones(distr_params.shape[1], device=distr_params.device)
        for i, size in enumerate(self.one_hot_max_sizes):
            if size <= 1:
                mask[cur_distr_col + 1] = 0
                cur_distr_col += 2
            else:
                cur_distr_col += size

        inverse_softplus = torch.ones_like(distr_params) * 0.54125
        return distr_params * mask[None] + inverse_softplus * (1 - mask)[None]

#%%
def get_frequency(
    X_gt: pd.DataFrame, X_synth: pd.DataFrame, n_histogram_bins: int = 10
):
    """
    Reference:
    [1] https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics/_utils.py
    
    Get percentual frequencies for each possible real categorical value.

    Returns:
        The observed and expected frequencies (as a percent).
    """
    res = {}
    for col in X_gt.columns:
        local_bins = min(n_histogram_bins, len(X_gt[col].unique()))

        if len(X_gt[col].unique()) < 5:  # categorical
            gt = (X_gt[col].value_counts() / len(X_gt)).to_dict()
            synth = (X_synth[col].value_counts() / len(X_synth)).to_dict()
        else:
            gt_vals, bins = np.histogram(X_gt[col], bins=local_bins)
            synth_vals, _ = np.histogram(X_synth[col], bins=bins)
            gt = {k: v / (sum(gt_vals) + 1e-8) for k, v in zip(bins, gt_vals)}
            synth = {k: v / (sum(synth_vals) + 1e-8) for k, v in zip(bins, synth_vals)}

        for val in gt:
            if val not in synth or synth[val] == 0:
                synth[val] = 1e-11
        for val in synth:
            if val not in gt or gt[val] == 0:
                gt[val] = 1e-11

        if gt.keys() != synth.keys():
            raise ValueError(f"Invalid features. {gt.keys()}. syn = {synth.keys()}")
        res[col] = (list(gt.values()), list(synth.values()))

    return res
#%%
def marginal_plot(train, syndata, config, model_name):
    model_name = re.sub(".pth", "", model_name)
    if not os.path.exists(f"./assets/figs/{config['dataset']}/{model_name}/"):
        os.makedirs(f"./assets/figs/{config['dataset']}/{model_name}/")
    
    figs = []
    for idx, feature in tqdm(enumerate(train.columns), desc="Plotting Histograms..."):
        fig = plt.figure(figsize=(7, 4))
        fig, ax = plt.subplots(1, 1)
        sns.histplot(
            data=syndata,
            x=syndata[feature],
            stat='density',
            label='synthetic',
            ax=ax,
            bins=int(np.sqrt(len(syndata)))) 
        sns.histplot(
            data=train,
            x=train[feature],
            stat='density',
            label='train',
            ax=ax,
            bins=int(np.sqrt(len(train)))) 
        ax.legend()
        ax.set_title(f'{feature}', fontsize=15)
        plt.tight_layout()
        plt.savefig(f"./assets/figs/{config['dataset']}/{model_name}/hist_{re.sub(' ', '_', feature)}.png")
        # plt.show()
        plt.close()
        figs.append(fig)
    return figs