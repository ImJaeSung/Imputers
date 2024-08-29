# current implementation: only support numerical values
import random
import numpy as np
import torch, os
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import argparse

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

class MaskEmbed(nn.Module):
    """ record to mask embedding
    """
    def __init__(self, rec_len=25, embed_dim=64, norm_layer=None):
        
        super().__init__()
        self.rec_len = rec_len
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class ActiveEmbed(nn.Module):
    """ record to mask embedding
    """
    def __init__(self, rec_len=25, embed_dim=64, norm_layer=None):
        
        super().__init__()
        self.rec_len = rec_len
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = torch.sin(x)
        x = x.transpose(1, 2)
        #   x = torch.cat((torch.sin(x), torch.cos(x + math.pi/2)), -1)
        x = self.norm(x)
        return x



def get_1d_sincos_pos_embed(embed_dim, pos, cls_token=False):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(pos)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def adjust_learning_rate(optimizer, epoch, lr, min_lr, max_epochs, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        tmp_lr = lr * epoch / warmup_epochs 
    else:
        tmp_lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = tmp_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = tmp_lr
    return tmp_lr


def get_grad_norm_(parameters, norm_type: float = 5.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == np.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScaler:

    state_dict_key = "amp_scaler"
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--max_epochs', default=600, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--embed_dim', default=32, type=int, help='embedding dimensions')
    parser.add_argument('--depth', default=6, type=int, help='encoder depth')
    parser.add_argument('--decoder_depth', default=4, type=int, help='decoder depth')
    parser.add_argument('--num_heads', default=4, type=int, help='number of heads')
    parser.add_argument('--mlp_ratio', default=4., type=float, help='mlp ratio')
    parser.add_argument('--encode_func', default='linear', type=str, help='encoding function')

    parser.add_argument('--norm_field_loss', default=False,
                        help='Use (per-patch) normalized field as targets for computing loss')
    parser.set_defaults(norm_field_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--pin_mem', action='store_false')

    # distributed training parameters
    return parser

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
