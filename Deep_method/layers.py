import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from scipy import optimize
from sklearn.linear_model.base import LinearModel, LinearClassifierMixin

from utils import spherical_kmeans, gaussian_filter_1d, normalize_, matrix_inverse_sqrt

#-----------------------------------------------------------------------------------------
#                                      Kernels
#-----------------------------------------------------------------------------------------
def exp(x, alpha):
    return torch.exp(alpha*(x - 1.))

def add_exp(x, alpha):
    return 0.5 * (exp(x, alpha) + x)


kernels = {
    "exp": exp,
    "add_exp": add_exp
}
#-----------------------------------------------------------------------------------------

class CKNLayer(nn.Conv1d):
    """https://gitlab.inria.fr/dchen/CKN-seq"""
    def __init__(self, in_channels, out_channels, filter_size,
                 padding=0, dilation=1, groups=1, subsampling=1,
                 kernel_func="exp", kernel_args=[0.5]):

        if padding == "SAME":
            padding = (filter_size - 1)//2
        else:
            padding = 0

        super(CKNLayer, self).__init__(in_channels, out_channels, filter_size,
                                       stride=1, padding=padding,
                                       dilation=dilation, groups=groups,
                                       bias=False)

        self.subsampling = subsampling
        self.filter_size = filter_size
        self.patch_dim = self.in_channels * self.filter_size

        self._need_lintrans_computed = True

        if isinstance(kernel_args, (int, float)):
            kernel_args = [kernel_args]
        if kernel_func == "exp" or kernel_func == "add_exp":
            kernel_args = [1./kernel_arg ** 2 for kernel_arg in kernel_args]
        self.kernel_args = kernel_args

        kernel_func = kernels[kernel_func]
        self.kappa = lambda x: kernel_func(x, *self.kernel_args)

        ones = torch.ones(1, self.in_channels // self.groups, self.filter_size)
        self.register_buffer("ones", ones)
        self.init_pooling_filter()

        self.register_buffer("lintrans",
                             torch.Tensor(out_channels, out_channels))

    def init_pooling_filter(self):
        if self.subsampling <= 1:
            return
        size = 2 * self.subsampling - 1
        pooling_filter = gaussian_filter_1d(size)
        pooling_filter = pooling_filter.expand(self.out_channels, 1, size)
        self.register_buffer("pooling_filter", pooling_filter)

    def train(self, mode=True):
        super(CKNLayer, self).train(mode)
        if self.training is True:
            self._need_lintrans_computed = True

    def _compute_lintrans(self):
        """Compute the linear transformation factor kappa(ZtZ)^(-1/2)
        Returns:
            lintrans: out_channels x out_channels
        """
        if not self._need_lintrans_computed:
            return self.lintrans

        lintrans = self.weight.view(self.out_channels, -1)
        lintrans = lintrans.mm(lintrans.t())
        lintrans = self.kappa(lintrans)
        lintrans = matrix_inverse_sqrt(lintrans)
        if not self.training:
            self._need_lintrans_computed = False
            self.lintrans.data = lintrans.data

        return lintrans

    def _conv_layer(self, x_in):
        """Convolution layer
        Compute x_out = ||x_in|| x kappa(Zt x_in/||x_in||)
        Args:
            x_in: batch_size x in_channels x H
            self.filters: out_channels x in_channels x filter_size
            x_out: batch_size x out_channels x (H - filter_size + 1)
        """
        patch_norm = torch.sqrt(F.conv1d(x_in.pow(2), self.ones,
                                padding=self.padding, dilation=self.dilation,
                                groups=self.groups))
        patch_norm = patch_norm.clamp(1e-4)
        x_out = super(CKNLayer, self).forward(x_in)
        x_out = x_out / patch_norm
        x_out = self.kappa(x_out)
        x_out = patch_norm * x_out
        return x_out

    def _mult_layer(self, x_in, lintrans):
        """Multiplication layer
        Compute x_out = kappa(ZtZ)^(-1/2) x x_in
        Args:
            x_in: batch_size x out_channels x H
            lintrans: out_channels x out_channels
            x_out: batch_size x out_channels x H
        """
        batch_size, out_c, _ = x_in.size()
        if x_in.dim() == 2:
            return torch.mm(x_in, lintrans)
        return torch.bmm(lintrans.expand(batch_size, out_c, out_c), x_in)

    def _pool_layer(self, x_in):
        """Pooling layer
        Compute I(z) = \sum_{z'} phi(z') x exp(-\beta_1 ||z'-z||_2^2)
        """
        if self.subsampling <= 1:
            return x_in
        x_out = F.conv1d(x_in, self.pooling_filter, stride=self.subsampling,
                         padding=self.subsampling-1, groups=self.out_channels)
        return x_out

    def forward(self, x_in):
        """Encode function for a CKN layer
        Args:
            x_in: batch_size x in_channels x H x W
        """
        x_out = self._conv_layer(x_in)
        x_out = self._pool_layer(x_out)
        lintrans = self._compute_lintrans()
        x_out = self._mult_layer(x_out, lintrans)
        return x_out

    def compute_mask(self, mask=None):
        if mask is None:
            return mask
        mask = mask.float().unsqueeze(1)
        mask = F.avg_pool1d(mask, kernel_size=self.filter_size,
                            stride=self.subsampling)
        mask = mask.squeeze(1) != 0
        return mask

    def extract_1d_patches(self, input, mask=None):
        output = input.unfold(-1, self.filter_size, 1).transpose(1, 2)
        output = output.contiguous().view(-1, self.patch_dim)
        if mask is not None:
            mask = mask.float().unsqueeze(1)
            mask = F.avg_pool1d(mask, kernel_size=self.filter_size, stride=1)
            # option 2: mask = mask.view(-1) == 1./self.filter_size
            mask = mask.view(-1) != 0
            output = output[mask]
        return output

    def sample_patches(self, x_in, mask=None, n_sampling_patches=1000):
        """Sample patches from the given Tensor
        Args:
            x_in (Tensor batch_size x in_channels x H)
            n_sampling_patches (int): number of patches to sample
        Returns:
            patches: (batch_size x (H - filter_size + 1)) x (in_channels x filter_size)
        """
        patches = self.extract_1d_patches(x_in, mask)
        n_sampling_patches = min(patches.size(0), n_sampling_patches)

        indices = torch.randperm(patches.size(0))[:n_sampling_patches]
        patches = patches[indices]
        normalize_(patches)
        return patches

    def unsup_train(self, patches):
        """Unsupervised training for a CKN layer
        Args:
            patches: n x in_channels x H
        Updates:
            filters: out_channels x in_channels x filter_size
        """
        weight = spherical_kmeans(patches, self.out_channels)
        weight = weight.view_as(self.weight)
        self.weight.data = weight.data
        self._need_lintrans_computed = True

    def normalize_(self):
        norm = self.weight.data.view(
            self.out_channels, -1).norm(p=2, dim=-1).view(-1, 1, 1)
        norm.clamp_(min=1e-4)
        self.weight.data.div_(norm)


class BioEmbedding(nn.Module):
    """https://gitlab.inria.fr/dchen/CKN-seq"""
    def __init__(self, num_embeddings, mask_zeros=False):
        """Embedding layer for biosequences
        Args:
            num_embeddings (int): number of letters in alphabet
        """
        super(BioEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.mask_zeros = mask_zeros

        self.embedding = lambda x, weight: F.embedding(x, weight)
        weight = self._make_weight()
        self.register_buffer("weight", weight)


    def _make_weight(self):
        weight = torch.zeros(self.num_embeddings + 1, self.num_embeddings)
        weight[0] = 1./self.num_embeddings
        weight[1:] = torch.diag(torch.ones(self.num_embeddings))
        return weight

    def compute_mask(self, x):
        """Compute the mask for the given Tensor
        """
        if self.mask_zeros:
            mask = (x != 0)
            return mask
        return None

    def forward(self, x):
        """
        Args:
            x: LongTensor of indices
        """
        x_out = self.embedding(x, self.weight)
        return x_out.transpose(1, 2).contiguous()


class GlobalAvg1D(nn.Module):
    """https://gitlab.inria.fr/dchen/CKN-seq"""
    def __init__(self):
        super(GlobalAvg1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=-1)
        mask = mask.float().unsqueeze(1)
        x = x * mask
        return x.sum(dim=-1)/mask.sum(dim=-1)

class RowPreprocessor(nn.Module):
    """https://gitlab.inria.fr/dchen/CKN-seq"""
    def __init__(self):
        super(RowPreprocessor, self).__init__()
        self.register_buffer("mean", None)
        self.register_buffer("var", None)
        self.register_buffer("scale", None)
        self.count = 0
        self.fitted = False

    def reset(self):
        self.mean = None
        self.var = None
        self.scale = None
        self.count = 0.
        self.fitted = False

    def forward(self, input):
        if not self.fitted:
            # self.partial_fit(input)
            return input
        input -= self.mean
        input /= self.scale
        return input

    def fit(self, input):
        self.mean = input.mean(dim=0)
        self.var = input.var(dim=0, unbiased=False)
        self.scale = self.var.sqrt()

    def fit_transform(self, input):
        self.fit(input)
        return self(input)

    def partial_fit(self, input):
        if self.count == 0.:
            self.mean = input.mean(0)
            self.var = input.var(0, unbiased=False)
            self.scale = self.var.sqrt()
            self.count += input.shape[0]
        else:
            last_sum = self.count * self.mean
            new_sum = input.sum(0)
            updated_count = self.count + input.shape[0]
            self.mean = (last_sum + new_sum) / updated_count

            new_unnorm_var = input.var(0, unbiased=False) * input.shape[0]
            last_unnorm_var = self.var * self.count
            last_over_new_count = self.count / input.shape[0]
            self.var = (
                new_unnorm_var + last_unnorm_var + 
                last_over_new_count / updated_count * 
                (last_sum / last_over_new_count - new_sum) ** 2) / updated_count
            self.count = updated_count
            self.scale = self.var.sqrt()

    def _load_from_state_dict(self, state_dict, prefix, metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        self.fitted = True
        for k, v in self._buffers.items():
            key = prefix + k
            setattr(self, k, state_dict[key])
        super(RowPreprocessor, self)._load_from_state_dict(
            state_dict, prefix, metadata,
            strict, missing_keys, unexpected_keys, error_msgs)


class LinearMax(nn.Linear, LinearModel, LinearClassifierMixin):
    """https://gitlab.inria.fr/dchen/CKN-seq"""
    def __init__(self, in_features, out_features, alpha=0.0, penalty="l2"):
        super(LinearMax, self).__init__(in_features, 128)
        self.alpha = alpha
        self.penalty = penalty
        self.extra_layer = nn.Linear(128, 64)
        self.extra_layer2 = nn.Linear(64, 32)
        self.extra_layer3 = nn.Linear(32, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input, proba=False):
        out = super(LinearMax, self).forward(input)
        out = self.dropout(out)
        out = self.activation(out)
        out = self.extra_layer(out)
        out = self.dropout(out)
        out = self.activation(out)
        out = self.extra_layer2(out)
        out = self.activation(out)
        out = self.extra_layer3(out)
        if proba:
            return out.sigmoid()
        return out

    def fit(self, x, y, criterion=None):
        use_cuda = self.weight.data.is_cuda
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()
        reduction = criterion.reduction
        criterion.reduction = 'sum'
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        def eval_loss(w):
            w = w.reshape((self.out_features, -1))
            if self.weight.grad is not None:
                self.weight.grad = None
            if self.bias is None:
                self.weight.data.copy_(torch.from_numpy(w))
            else:
                if self.bias.grad is not None:
                    self.bias.grad = None
                self.weight.data.copy_(torch.from_numpy(w[:, :-1]))
                self.bias.data.copy_(torch.from_numpy(w[:, -1]))
            y_pred = self(x).view(-1)
            loss = criterion(y_pred, y)
            loss.backward()
            if self.alpha != 0.0:
                if self.penalty == "l2":
                    penalty = 0.5 * self.alpha * torch.norm(self.weight)**2
                elif self.penalty == "l1":
                    penalty = self.alpha * torch.norm(self.weight, p=1)
                    penalty.backward()
                loss = loss + penalty
            return loss.item()

        def eval_grad(w):
            dw = self.weight.grad.data
            if self.alpha != 0.0:
                if self.penalty == "l2":
                    dw.add_(self.alpha, self.weight.data)
            if self.bias is not None:
                db = self.bias.grad.data
                dw = torch.cat((dw, db.view(-1, 1)), dim=1)
            return dw.cpu().numpy().ravel().astype("float64")

        w_init = self.weight.data
        if self.bias is not None:
            w_init = torch.cat((w_init, self.bias.data.view(-1, 1)), dim=1)
        w_init = w_init.cpu().numpy().astype("float64")

        w = optimize.fmin_l_bfgs_b(
            eval_loss, w_init, fprime=eval_grad, maxiter=100, disp=0)
        if isinstance(w, tuple):
            w = w[0]

        w = w.reshape((self.out_features, -1))
        self.weight.grad.data.zero_()
        if self.bias is None:
            self.weight.data.copy_(torch.from_numpy(w))
        else:
            self.bias.grad.data.zero_()
            self.weight.data.copy_(torch.from_numpy(w[:, :-1]))
            self.bias.data.copy_(torch.from_numpy(w[:, -1]))
        criterion.reduction = reduction

