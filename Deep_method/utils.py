import math
import random
import numpy as np
import torch
from torch.autograd import Variable


class MatrixInverseSqrt(torch.autograd.Function):
    """Matrix inverse square root for a symmetric definite positive matrix
    """
    @staticmethod
    def forward(ctx, input, eps=1e-2):
        use_cuda = input.is_cuda
        input = input.cpu()
        e, v = torch.symeig(input, eigenvectors=True)
        if use_cuda:
            e = e.cuda()
            v = v.cuda()
        e = e.clamp(min=0)
        e_sqrt = e.sqrt_().add_(eps)
        ctx.e_sqrt = e_sqrt
        ctx.v = v
        e_rsqrt = e_sqrt.reciprocal()

        output = v.mm(torch.diag(e_rsqrt).mm(v.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        e_sqrt, v = Variable(ctx.e_sqrt), Variable(ctx.v)
        ei = e_sqrt.expand_as(v)
        ej = e_sqrt.view([-1, 1]).expand_as(v)
        f = torch.reciprocal((ei + ej) * ei * ej)
        grad_input = -v.mm((f*(v.t().mm(grad_output.mm(v)))).mm(v.t()))
        return grad_input, None


def matrix_inverse_sqrt(input, eps=1e-2):
    """Wrapper for MatrixInverseSqrt"""
    return MatrixInverseSqrt.apply(input, eps)

def gaussian_filter_1d(size, sigma=None):
    """Create 1D Gaussian filter
    """
    if size == 1:
        return torch.ones(1)
    if sigma is None:
        sigma = (size - 1.)/(2.*math.sqrt(2))
    m = float((size - 1) // 2)
    filt = torch.arange(-m, m+1)
    filt = torch.exp(-filt.pow(2)/(2.*sigma*sigma))
    return filt/torch.sum(filt)


def spherical_kmeans(x, n_clusters, max_iters=100, verbose=True, eps=1e-4):
    """Spherical kmeans
    Args:
        x (Tensor n_samples x n_features): data points
        n_clusters (int): number of clusters
    """
    use_cuda = x.is_cuda
    n_samples, n_features = x.size()
    
    indices = torch.randperm(n_samples)[:n_clusters]
    if use_cuda:
        indices = indices.cuda()
    clusters = x[indices]

    prev_sim = np.inf

    for n_iter in range(max_iters):
        # assign data points to clusters
        cos_sim = x.mm(clusters.t())
        tmp, assign = cos_sim.max(dim=-1)
        sim = tmp.mean()
        if (n_iter + 1) % 10 == 0 and verbose:
            print("Spherical kmeans iter {}, objective value {}".format(
                n_iter + 1, sim))

        # update clusters
        for j in range(n_clusters):
            index = assign == j
            if index.sum() == 0:
                # clusters[j] = x[random.randrange(n_samples)]
                idx = tmp.argmin()
                clusters[j] = x[idx]
                tmp[idx] = 1
            else:
                xj = x[index]
                c = xj.mean(0)
                clusters[j] = c / c.norm()

        if np.abs(prev_sim - sim)/(np.abs(sim)+1e-20) < 1e-6:
            break
        prev_sim = sim
    return clusters


def normalize_(x, p=2, dim=-1):
    norm = x.norm(p=p, dim=dim, keepdim=True)
    x.div_(norm.clamp(min=1e-4))
    return x

