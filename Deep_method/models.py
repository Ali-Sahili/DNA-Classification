import copy
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from layers import BioEmbedding, CKNLayer, RowPreprocessor, GlobalAvg1D, LinearMax


class CKNSequential(nn.Module):
    """https://gitlab.inria.fr/dchen/CKN-seq"""
    def __init__(self, in_channels, out_channels_list, filter_sizes,
                 subsamplings, kernel_func="exp", kernel_args=0.3):
        assert len(out_channels_list) == len(filter_sizes) == len(subsamplings), "incompatible dimensions"

        super(CKNSequential, self).__init__()

        self.n_layers = len(out_channels_list)

        ckn_layers = []

        for i in range(self.n_layers):
            ckn_layer = CKNLayer(in_channels, out_channels_list[i],
                                 filter_sizes[i], subsampling=subsamplings[i],
                                 kernel_func=kernel_func,
                                 kernel_args=kernel_args)

            ckn_layers.append(ckn_layer)
            in_channels = out_channels_list[i]

        self.ckn_layers = nn.Sequential(*ckn_layers)

    def __iter__(self):
        return iter(self.ckn_layers._modules.values())

    def forward_at(self, x, i=0):
        assert x.size(1) == self.ckn_layers[i].in_channels, "bad dimension"
        return self.ckn_layers[i](x)

    def forward(self, x):
        return self.ckn_layers(x)

    def representation(self, x, n=0):
        if n == -1:
            n = self.n_layers
        for i in range(n):
            x = self.forward_at(x, i)
        return x

    def compute_mask(self, mask=None, n=-1):
        if mask is None:
            return mask
        if n > self.n_layers:
            raise ValueError("Index larger than number of layers")
        if n == -1:
            n = self.n_layers
        for i in range(n):
            mask = self.ckn_layers[i].compute_mask(mask)
        return mask

    def normalize_(self):
        for module in self.ckn_layers:
            module.normalize_()

class CKN(nn.Module):
    """https://gitlab.inria.fr/dchen/CKN-seq"""
    def __init__(self, in_channels, out_channels_list, filter_sizes,
                 subsamplings, kernel_func="exp", kernel_args=0.3,
                 alpha=0., penalty='l2'):
                 
        super(CKN, self).__init__()
        self.embed_layer = BioEmbedding(in_channels, mask_zeros=True)
        self.ckn_model = CKNSequential(in_channels, out_channels_list, filter_sizes,
                                       subsamplings, kernel_func, kernel_args)
        self.global_pool = GlobalAvg1D()
        self.out_features = out_channels_list[-1]
        self.scaler = RowPreprocessor()
        self.classifier = LinearMax(self.out_features, 1, alpha=alpha, penalty=penalty)

    def normalize_(self):
        self.ckn_model.normalize_()

    def representation_at(self, input, n=0):
        output = self.embed_layer(input)
        mask = self.embed_layer.compute_mask(input)
        output = self.ckn_model.representation(output, n)
        mask = self.ckn_model.compute_mask(mask, n)
        return output, mask

    def representation(self, input):
        output = self.embed_layer(input)
        mask = self.embed_layer.compute_mask(input)
        output = self.ckn_model(output)
        mask = self.ckn_model.compute_mask(mask)
        output = self.global_pool(output, mask)
        return output

    def forward(self, input, proba=False):
        output = self.representation(input)
        return self.classifier(output, proba)

    def unsup_train_ckn(self, data_loader, use_cuda=False):
        n_sampling_patches=250000
        self.train(False)
        if use_cuda:
            self.cuda()
        for i, ckn_layer in enumerate(self.ckn_model):
            print("Training layer {}".format(i))
            n_patches = 0
            try:
                n_patches_per_batch = (n_sampling_patches + len(data_loader) - 1) // len(data_loader)
            except:
                n_patches_per_batch = 1000
            patches = torch.Tensor(n_sampling_patches, ckn_layer.patch_dim)
            if use_cuda:
                patches = patches.cuda()

            for data, _ in data_loader:
                if n_patches >= n_sampling_patches:
                    continue
                if use_cuda:
                    data = data.cuda()
                with torch.no_grad():
                    data, mask = self.representation_at(data, i)
                    data_patches = ckn_layer.sample_patches(
                        data, mask, n_patches_per_batch)
                size = data_patches.size(0)
                if n_patches + size > n_sampling_patches:
                    size = n_sampling_patches - n_patches
                    data_patches = data_patches[:size]
                patches[n_patches: n_patches + size] = data_patches
                n_patches += size

            print("total number of patches: {}".format(n_patches))
            patches = patches[:n_patches]
            ckn_layer.unsup_train(patches)

    def unsup_train_classifier(self, data_loader, criterion=None, use_cuda=False):
        encoded_train, encoded_target = self.predict(data_loader, True, use_cuda=use_cuda)
        if hasattr(self, 'scaler') and not self.scaler.fitted:
            self.scaler.fitted = True
            size = encoded_train.shape[0]
            encoded_train = self.scaler.fit_transform(encoded_train.view(-1, self.out_features)
                                                     ).view(size, -1)
        self.classifier.fit(encoded_train, encoded_target, criterion)

    def predict(self, data_loader, only_representation=False, proba=False, use_cuda=False):
        self.train(False)
        if use_cuda:
            self.cuda()
        n_samples = len(data_loader.dataset)
        target_output = torch.Tensor(n_samples)
        batch_start = 0
        for i, (data, target, *_) in enumerate(data_loader):
            batch_size = data.shape[0]
            if use_cuda:
                data = data.cuda()
            with torch.no_grad():
                if only_representation:
                    batch_out = self.representation(data).data.cpu()
                else:
                    batch_out = self(data, proba).data.cpu()

            #batch_out = torch.cat((batch_out[:batch_size], batch_out[batch_size:]), dim=-1)
            if i == 0:
                output = torch.Tensor(n_samples, batch_out.shape[-1])
            output[batch_start:batch_start+batch_size] = batch_out
            target_output[batch_start:batch_start+batch_size] = target
            batch_start += batch_size
        output.squeeze_(-1)
        return output, target_output
