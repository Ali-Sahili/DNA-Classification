import torch
from torch.utils.data import Dataset, DataLoader


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#-----------------------------------------------------------------------------------------
#                                     To prepare dataloader
#-----------------------------------------------------------------------------------------
class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, noise=0.0, max_index=4):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.noise = noise
        self.max_index = max_index + 1

    def __getitem__(self, index):
        if self.noise == 0.:
            return self.data_tensor[index], self.target_tensor[index]
        data_tensor = self.data_tensor[index].clone()
        noise_mask = torch.ByteTensor([0])
        if np.random.rand() < 0.5:
            noise_mask = torch.ByteTensor([1])
            # mask = torch.Tensor(data_tensor.size(0)).uniform_() < self.noise
            mask = torch.rand_like(data_tensor, dtype=torch.float) < self.noise
            data_tensor[mask] = torch.LongTensor(
                mask.sum().item()).random_(1, self.max_index)
        return data_tensor, self.target_tensor[index], noise_mask

    def __len__(self):
        return self.data_tensor.size(0)

    def augment(self, noise=0.1, quantity=10):
        if noise <= 0.:
            return
        new_tensor = [self.data_tensor]
        for i in range(quantity - 1):
            t = self.data_tensor.clone()
            mask = torch.rand_like(t, dtype=torch.float) < noise
            t[mask] = torch.LongTensor(mask.sum().item()).random_(1, self.max_index)
            new_tensor.append(t)
        self.data_tensor = torch.cat(new_tensor)
        self.target_tensor = self.target_tensor.repeat(quantity)


#-----------------------------------------------------------------------------------------
#                                      Loading data
#-----------------------------------------------------------------------------------------
#alpha_nb = len(alphabet)
def load_data(filename, label_filename, phase="train", 
               batch_size=10, alpha_nb=4, noise=0, len_motifs=12):

    def seq2index(seq):
        alphabet, code = ('ACGT', '\x01\x02\x03\x04')
        alpha_ambi, code_ambi = ('N', '\x00')

        translator = str.maketrans( alpha_ambi + alphabet, code_ambi + code)
        seq = seq.translate(translator)
        seq = np.fromstring(seq, dtype='uint8')
        return seq.astype('int64')

    def pad_sequences(sequences, pre_padding=0, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):

        lengths = []
        for x in sequences:
            lengths.append(len(x))

        num_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths) + 2*pre_padding

        sample_shape = tuple()
        for s in sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break

        x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if not len(s):
                continue  # empty list/array was found
            pre_pad = [value]*pre_padding
            s = np.hstack([pre_pad, s, pre_pad])
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)

            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' % (trunc.shape[1:], idx, sample_shape))
            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)
        return x


    df = pd.read_csv(filename)
    df_label = pd.read_csv(label_filename)
    df['seq_index'] = df['seq'].apply(seq2index)
        
    X, y = df['seq_index'], df_label['Bound'].values
    #print(X.shape, y.shape)
    
    X = pad_sequences(X, pre_padding=len_motifs-1, maxlen=None, padding='post',
                                             truncating='post', dtype='int64')
    
    if phase=="train":    
        X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
        X, y = torch.from_numpy(X), torch.from_numpy(y)
        X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)

        train_dset = TensorDataset(X, y, noise=noise, max_index=alpha_nb)
        val_dset = TensorDataset(X_val, y_val, max_index=alpha_nb)
        print("dimensions of training and validation sets: ", len(train_dset), len(val_dset))
        print()
        
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    elif phase == "test":
        X, y = torch.from_numpy(X), torch.from_numpy(y)

        test_dset = TensorDataset(X, y, noise=noise, max_index=alpha_nb)
        print("size of test set: ",len(test_dset))
        print()
        
        test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)
        return test_loader
