# -*- coding: UTF-8 -*-
import torch

class RGANDataset(torch.utils.data.Dataset):
    def __init__(self, data, time=None):
        # sanity check
        if len(data) != len(time):
            raise ValueError(
                f"len(data) `{len(data)}` != len(time) {len(time)}"
            )
        if isinstance(time, type(None)):
            time = [len(x) for x in data]

        # Labels are considered as features
        self.X = torch.FloatTensor(data)
        self.T = torch.LongTensor(time)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx]

    def collate_fn(self, batch):
        """Minibatch sampling
        """
        # Pad sequences to max length
        X_mb = [X for X in batch[0]]
        
        # The actual length of each data
        T_mb = [T for T in batch[1]]
        
        return X_mb, T_mb

class RCGANDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=None, time=None):
        """Labels are considered as features (same as TimeGAN)
        """
        # sanity check
        if len(data) != len(time):
            raise ValueError(
                f"len(data) `{len(data)}` != len(time) {len(time)}"
            )
        if len(data) != len(labels):
            raise ValueError(
                f"len(data) `{len(data)}` != len(time) {len(time)}"
            )
        if isinstance(time, type(None)):
            time = [len(x) for x in data]

        # Labels are appended by model
        self.X = torch.FloatTensor(data[:, :, :-1]) # Input does not include labels
        self.T = torch.LongTensor(time)
        self.Y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y[idx]

    def collate_fn(self, batch):
        """Minibatch sampling
        """
        # Pad sequences to max length
        X_mb = [X for X in batch[0]]
        
        # The actual length of each data
        T_mb = [T for T in batch[1]]

        # The labels for each data
        Y_mb = [Y for Y in batch[2]]
        
        return X_mb, T_mb, Y_mb
