import json
from typing import Union

import pytorch_lightning as pl
import torch
from os import path
from torch.utils.data import random_split
from data.flipped import FlippedLabels
from data.util import get_dataset_path


class UCIHeartModule(pl.LightningDataModule):
    def __init__(self, root_dir=None, shift=True, test_samples: Union[int, str] = 'all',
                 test_seed=0, batch_size=32, num_workers: int = 96 // 2, negative_labels=True,
                 combine_val_and_test=False):
        super(UCIHeartModule, self).__init__()
        if root_dir is None:
            root_dir = get_dataset_path('uci_heart')
        self.data = torch.load(path.join(root_dir, 'uci_heart_torch.pt'))
        self.train = self.data['train']
        self.ood_test = self.data['ood_test']
        self.iid_test = self.data['iid_test']
        if shift:
            self.test = self.ood_test
        else:
            self.test = self.iid_test
        if test_samples != 'all':
            self.test, _ = random_split(self.test, [test_samples, len(self.test) - test_samples],
                                        generator=torch.Generator().manual_seed(test_seed))

        if negative_labels:
            self.test = FlippedLabels(self.test)

        self.val = self.data['val']
        if combine_val_and_test:
            self.val = torch.utils.data.ConcatDataset([self.val, self.iid_test])

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_samples = test_samples
        self.test_seed = test_seed
        self.negative_labels = negative_labels

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, shuffle=False,
                                           num_workers=self.num_workers)

    def test_dataloader(self, specific_test_loader=None):
        if specific_test_loader is None:
            return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=True,
                                               num_workers=self.num_workers)
        else:
            if specific_test_loader == 'iid':
                test = self.iid_test
            elif specific_test_loader == 'ood':
                test = self.ood_test
            else:
                raise ValueError(f'Invalid test loader {specific_test_loader}')
            return torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=True,
                                               num_workers=self.num_workers)
