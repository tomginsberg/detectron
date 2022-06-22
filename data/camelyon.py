from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import ToTensor, Resize, Compose
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset

from .base import DetectronDataModule
from .flipped import FlippedLabels
from .subset import Subsetable, DropMeta
import json


class CamelyonModule(DetectronDataModule):

    def __init__(self, root_dir=None,
                 train_samples: Union[int, str] = 70000,
                 val_samples: Union[int, str] = 10000,
                 batch_size: int = 512,
                 test_seed: int = 42,
                 test_samples: Union[int, str] = 'all',
                 shift=True,
                 num_workers: int = 96 // 2,
                 return_meta=False,
                 negative_labels=False,
                 small_dev_sets=False,
                 id_val=True,
                 test_domain=5,
                 ):
        super(CamelyonModule, self).__init__()
        if root_dir is None:
            root_dir = json.load(open('paths.json'))['camelyon17']
        assert test_domain in {4, 5}, f'Test domain must be either hospital 4 or 5, not {test_domain}'
        self.test_split = {5: 'test', 4: 'val'}[
            test_domain]  # the val split is from hospital 4, the test split is from hospital 5
        self.save_hyperparameters()
        self.root_dir = root_dir
        meta = lambda x: x
        if not return_meta:
            meta = DropMeta

        d = Camelyon17Dataset(root_dir=self.root_dir, download=False)

        self.train, self.val = [meta(d.get_subset(i, transform=Compose([Resize((224, 224)), ToTensor()])))
                                for i in ['train', "id_val" if id_val else "val"]]
        self.id_val = id_val

        if small_dev_sets:
            self.train = Subsetable(self.train)
            self.train.refine_to_amount(100, random_seed=42)
            self.val = Subsetable(self.val)
            self.val.refine_to_amount(100, random_seed=42)

        if val_samples != 'all':
            self.val, _ = random_split(self.val, [val_samples, len(self.val) - val_samples],
                                       generator=torch.Generator().manual_seed(test_seed))
        if train_samples != 'all':
            n_train = len(self.train)
            self.train, _ = random_split(self.train, [train_samples, n_train - train_samples],
                                         generator=torch.Generator().manual_seed(0))

        self.train_dl = DataLoader(self.train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_dl = DataLoader(self.val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.meta = meta
        self.batch_size = batch_size
        self.shift = shift
        self.negative_labels = negative_labels
        self.test_samples = test_samples
        self.test_seed = test_seed
        self.d = d
        self.num_workers = num_workers

        self.test = None
        self.test_dl = None
        self.configure_test_set(test_seed, test_samples, shift)

    def train_dataloader(self) -> DataLoader:
        return self.train_dl

    def val_dataloader(self) -> DataLoader:
        return self.val_dl

    def test_dataloader(self) -> DataLoader:
        return self.test_dl

    def configure_test_set(self, test_seed, test_samples, shift, exclusion_amount=None, exclusion_seed=None):
        if self.negative_labels:  # flipped label datasets automatically drop metadata
            f = lambda x: x
        else:
            f = self.meta

        self.shift = shift
        self.test_samples = test_samples

        if not shift:
            self.test = self.val
        else:
            self.test = f(self.d.get_subset(self.test_split, transform=Compose([Resize((224, 224)), ToTensor()])))

        if self.negative_labels:
            self.test = FlippedLabels(self.test)

        if test_samples == 'all':
            pass
        elif test_samples <= len(self.test):
            self.test = Subsetable(self.test)

            self.test.refine_to_amount(test_samples, random_seed=test_seed, exclusion_amount=exclusion_amount,
                                       exclusion_seed=exclusion_seed)
        else:
            raise ValueError(
                f'Test samples ({test_samples}) must be less than or equal to the number of samples in the test set: '
                f'({len(self.test)})')

        self.test_dl = DataLoader(self.test, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
