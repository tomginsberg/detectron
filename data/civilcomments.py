from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import DistilBertTokenizerFast
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset

from .base import DetectronDataModule
from .flipped import FlippedLabels
from .subset import Subsetable, DropMeta


class CivilComments(DetectronDataModule):
    def __init__(self, root_dir='/voyager/datasets', max_token_length=300,
                 batch_size: int = 512,
                 test_seed: int = 42,
                 test_samples: Union[int, str] = 'all',
                 shift=True,
                 num_workers: int = 96 // 2,
                 return_meta=False,
                 negative_labels=False,
                 small_dev_sets=False,
                 predict_seed=0,
                 predict_samples=10000):
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = root_dir
        meta = lambda x: x
        if not return_meta:
            meta = DropMeta

        self.dataset = CivilCommentsDataset(root_dir=root_dir)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.max_token_length = max_token_length

        self.train = self.dataset.get_subset('train', transform=self.transform)
        self.val = self.dataset.get_subset('val', transform=self.transform)
        self.test = self.dataset.get_subset('test', transform=self.transform)

        if small_dev_sets:
            self.train = Subsetable(self.train)
            self.train.refine_to_amount(100, random_seed=42)
            self.val = Subsetable(self.val)
            self.val.refine_to_amount(100, random_seed=42)
            predict_samples = 20

        n_train = len(self.train)
        self.train, self.predict = random_split(self.train, [n_train - predict_samples, predict_samples],
                                                generator=torch.Generator().manual_seed(predict_seed))

        self.train_dl = DataLoader(self.train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_dl = DataLoader(self.val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.predict_dl = DataLoader(self.predict, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.meta = meta
        self.batch_size = batch_size
        self.shift = shift
        self.negative_labels = negative_labels
        self.test_samples = test_samples
        self.test_seed = test_seed
        self.num_workers = num_workers

        self.test_dl = None
        self.configure_test_set(test_seed, test_samples, shift)

    def transform(self, text):
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt",
        )
        x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    def configure_test_set(self, test_seed, test_samples, shift):
        if self.negative_labels:  # flipped label datasets automatically drop metadata
            f = lambda x: x
        else:
            f = self.meta

        self.shift = shift
        self.test_samples = test_samples

        if not shift:
            self.test = self.val
        else:
            self.test = f(self.dataset.get_subset('test', transform=self.transform))

        if self.negative_labels:
            self.test = FlippedLabels(self.test)

        if test_samples == 'all':
            pass
        elif test_samples <= len(self.test):
            self.test = Subsetable(self.test)
            self.test.refine_to_amount(test_samples, random_seed=test_seed)
        else:
            raise ValueError(
                f'Test samples ({test_samples}) must be less than or equal to the number of samples in the test set: '
                f'({len(self.test)})')

        self.test_dl = DataLoader(self.test, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def train_dataloader(self) -> DataLoader:
        return self.train_dl

    def val_dataloader(self) -> DataLoader:
        return self.val_dl

    def predict_dataloader(self) -> DataLoader:
        return self.predict_dl

    def test_dataloader(self) -> DataLoader:
        return self.test_dl
