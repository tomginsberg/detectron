import json
import random
import warnings
from os.path import join
from typing import Union, Collection, Tuple

import numpy as np
import torch.nn
import torchvision
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from data.flipped import FlippedLabels
import pandas as pd
import matplotlib.pyplot as plt

from data.util import get_dataset_path
from utils.image_transforms import UnNormalize

TransformType = Union[
    torchvision.transforms.Compose, torch.nn.Module, torchvision.transforms.ToTensor]

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


class CIFAR10C(Dataset):
    def __init__(self, root_dir=None, shift_types=('frost', 'fog', 'snow'), seed=42, num_images=10000,
                 severity_range=(3, 5), negative_labels=True, return_meta=False):
        super(CIFAR10C, self).__init__()
        if root_dir is None:
            root_dir = get_dataset_path('cifar10C')
        self.num_images = num_images
        _images = [np.load(join(root_dir, 'CIFAR-10-C', x + '.npy')) for x in shift_types]
        _labels = np.load(join(root_dir, 'CIFAR-10-C', 'labels.npy')).astype(int)

        rd = random.Random(seed)
        severity = np.array([rd.randint(severity_range[0] - 1, severity_range[1] - 1) for _ in range(num_images)])
        corruption = np.array([rd.randint(0, len(shift_types) - 1) for _ in range(num_images)])
        subset = np.array(rd.sample(range(10000), num_images))
        images = []
        labels = []
        for s, c, idx in zip(severity, corruption, subset):
            # severity increases by 1 every 10,000 images
            images.append(Image.fromarray(_images[c][idx + 10000 * s]))
            labels.append(_labels[idx + 10000 * s])

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

        self.images = images
        self.labels = labels
        self.nl = negative_labels
        self.return_meta = return_meta
        self.severity = severity
        self.corruption = corruption

    def __len__(self):
        return self.num_images

    def __getitem__(self, index) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, Tuple[int, int, int]]]:
        label = -self.labels[index] - 1 if self.nl else self.labels[index]
        if self.return_meta:
            return self.tf(self.images[index]), (label, self.severity[index], self.corruption[index])
        return self.tf(self.images[index]), label


class CIFAR10_1(Dataset):
    def __init__(self,
                 root='/voyager/datasets/',
                 negative_labels=True,
                 test_seed: int = 42,
                 test_samples: Union[int, str] = 'all'):
        """
        Cifar 10.1 dataset.
        Code: https://github.com/modestyachts/CIFAR-10.1
        Paper: https://arxiv.org/abs/1806.00451 (Do CIFAR-10 Classifiers Generalize to CIFAR-10?)
        :param root:  `root_path_to`/cifar-10-1/cifar10.1_v6_data.pt and cifar-10-1/cifar10.1_v6_labels.pt
        :param negative_labels: returns label = - label - 1 if True
        :param test_seed: seed for test set
        :param test_samples: number of samples in test set, or 'all'
        """
        self.images = torch.load(join(root, 'cifar-10-1', 'cifar10.1_v6_data.pt'))
        self.labels = torch.load(join(root, 'cifar-10-1', 'cifar10.1_v6_labels.pt')).long()
        if test_samples != 'all':
            assert test_seed <= len(self.images), f'test_seed must be <= {len(self.images)}'
            # randomly take test_samples from the test set
            rd = random.Random(test_seed)
            test_samples = rd.sample(range(len(self.images)), test_samples)
            self.images = self.images[test_samples]
            self.labels = self.labels[test_samples]
        self.nl = negative_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, Tuple[int, int, int]]]:
        label = -self.labels[index] - 1 if self.nl else self.labels[index]
        return self.images[index], label.item()

    @staticmethod
    def create(root_dir='/voyager/datasets/'):
        c10_1 = np.load(join(root_dir, 'cifar-10-1/cifar10.1_v6_data.npy'))
        c10_1_l = np.load(join(root_dir, 'cifar-10-1/cifar10.1_v6_labels.npy'))
        c10_1_l = torch.from_numpy(c10_1_l)
        c10_1 = torch.from_numpy(c10_1).permute(0, 3, 1, 2).float() / 255.0
        tf = transforms.Normalize(MEAN, STD)
        c10_1 = torch.stack(
            [tf(x) for x in tqdm(c10_1)])
        torch.save(c10_1, join(root_dir, 'cifar-10-1', 'cifar10.1_v6_data.pt'))
        torch.save(c10_1_l, join(root_dir, 'cifar-10-1', 'cifar10.1_v6_labels.pt'))


class CIFAR10CFrostFogSnow(Dataset):
    def __init__(self, root_dir=None,
                 negative_labels=True,
                 return_meta=False,
                 test_seed: int = 42,
                 test_samples: Union[int, str] = 'all', ):
        super(CIFAR10CFrostFogSnow, self).__init__()
        if root_dir is None:
            root_dir = get_dataset_path('cifar10C')
        self.return_meta = return_meta
        self.images = torch.load(join(root_dir, 'CIFAR-10-C', 'frost-fog-snow.pt'))
        self.labels = pd.read_csv(join(root_dir, 'CIFAR-10-C', 'frost-fog-snow.csv'))
        if test_samples != 'all':
            assert test_seed <= len(self.images), f'test_seed must be <= {len(self.images)}'
            # randomly take test_samples from the test set
            rd = random.Random(test_seed)
            test_samples = rd.sample(range(len(self.images)), test_samples)
            self.images = [self.images[i] for i in test_samples]
            self.labels = self.labels.iloc[test_samples]

        self.nl = negative_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, Tuple[int, int, int]]]:
        label, severity, corruption = list(self.labels.iloc[index])
        label = -label - 1 if self.nl else label
        if self.return_meta:
            return self.images[index], (label, severity, corruption)
        return self.images[index], label

    @staticmethod
    def create(root=None):
        dm = CIFAR10C(root_dir=None, return_meta=True, negative_labels=False)
        if root is None:
            root = get_dataset_path('cifar10C')
        ims = []
        data = []
        for im, (label, severity, corruption) in dm:
            ims.append(im)
            data.append(dict(label=label, severity=severity, corruption=corruption))

        ims = torch.stack(ims)
        torch.save(ims, join(root, 'CIFAR-10-C', 'frost-fog-snow.pt'))
        pd.DataFrame(data).to_csv(join(root, 'CIFAR-10-C', 'frost-fog-snow.csv'))


class CIFAR10DataModule(LightningDataModule):
    shift_options = {'brightness',
                     'frost',
                     'jpeg_compression',
                     'shot_noise',
                     'contrast',
                     'gaussian_blur',
                     'labels',
                     'snow',
                     'defocus_blur',
                     'gaussian_noise',
                     'motion_blur',
                     'spatter',
                     'elastic_transform',
                     'glass_blur',
                     'pixelate',
                     'speckle_noise',
                     'fog',
                     'impulse_noise',
                     'saturate',
                     'zoom_blur',
                     'frost-fog-snow',
                     'cifar-10-1'}

    def __init__(
            self, root_dir: str = None,
            batch_size: int = 512,
            test_seed: int = 42,
            test_samples: Union[int, str] = 'all',
            shift_types: Union[str, Collection[str]] = 'frost-fog-snow',
            shift_severity_range=(3, 5),
            shift=True,
            num_workers: int = 96 // 2,
            negative_labels=True,
            return_meta=False,
            split_val=True,
            train_samples=None,
            val_samples=None,
    ):
        super().__init__()
        if root_dir is None:
            root_dir = get_dataset_path('cifar10')
        if train_samples is not None or val_samples is not None:
            warnings.warn('train_samples and val_samples cannot be specified, arguments exist only for compatibility')
        if test_samples != 'all':
            if shift:
                assert test_samples <= 10000
            else:
                assert test_samples <= 1000
        if not split_val:
            assert shift, 'Must have split_val=True if shift=False'
        self.save_hyperparameters()
        self.shift_type = shift_types
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        self.transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        self.train = torchvision.datasets.CIFAR10(root_dir, train=True, transform=self.transform_train)
        self.val = torchvision.datasets.CIFAR10(root_dir, train=False, transform=self.transform_val)
        if split_val:
            self.val, iid_test = torch.utils.data.random_split(self.val, [9000, 1000], torch.Generator().manual_seed(0))
        else:
            iid_test = ValueError()  # should not refer to iid_test if split_val=False
        self.batch_size = batch_size
        self.root = root_dir

        if test_samples == 'all':
            if ~shift:
                test_samples = len(iid_test)
            else:
                test_samples = len(self.val)

        self.test_samples = test_samples
        self.test_seed = test_seed
        self.un_normalize = UnNormalize(MEAN, STD)
        if shift_types == 'all':
            shift_types = self.shift_options
        elif isinstance(shift_types, str):
            assert shift_types in self.shift_options, \
                f'shift type {shift_types} is not in available shift types {self.shift_options}'
        else:
            for x in shift_types:
                assert x in self.shift_options, f'shift type {x} is not in available shift types {self.shift_options}'
        if shift:
            # special cases
            if shift_types == 'frost-fog-snow':
                self.test = CIFAR10CFrostFogSnow(
                    root_dir=None, test_samples=test_samples,
                    test_seed=test_seed,
                    negative_labels=negative_labels,
                    return_meta=return_meta
                )
            elif shift_types == 'cifar-10-1':
                self.test = CIFAR10_1(
                    root_dir, test_samples=test_samples,
                    test_seed=test_seed,
                    negative_labels=negative_labels,
                )
            # generic 10C dataset
            else:
                self.test = CIFAR10C(root_dir=None, shift_types=shift_types, severity_range=shift_severity_range,
                                     negative_labels=negative_labels, seed=test_seed, num_images=test_samples,
                                     return_meta=return_meta)
        else:
            self.test = iid_test
            if self.test_samples < 1000:
                self.test = torch.utils.data.random_split(
                    self.test, [self.test_samples, 1000 - self.test_samples],
                    torch.Generator().manual_seed(self.test_seed)
                )[0]
            if negative_labels:
                self.test = FlippedLabels(self.test)
        self.num_workers = num_workers

    def __dataloader(self, split='train') -> DataLoader:
        return DataLoader(
            self.train if split == 'train' else self.val,
            batch_size=self.batch_size,
            # pin_memory=True,
            num_workers=self.num_workers,
            shuffle=True if split == 'train' else False,
            persistent_workers=True
        )

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader(split='train')

    def val_dataloader(self) -> DataLoader:
        return self.__dataloader(split='val')

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True
        )

    def update_test_transform(self, train=True):
        """
        :param train: bool, if true uses the train transform for the test set (i.e data augmentation)
        if not uses the default validation transform (i.e normalize + ToTensor)
        """
        if train:
            self.test.tf = self.transform_train
        else:
            self.test.tf = self.transform_val

    def preview(self, n=5):
        # pyplot image grid of the first n images from val and test
        fig, ax = plt.subplots(2, n, figsize=(n * 3, 6))
        for i in range(n):
            va = self.un_normalize(self.val[i][0]).permute(1, 2, 0).numpy()
            te = self.un_normalize(self.test[i][0]).permute(1, 2, 0).numpy()
            ax[0, i].imshow(va)
            ax[1, i].imshow(te)
            # turn off axes
            ax[0, i].axis('off')
            ax[1, i].axis('off')
        plt.show()


if __name__ == '__main__':
    CIFAR10_1.create()
