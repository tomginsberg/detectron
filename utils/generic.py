from typing import Dict, Any

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import inspect


def update_functional(args: Dict[str, Any]):
    if args is None:
        return lambda x, y: x

    def f(default: Any, name: str):
        if name in args:
            return args[name]
        return default

    return f


class no_train(object):
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        self.module.train(False)

    def __exit__(self):
        self.module.train(True)


def vprint(verbose=True):
    def f(x):
        if verbose:
            print(x)

    return f


def compute_mean_and_std(dataset):
    # taken from https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949
    loader = DataLoader(dataset,
                        batch_size=512,
                        num_workers=96,
                        shuffle=False)

    mean = 0.
    std = 0.
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(dataset)
    std /= len(dataset)
    return mean, std


def key_replace(dc, pattern):
    return {k.replace(pattern, ''): v for k, v in dc.items()}


def dict_print(d):
    print(pd.DataFrame([d]))


def has_arg(func, arg):
    return arg in inspect.signature(func).parameters
