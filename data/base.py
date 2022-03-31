from abc import abstractmethod
from typing import Union

import pytorch_lightning as pl


class DetectronDataModule(pl.LightningDataModule):

    @abstractmethod
    def configure_test_set(self, test_seed, test_samples, unshifted_test):
        raise NotImplementedError
