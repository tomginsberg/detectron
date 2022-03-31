import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader

from rejectron.rejectronmodule import RejectronModule


def sample_rejectron(dataloader: DataLoader, h, ckpt_path) -> dict[str, Tensor]:
    t = pl.Trainer(gpus=1)
    r = RejectronModule(h)
    r.load_from_directory(ckpt_path)

    t.validate(r, dataloader)
    return r.get_val_stats()
