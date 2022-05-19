from glob import glob
from os import path
from typing import Union, Tuple, Dict, Iterable

import pytorch_lightning as pl
import torch
from rejectron.rejectronstep import RejectronStep
from typing import Any
from tqdm import tqdm
from copy import deepcopy
from shift_detection.shiftdetection import ensemble_entropy


class CWarpper(pl.LightningModule):
    def __init__(self, c: pl.LightningModule, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.c = c

    def forward(self, *args, **kwargs) -> Any:
        return self.c(*args, **kwargs)

    def get_c(self) -> pl.LightningModule:
        return self.c


class RejectronModule(pl.LightningModule):
    def __init__(self, h: pl.LightningModule, C: Iterable[Union[RejectronStep, pl.LightningModule]] = None, **kwargs):
        """

        :param h:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.C = torch.nn.ModuleList()
        if C is not None:
            self.add_all_c(C)
        self.h = h
        self.val_stats = []
        self.labels = []

    def load_from_directory(self, directory: str, sort_key=None, cls=None, sub_dirs=True):
        if sub_dirs:
            ckpts = glob(path.join(directory, "*", "last.ckpt"))
        else:
            ckpts = glob(path.join(directory, "*.ckpt"))
        if sort_key is not None:
            sort_key = lambda x: int(x.split('/')[-1].split('_')[0][1:])

        ckpts = sorted(ckpts, key=sort_key)
        print(f"Loading {len(ckpts)} checkpoints from {directory} using class {type(self.h)}")
        for ck in tqdm(ckpts):
            c = deepcopy(self.h)
            # noinspection PyTypeChecker
            c.load_state_dict(
                {k.replace('c.model', 'model'): v for k, v in torch.load(ck)['state_dict'].items() if
                 k.startswith('c')})
            self.add_new_c(c)

    def add_new_c(self, c_step: Union[RejectronStep, pl.LightningModule]):
        if hasattr(c_step, 'get_c'):
            self.C.append(c_step.get_c())
        else:
            self.C.append(c_step)

    def add_all_c(self, c_step: Iterable[Union[RejectronStep, pl.LightningModule]]):
        for c in c_step:
            self.add_new_c(c)
        return self

    def eval(self):
        print('Setting module to eval mode')
        self.h.eval()
        for c in self.C:
            c.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = torch.argmax(self.h.forward(x), dim=1)

            for c in self.C:
                # mask out indices with -1 whenever a c model disagrees with h
                mask = torch.argmax(c(x), dim=1) != y
                y[mask] = -1

            return y

    def logits(self, x) -> torch.Tensor:
        with torch.no_grad():
            logits = torch.stack([self.h(x), *[c(x) for c in self.C]], dim=0)
            return logits

    def entropy(self, x) -> torch.Tensor:
        with torch.no_grad():
            return ensemble_entropy(self.logits(x))

    def get_val_stats(self) -> Dict[str, torch.Tensor]:
        stats = torch.cat(self.val_stats, dim=0)
        labels = torch.cat(self.labels, dim=0)
        p = stats.argmax(dim=-1)
        r = torch.tensor([(x[0] == x[1:]).all() for x in p])
        rejection = 1 - r.float().mean()
        # print(f"Rejection: {rejection}")
        accepted_acc = (p[r, 0] == labels[r]).float().mean()
        rejected_acc = (p[~r, 0] == labels[~r]).float().mean()
        global_acc = (p[:, 0] == labels).float().mean()
        # print(f"Accepted Accuracy: {(p[r,0] == labels[r]).float().mean()}")
        # print(f"Rejected Accuracy: {(p[~r, 0] == labels[~r]).float().mean()}")
        # print(f"Global Accuracy: {(p[:, 0] == labels).float().mean()}")
        self.val_stats = []
        self.labels = []
        return dict(p=p, r=r, stats=stats, labels=labels, rejection=rejection, accepted_acc=accepted_acc,
                    rejected_acc=rejected_acc, global_acc=global_acc)

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        self.labels.append(y)
        with torch.no_grad():
            y = self.h.forward(x)
            z = torch.stack([y] + [c(x) for c in self.C]).permute(1, 0, 2)
            self.val_stats.append(z)


if __name__ == '__main__':
    pass
