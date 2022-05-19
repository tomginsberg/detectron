import random
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from scipy.stats import ks_2samp, binomtest
from tqdm import tqdm
import os
from data.base import DetectronDataModule
import torch.nn.functional as F


def disjoint_index_sets(length, seed=88, samples=1000):
    """
    Returns two disjoint sets A, B of elements from 0 to length-1
    |A| = samples,
    |B| = length - samples
    """
    rng = random.Random(seed)
    r = range(length)
    r1 = rng.sample(r, k=samples)
    r2 = list(set(r) - set(r1))
    return r1, r2


def reverse_pairs(lst):
    """
    Reverses each pair of elements in a list
    l [l1 l2 l3 l4] -> [l2 l1 l4 l3]
    """
    ls = []
    for i in range(0, len(lst), 2):
        ls += lst[i:i + 2][::-1]
    return ls


def ensemble_entropy(x):
    """
    Computes the entropy of a distribution
    """
    eps = 1e-8
    p_x = x.softmax(dim=-1).mean(dim=0)
    return -torch.sum(p_x * torch.log(p_x + eps), -1)


def entropy(x):
    """
    Computes the entropy of a distribution
    """
    eps = 1e-8
    p_x = x.softmax(dim=1)
    return -torch.sum(p_x * torch.log(p_x + eps), -1)


class ShiftDetection:
    def __init__(self, models: List[pl.LightningModule], model_names: List[str], datamodule: pl.LightningDataModule,
                 df_path: str, logit_path: str, load_logits: bool = False, baseline_samples: int = 1000):
        self.models = [m.eval() for m in models]
        self.model_names = model_names
        self.datamodule = datamodule
        self.df = pd.DataFrame()
        self.df_path = df_path
        self.logit_path = logit_path
        # create directory if it doesn't exist
        if not os.path.exists(self.df_path):
            os.makedirs(self.df_path)
        if not os.path.exists(self.logit_path):
            os.makedirs(self.logit_path)

        if load_logits:
            self.q_data = self.get_logits_from_file('q')
            self.p_data = self.get_logits_from_file('p')
            # self.val = self.get_logits_from_file('val')
        else:
            self.q_data = self.get_logits(datamodule.test_dataloader(), name='q')
            self.p_data = self.get_logits(datamodule.val_dataloader(), name='p')
            # self.val = self.get_logits(datamodule.val_dataloader(), name='val')

        p1_idx, p2_idx = disjoint_index_sets(len(self.p_data[-1]), seed=88, samples=baseline_samples)
        self.p1 = self.p_data[0][:, p1_idx], self.p_data[1][p2_idx]
        self.ensemble_entropy_p1 = ensemble_entropy(self.p1[0])
        self.bbsd_entropy_p1 = entropy(self.p1[0][0])
        self.max_p1 = self.p1[0][0].max(1).values
        self.max_softmax_p1 = self.p1[0][0].softmax(dim=1).max(1).values

        self.p2 = self.p_data[0][:, p2_idx], self.p_data[1][p2_idx]

        self.baseline_rejection = self.rejection_rate(self.p1[0])

    def bbsd_sweep(self, test_samples, test_seeds, shifts=(True, False)):
        for test_seed in tqdm(test_seeds):
            for test_sample in test_samples:
                for shift in shifts:
                    self.bbsd(test_sample, test_seed, shift)

        self.df.to_json(os.path.join(self.df_path, 'shift.json'))

    def print_results(self):
        for algo in self.df.algorithm.unique():
            print(f'\n{algo}')
            df = self.df[self.df.algorithm == algo]
            me = 100 * df.groupby(['test_samples', 'shift']).significant.mean()
            se = 100 * df.groupby(['test_samples', 'shift']).significant.sem()
            # print(f'\tMean: {me.to_string()}')
            # print(f'\tSEM: {se.to_string()}')
            for m, s in reverse_pairs(list(zip(list(me), list(se)))):
                print(f'${m:.2f} \pm {s:.2f}$', end=' & ')

    def bbsd(self, test_samples, test_seed, shift):
        # print(f'Running BBSD on {test_samples} samples with seed {test_seed} and shift {shift}')
        test_labels, logits = self.prepare_data(test_samples, test_seed, shift)
        data = []

        for (model_name, p_data, test_logits) in (
                (zip(self.model_names, self.p1[0], logits))):
            test_acc = (test_logits.argmax(dim=1) == test_labels).float().mean().item()
            pvals = [ks_2samp(p_data[:, j], test_logits[:, j]).pvalue for j in range(test_logits.shape[-1])]
            significant = min(pvals) < 0.05 / len(pvals)
            data.append({'model': model_name, 'shift': shift, 'test_acc': test_acc,
                         'significant': significant, 'pvals': pvals, 'test_samples': test_samples,
                         'test_seed': test_seed, 'algorithm': 'bbsd'})

            pval = ks_2samp(self.bbsd_entropy_p1, entropy(test_logits)).pvalue
            significant = pval < 0.05
            data.append({'model': model_name, 'shift': shift, 'test_acc': test_acc,
                         'significant': significant, 'pvals': pval, 'test_samples': test_samples,
                         'test_seed': test_seed, 'algorithm': 'bbsd_entropy'})

            pval = ks_2samp(self.max_softmax_p1, test_logits.softmax(1).max(1).values).pvalue
            significant = pval < 0.05
            data.append({'model': model_name, 'shift': shift, 'test_acc': test_acc,
                         'significant': significant, 'pvals': pval, 'test_samples': test_samples,
                         'test_seed': test_seed, 'algorithm': 'bbsd_max_softmax'})

            pval = ks_2samp(self.max_p1, test_logits.max(1).values).pvalue
            significant = pval < 0.05
            data.append({'model': model_name, 'shift': shift, 'test_acc': test_acc,
                         'significant': significant, 'pvals': pval, 'test_samples': test_samples,
                         'test_seed': test_seed, 'algorithm': 'bbsd_max_logit'})

            break  # to be fair we only do bbsd on a single model, otherwise this is just an ensemble method

        self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True)

    def get_logits(self, dataloader, name) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.models[0].device
        assert all(m.device == device for m in self.models), 'Models must be on the same device'
        print(f'Getting logits on {len(dataloader.dataset)} {name} samples on GPU {device}')

        logits = []
        labels = []
        for (x, y) in tqdm(dataloader):
            x = x.to(device)
            with torch.no_grad():
                logits.append(torch.stack([model(x) for model in self.models]).cpu())
            labels.append(y)
        logits = torch.cat(logits, dim=1)  # models x samples x classes
        labels = torch.cat(labels, dim=0)
        torch.save(dict(logits=logits, labels=labels, model_names=self.model_names),
                   f'{self.logit_path}/{name}.pt')
        return logits, labels

    def get_logits_from_file(self, name) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.load(f'{self.logit_path}/{name}.pt')
        # assert data['model_names'] == self.model_names, 'Model names do not match'
        return data['logits'], data['labels']

    @staticmethod
    def rejection_rate(logits):
        preds = logits.argmax(dim=-1)
        return 1 - (preds[0, :] == preds[1:, :]).float().prod(0).mean()

    def ensemble_rejection(self, test_samples, test_seed, shift):
        # print(f'Running ensemble rejection on {test_samples} samples with seed {test_seed} and shift {shift}')
        test_labels, logits = self.prepare_data(test_samples, test_seed, shift)
        data = []

        rej = self.rejection_rate(logits)

        pval = binomtest(int(rej * (n := logits.shape[1])),
                         n,
                         self.baseline_rejection,
                         # alternative='greater'
                         ).pvalue

        significant = pval < 0.05
        test_acc = (logits.argmax(dim=-1) == test_labels).float().mean().item()

        data.append({'model': None, 'shift': shift, 'test_acc': test_acc,
                     'significant': significant, 'pvals': pval, 'test_samples': test_samples,
                     'test_seed': test_seed, 'algorithm': 'ensemble', 'rejection_rate': rej.item()})

        self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True)

    def ensemble_entropy(self, test_samples, test_seed, shift):
        # print(f'Running ensemble rejection on {test_samples} samples with seed {test_seed} and shift {shift}')
        test_labels, logits = self.prepare_data(test_samples, test_seed, shift)
        data = []

        pval = ks_2samp(self.ensemble_entropy_p1, ensemble_entropy(logits)).pvalue
        significant = pval < 0.05
        test_acc = (logits.argmax(dim=-1) == test_labels).float().mean().item()

        data.append({'model': None, 'shift': shift, 'test_acc': test_acc,
                     'significant': significant, 'pvals': pval, 'test_samples': test_samples,
                     'test_seed': test_seed, 'algorithm': 'ensemble_entropy'})

        self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True)

    def ensemble_sweep(self, test_samples, test_seeds, shifts=(True, False)):
        for test_seed in tqdm(test_seeds):
            for test_sample in test_samples:
                for shift in shifts:
                    self.ensemble_rejection(test_sample, test_seed, shift)
                    self.ensemble_entropy(test_sample, test_seed, shift)
        self.df.to_json(os.path.join(self.df_path, 'shift.json'))

    def prepare_data(self, test_samples, test_seed, shift):
        if shift == 'val':
            test = self.val
        elif shift:
            test = self.q_data
        elif not shift:
            test = self.p2
        else:
            raise ValueError('shift must be either "val", "True", or "False"')

        logits, labels = test
        rnd = random.Random(test_seed)
        idx = rnd.sample(range(logits.shape[1]), k=test_samples)
        logits = logits[:, idx]
        test_labels = labels[idx]
        return test_labels, logits


if __name__ == '__main__':
    from models import pretrained
    from data.camelyon import CamelyonModule

    dm = CamelyonModule(test_samples='all', batch_size=1024)

    models, names = pretrained.all_camelyon_model(return_names=True, device='cuda:2', wilds=False)
    bbsd = ShiftDetection(models=models, model_names=names, datamodule=dm, df_path='tables/camelyon',
                          logit_path='logits/camelyon', load_logits=True)
    bbsd.ensemble_sweep(test_samples=[10, 20, 50], test_seeds=range(100), shifts=(True, False))
    bbsd.bbsd_sweep(test_samples=[10, 20, 50], test_seeds=range(100), shifts=(True, False))
    bbsd.print_results()
