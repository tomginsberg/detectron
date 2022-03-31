import random
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from scipy.stats import ks_2samp, binomtest
from tqdm import tqdm

from data.base import DetectronDataModule


class EnsembleMethods:
    def __init__(self, models: List[pl.LightningModule], model_names: List[str], datamodule: pl.LightningDataModule,
                 df_path: str, logit_path: str, load_logits: bool = False, baseline_samples: int = 1000):
        self.models = [m.eval() for m in models]
        self.model_names = model_names
        self.datamodule = datamodule
        self.df = pd.DataFrame()
        self.df_path = df_path
        self.logit_path = logit_path

        if load_logits:
            self.q_data = self.get_logits_from_file('q')
            self.p_data = self.get_logits_from_file('p')
            self.val = self.get_logits_from_file('val')
        else:
            self.q_data = self.get_logits(datamodule.test_dataloader(), name='q')
            self.p_data = self.get_logits(datamodule.predict_dataloader(), name='p')
            self.val = self.get_logits(datamodule.val_dataloader(), name='val')

        rnd = random.Random(42)
        predict_idx = rnd.sample(range(len(self.p_data[-1])), k=baseline_samples)
        self.p1 = self.p_data[0][:, predict_idx], self.p_data[1][predict_idx]

        not_predict_idx = [i for i in range(len(self.p_data[-1])) if i not in predict_idx]
        self.p2 = self.p_data[0][:, not_predict_idx], self.p_data[1][not_predict_idx]

        self.baseline_rejection = self.rejection_rate(self.p1[0])

    def bbsd_sweep(self, test_samples, test_seeds, shifts=(True, False)):
        for test_seed in test_seeds:
            for test_sample in test_samples:
                for shift in shifts:
                    self.bbsd(test_sample, test_seed, shift)

    def print_results(self):
        print(self.df.groupby(['test_samples', 'shift', 'algorithm']).significant.mean())

    def bbsd(self, test_samples, test_seed, shift):
        # print(f'Running BBSD on {test_samples} samples with seed {test_seed} and shift {shift}')
        test_labels, logits = self.prepare_data(test_samples, test_seed, shift)
        data = []

        for (model_name, p_data, test_logits) in (
                tqdm(zip(self.model_names, self.p1[0], logits))):
            test_acc = (test_logits.argmax(dim=1) == test_labels).float().mean().item()
            pvals = [ks_2samp(p_data[:, j], test_logits[:, j]).pvalue for j in range(test_logits.shape[-1])]
            significant = min(pvals) < 0.05 / len(pvals)
            data.append({'model': model_name, 'shift': shift, 'test_acc': test_acc,
                         'significant': significant, 'pvals': pvals, 'test_samples': test_samples,
                         'test_seed': test_seed, 'algorithm': 'bbsd'})

        self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True)
        self.df.to_json(self.df_path + '.json')

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
        assert data['model_names'] == self.model_names, 'Model names do not match'
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

        pval = binomtest(int(rej * (n := logits.shape[1])), n, self.baseline_rejection, alternative='greater').pvalue
        significant = pval < 0.05
        test_acc = (logits.argmax(dim=-1) == test_labels).float().mean().item()

        data.append({'model': None, 'shift': shift, 'test_acc': test_acc,
                     'significant': significant, 'pvals': pval, 'test_samples': test_samples,
                     'test_seed': test_seed, 'algorithm': 'ensemble', 'rejection_rate': rej.item()})

        self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True)
        # self.df.to_json(self.df_path + '.json')

    def ensemble_sweep(self, test_samples, test_seeds, shifts=(True, False, 'val')):
        for test_seed in test_seeds:
            for test_sample in test_samples:
                for shift in shifts:
                    self.ensemble_rejection(test_sample, test_seed, shift)
        self.df.to_json(self.df_path + '.json')

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

    models, names = pretrained.all_camelyon_model(return_names=True, device='cuda:1', wilds=False)
    bbsd = EnsembleMethods(models=models, model_names=names, datamodule=dm, df_path='tables/camelyon/bbsd',
                           logit_path='logits/camelyon', load_logits=True)
    bbsd.ensemble_sweep(test_samples=[10, 20, 50, 100, 1000], test_seeds=range(100), shifts=(True, False, 'val'))
    bbsd.bbsd_sweep(test_samples=[10, 20, 50, 100, 1000], test_seeds=range(100), shifts=(True, False, 'val'))
    bbsd.print_results()
