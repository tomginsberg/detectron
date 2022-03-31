from copy import deepcopy
from typing import Any, Optional

import pytorch_lightning as pl
import torch

from losses.ce_negative import ce_negative_labels_from_logits
from rejectron.metrics import RejectronMetric
from models import pretrained


# from detectron.plotting import plot_2d_decision_boundary


class RejectronStep(pl.LightningModule):
    def __init__(self,
                 h: pl.LightningModule,
                 c: pl.LightningModule,
                 n_train: int,
                 n_test: int,
                 beta=1,
                 alpha=None,
                 hparams=None,
                 **kwargs):
        """
        Rejectron Step module
        Args:
            h: is a model trained on dataset P
            c: will be trained to agree with h on P but disagree with h on Q
            **kwargs:
        """
        assert beta > 0, f'beta must be grater then zero, not {beta=}'

        super().__init__(**kwargs)

        # h will not be trained so can be set to eval mode
        self.h = h
        self.h.eval()

        self.c = c
        self.c.train()

        """
        * We are given a set of n_train examples from P and n_test examples from Q
        * We compute regular cross entropy loss on samples from P using labels from h -- L(P, h)
        * We compute cross entropy loss on samples from Q using target distributions equally weighted over 
            all classes but what h predicts -- L'(Q, h) 
        * Losses are combined using a weighted sum
            L(P, Q) = L(P, h) +  alpha * L'(Q, h) 
        * c is trained using this loss
        * The condition that should be satisfied is that c would rather agree with h on one more sample from P
            then disagree with it on every sample from Q
        * So if agreeing with one more sample on P gets a score of `1` then disagreeing with a sample on Q
            should get a score of no more then 1/(n_test + beta)
        * This ensures that even agreeing with every sample in Q only gets a score of n_test / (n_test + beta) < 1
            so long as beta > 0   
        """

        self.n_train = n_train
        self.n_test = n_test
        self.beta = beta
        self.alpha = 1 / (self.n_test + beta) if alpha is None else alpha
        self.rejectron_metric = RejectronMetric(beta=beta)
        self.rejectron_metric_val = RejectronMetric(beta=beta, val_metric=True)
        self.results = [dict(), dict()]
        self.epoch = 0

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        data, labels = batch
        with torch.no_grad():
            # find the predictions made by h on this batch
            h_pred = self.h(data).argmax(dim=1)

        c_logits = self.c(data)

        self.rejectron_metric.update(labels=labels, c_logits=c_logits, h_pred=h_pred)

        # flip h's labels for data points in Q (these will have negative labels in the batch)
        h_pred[labels < 0] = -h_pred[labels < 0] - 1
        loss = ce_negative_labels_from_logits(c_logits, h_pred, alpha=self.alpha)
        return loss

    def training_epoch_end(self, outputs) -> None:
        metric = self.rejectron_metric.compute()
        for k, v in metric.items():
            self.log(k, v)
        self.results[0] = metric
        self.epoch += 1
        self.rejectron_metric.reset()

    # def on_train_end(self) -> None:
    # plot_2d_decision_boundary(deepcopy(self.c).cpu(), title=f'Epoch {self.epoch}')

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        with torch.no_grad():
            # find the predictions made by h on this batch
            h_pred = self.h(data).argmax(dim=1)

        c_logits = self.c(data)

        self.rejectron_metric_val.update(labels=labels, c_logits=c_logits, h_pred=h_pred)

    def validation_epoch_end(self, *args, **kwargs):
        self.results[1] = self.rejectron_metric_val.compute()
        self.rejectron_metric_val.reset()

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.c.parameters(), lr=0.01, weight_decay=1e-4)

    def get_c(self):
        return self.c.eval()

    def selective_classify(self, x):
        self.h.eval()
        with torch.no_grad():
            y_h = torch.argmax(self.h(x), dim=1)

            self.c.eval()
            y_c = torch.argmax(self.c(x), dim=1)
            self.c.train()

            mask = y_c != y_h
            y_h[mask] = -1
            return y_h

    def get_results(self):
        return self.results
