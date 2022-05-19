from data.cifar10 import CIFAR10DataModule
from data.camelyon import CamelyonModule
import torch
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from rejectron.pqmodule import PQModule
import torchmetrics
from rejectron.training_utils import MetricDrop
import pandas as pd


class Model(pl.LightningModule):
    def __init__(self, n_test):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(512, 2)
        self.weights = torch.nn.Parameter(data=torch.tensor([4 / n_test, 1.]), requires_grad=False)
        self.corr = torch.nn.Parameter(data=torch.tensor([-.1, .1]), requires_grad=False)
        self.cnf = torchmetrics.ConfusionMatrix(2)
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y[y >= 0] = 1
        y[y < 0] = 0

        y_hat = self(x) + self.corr
        loss = F.cross_entropy(y_hat, y, weight=self.weights)
        self.cnf.update(y_hat.argmax(-1), y)
        return {'loss': loss}

    def on_train_epoch_end(self, **kwargs):
        cnf = self.cnf.compute()
        print(f"""\n\nConfusion matrix:
{cnf[0, 0]} | {cnf[0, 1]}\n{cnf[1, 0]} | {cnf[1, 1]}\n""")
        self.cnf.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y[y >= 0] = 1
        y[y < 0] = 0
        y_hat = self(x) + self.corr
        pred = y_hat.argmax(-1)
        self.val_acc.update(pred, y)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        print(f"Validation accuracy: {acc}")
        self.log("val_acc", acc)
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def selective_classify(self, x):
        y = self(x) + self.corr
        y = y.argmax(-1)
        y[y == 0] = -1
        print('Selective classification:', y)
        return y


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default='cifar')
    args = parser.parse_args()
    samples = args.samples
    gpu = args.gpu
    dataset = args.dataset
    batch_accum = 1
    DM, batch_size = {'cifar': (CIFAR10DataModule, 2048), 'camelyon': (CamelyonModule, 1024)}[dataset]
    if gpu > 1:
        # deal w smaller gpus
        batch_size = batch_size // 2
        batch_accum = 2

    for shift in [True, False]:
        for seed in range(10):
            print(f"seed: {seed}, shift: {shift}, samples: {samples}")
            pl.seed_everything(seed)
            pq = PQModule(
                datamodule=DM(shift=False, test_samples=samples, negative_labels=True, test_seed=seed,
                              train_samples=50000, val_samples=10000),
                batch_size=batch_size,
                num_workers=16)
            df = []
            for i in range(10):
                model = Model(pq.n_test)
                metric = MetricDrop(monitor='val_acc', patience=0, mode='max', verbose=True,
                                    min_delta=0.05)
                metric.best_score = torch.tensor(1.)

                trainer = pl.Trainer(gpus=[gpu], max_epochs=10,
                                     callbacks=[metric], num_sanity_val_steps=0, accumulate_grad_batches=batch_accum)

                trainer.fit(model, datamodule=pq)
                pq.refine(model)
                test_reject = 1 - pq.n_test / samples
                res = dict(Step=i, test_reject=test_reject)
                df.append(res)
                if test_reject == 1:
                    print("Rejected all samples")
                    break
            df = pd.DataFrame(df)
            df.to_csv(f"checkpoints/{dataset}/domain/{samples=}_{seed=}_{shift=}.csv", index=False)
