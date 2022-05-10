import os
import traceback
from collections import Callable
from glob import glob
from os import path
from os.path import join
from shutil import rmtree
from typing import Optional, List, Dict, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

from rejectron.pqmodule import PQModule
from rejectron.rejectronstep import RejectronStep
from rejectron.rejectronmodule import RejectronModule


class MetricDrop(pl.callbacks.EarlyStopping):
    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, str]:
        should_stop = False
        if current >= self.best_score.to(current.device) - self.min_delta:
            should_stop = False
            reason = f'Monitored metric {self.monitor}={current:.3f} ' \
                     f'which is >= baseline: {self.best_score:.3f} - tol: {self.min_delta}'
            # self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            reason = f'{self.monitor}={current:.3f} has fallen more then {100 * self.min_delta}% ' \
                     f'from the baseline {self.best_score}'
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor}={current:.3f} did not improve in the "
                    f"last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )

        return should_stop, reason


def rejectron_trainer(save_directory: str = None, run_name: str = None,
                      gpus=1, max_epochs=50, monitor: str = 'p_q_score',
                      patience: int = 15, dryrun=False, delete_previous=True,
                      max_acc_drop=0.05, **kwargs):
    if not dryrun:
        # assert run_name is not None, "Must provide a run name if dryrun is False"
        save_directory = os.path.join('checkpoints', save_directory)
        if path.isdir(save_directory):
            if delete_previous:
                print(f'Deleting previous run at {save_directory}')
                rmtree(save_directory)

    def f(iteration: int):
        callbacks = []
        if not dryrun:
            callbacks.append(pl.callbacks.ModelCheckpoint(
                save_directory,
                filename=f'c{iteration}_' + '{epoch}_{p_q_score:.5f}',
                save_top_k=1,
                save_last=True,
                monitor=monitor,
                mode='max',
                verbose=True,
            ))
        callbacks.append(pl.callbacks.EarlyStopping(monitor=monitor, patience=patience, mode='max', verbose=True))
        callbacks.append(
            MetricDrop(monitor='val_acc_all', patience=0, mode='max', verbose=True,
                       min_delta=max_acc_drop))
        return pl.Trainer(
            gpus=gpus, max_epochs=max_epochs, log_every_n_steps=1, num_sanity_val_steps=0,
            callbacks=callbacks,
            logger=False  # if dryrun else WandbLogger(project="pqlearning", offline=dryrun,
            #                          name=f'{run_name}_{iteration}'), **kwargs
        )

    return f


def benchmark(h: pl.LightningModule, pq: PQModule, gpus, skip_train=False, cached=None) -> pd.DataFrame:
    if isinstance(cached, str):
        try:
            return pd.read_csv(cached)
        except FileNotFoundError:
            print(f'Could not find cached results at {cached}')

    print('Benchmarking h')
    t = pl.Trainer(gpus=gpus)
    tr = float('nan') if skip_train else t.validate(h, pq.p_base_dataloader(), verbose=False)[0]['val_acc']
    va = t.validate(h, pq.val_dataloader(), verbose=False)[0]['val_acc']
    te = t.validate(h, pq.test_dataloader(), verbose=False)[0]['val_acc']
    df = pd.DataFrame([dict(Step=0,
                            train_acc=tr,
                            train_acc_all=tr,
                            train_agree=1,
                            test_reject=0,
                            test_acc=te,
                            test_acc_all=te,
                            p_q_score=100,
                            val_agree=1,
                            val_acc=va,
                            val_acc_all=va)]
                      )
    return df


def train_rejectors(pq: PQModule,
                    h: pl.LightningModule,
                    create_model: Callable[[], pl.LightningModule],
                    trainer: Callable[[int], pl.Trainer],
                    num_rejectors: int = 10,
                    logfile: Optional[str] = None,
                    patience: int = 5,
                    alpha=None,
                    benchmark_file: Optional[str] = None):
    size_of_q = len(pq.q)
    count = 0
    os.makedirs(os.path.split(logfile)[0], exist_ok=True)

    df = benchmark(h, pq, trainer(0).gpus, cached=benchmark_file)
    init_val_acc = df['val_acc_all'].item()
    print(df.to_string())
    # rm = RejectronModule(h)

    for i in range(1, num_rejectors + 1):
        h = create_model()
        c = create_model()
        c_step = RejectronStep(h, c=c, n_train=pq.n_train, n_test=pq.n_test, alpha=alpha,
                               )
        tr = trainer(i)
        for idx, _ in enumerate(tr.callbacks):
            if isinstance(tr.callbacks[idx], MetricDrop):
                tr.callbacks[idx].best_score = torch.tensor(init_val_acc)

        tr.fit(model=c_step, datamodule=pq)
        wandb.finish()
        # rm.add_new_c(c_step)

        lq_1 = len(pq.q)
        pq.refine(rs=c_step)
        lq_2 = len(pq.q)

        if lq_2 < lq_1:
            size_of_q = lq_2
        else:
            count += 1

        df = format_results(results=c_step.get_results(), df=df, i=i, pq=pq, size_of_q=size_of_q)

        if logfile is not None:
            # save a partial logfile
            df.to_csv(logfile.replace('.csv', '_partial.csv'), index=False)

        if count > patience:
            print(f'Rejectors have not improved for {patience} rounds. Stopping training.')
            break

        if lq_2 == 0:
            print(f'100% rejection. Stopping training.')
            break

    if logfile is not None:
        df.to_csv(logfile, index=False)

    os.remove(logfile.replace('.csv', '_partial.csv'))  # cleanup

    return df


def format_results(results: List[Dict[str, float]], df: pd.DataFrame, i: int, pq: PQModule, size_of_q: int):
    """
    Get the results of the current rejector and add them to the dataframe.
    Args:
        results: current results
        df: dataframe
        i: rejector index
        pq: datasets
        size_of_q: how mich q data is left

    Returns: dataframe with results appended

    """
    train_test, val = results
    # update q_disagree
    q_disagree = 1 - size_of_q / pq.q_base_length
    p_q_score = int(train_test['p_q_score']) + q_disagree
    res = ({'Step': i} | train_test | val)
    res.update(p_q_score=p_q_score, test_reject=q_disagree)
    df = pd.concat([df, pd.DataFrame([res])]).reset_index(drop=True)
    print(df.to_string())
    return df


def reformat_checkpoints(loc: str):
    for x in glob(join(loc, '*')):
        y = torch.load(x)
        y['state_dict'] = {k.replace('h.', ''): v for k, v in y['state_dict'].items()}
        torch.save(y, x)
