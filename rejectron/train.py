import warnings
import pandas as pd
from data.camelyon import CamelyonModule
from data.cifar10 import CIFAR10DataModule
from models import pretrained
from rejectron.pqmodule import PQModule
from rejectron.training_utils import rejectron_trainer, train_rejectors
import os

warnings.filterwarnings("ignore")


def cifar(seeds=10, test_samples=[10, 20, 50], gpu=0, num_workers=16, batchsize=512, accumulate_grad_batches=1,
          domain_classifier=False,
          **kwargs):
    dataset = 'cifar'
    for shift in [True, False]:
        for seed in range(seeds):
            for samples in test_samples:
                print(f'Starting run with {shift=} {seed=} {test_samples=}')
                logfile = f'checkpoints/{dataset}/detectron/{seed=}_{test_samples=}_{shift=}.csv'
                if os.path.exists(logfile):
                    print(f'Found existing logfile {logfile}, skipping this run')
                    continue

                create_model = lambda: pretrained.resnet18_trained_on_cifar10(domain_classifier=domain_classifier)

                dm = CIFAR10DataModule(test_samples=samples, negative_labels=True, shift=shift, test_seed=seed,
                                       shift_types='frost-fog-snow', split_val=True, batch_size=batchsize)
                h = create_model()
                pq = PQModule(datamodule=dm, batch_size=1024, num_workers=num_workers)
                method = 'detectron' if not domain_classifier else 'domain'
                trainer = rejectron_trainer(dryrun=False, max_epochs=10, max_acc_drop=0.05, gpus=[gpu],
                                            save_directory=f'{dataset}/{method}/ckp/{seed=}_{samples=}_{shift=}',
                                            accumulate_grad_batches=accumulate_grad_batches)

                train_rejectors(pq, h, create_model=create_model, num_rejectors=10, trainer=trainer,
                                logfile=f'checkpoints/{dataset}/{method}/{seed=}_{samples=}_{shift=}.csv',
                                domain_classifier=domain_classifier)


def camelyon(seeds=10, test_samples=[10, 20, 50], gpu=0, num_workers=16, batchsize=512, accumulate_grad_batches=1,
             train_samples=50000, val_samples=10000, **kwargs):
    dataset = 'camelyon'
    for shift in [True, False]:
        for seed in range(seeds):
            for samples in test_samples:
                print(f'Starting run with {shift=} {seed=} {test_samples=}')
                logfile = f'checkpoints/{dataset}/detectron/{seed=}_{samples=}_{shift=}.csv'
                if os.path.exists(logfile):
                    print(f'Found existing logfile {logfile}, skipping this run')
                    continue

                create_model = pretrained.camelyon_model
                dm = CamelyonModule(test_samples=samples, negative_labels=True, shift=shift, test_seed=seed,
                                    train_samples=train_samples, val_samples=val_samples)
                h = create_model()
                pq = PQModule(datamodule=dm, batch_size=batchsize, num_workers=num_workers)
                trainer = rejectron_trainer(dryrun=False, max_epochs=10, max_acc_drop=0.05, gpus=[gpu],
                                            accumulate_grad_batches=accumulate_grad_batches,
                                            save_directory=f'{dataset}/detectron/ckp/{seed=}_{samples=}_{shift=}')
                train_rejectors(pq, h, create_model=create_model, num_rejectors=10, trainer=trainer,
                                logfile=logfile,
                                benchmark_file=f'tables/{dataset}_benchmark.csv')


experiments = {'camelyon': camelyon, 'cifar': cifar}
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, help='dataset to run on', choices=experiments.keys())
    parser.add_argument('--seeds', '-s', type=int, default=10, help='range of seeds to use')
    parser.add_argument('--test_samples', '-t', nargs='+', type=int, default=[10, 20, 50],
                        help='list of test samples to use')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='cuda device to use')
    parser.add_argument('--num_workers', '-w', type=int, default=16, help='number of workers to use')
    parser.add_argument('--batch_size', '-b', type=int, default=512, help='batch size to use')
    parser.add_argument('--accumulate_grad_batches', '-ba', type=int, default=1, help='batch size to use')
    parser.add_argument('--domain_classifier', '-dc', type=bool, default=False, help='use domain classifier')
    args = parser.parse_args()
    print('Dataset:', args.dataset)
    print('Seed:', args.seeds)
    print('Test Samples:', args.test_samples)
    print('GPU:', args.gpu)
    print('Num workers:', args.num_workers)
    print('Batch size:', args.batch_size)
    print('Batch accumulate:', args.accumulate_grad_batches)
    print('Domain classifier:', args.domain_classifier)
    experiments[args.dataset](**vars(args))
