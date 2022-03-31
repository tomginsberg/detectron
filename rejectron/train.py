import os

import pandas as pd
from pytorch_lightning import seed_everything

from data.camelyon import CamelyonModule
from models import pretrained
from rejectron.pqmodule import PQModule
from rejectron.training_utils import rejectron_trainer, train_rejectors


def make_c():
    return pretrained.camelyon_model(seed=0, wilds=False)


if __name__ == '__main__':
    # define h
    dataset = 'camelyon'
    seed_range = range(10)
    sample_range = [10, 100]
    h = pretrained.camelyon_model(seed=0, wilds=False)
    try:
        df = pd.read_csv(f'checkpoints/{dataset}/rejectron_results.csv')
        print(f'Found {len(df)} results in {dataset}')
    except FileNotFoundError:
        df = pd.DataFrame()

    for shift in (True, False):
        for test_samples in sample_range:
            for test_seed in seed_range:
                if os.path.exists(f := f'checkpoints/{dataset}/rejectron_{test_seed=}_{test_samples=}_{shift=}'):
                    print(f'Skipping {f}')
                    continue
                seed_everything(test_seed)
                # define a PQDataModule

                dm = CamelyonModule(
                    test_seed=test_seed,
                    test_samples=test_samples,
                    shift=shift,
                    negative_labels=True,
                    small_dev_sets=False,
                    val_size=1000
                )
                pq = PQModule(
                    p=dm.train, p_prime=dm.val, q=dm.test,
                    batch_size=256, num_workers=96 // 2, drop_last=False
                )

                # get a default rejectron-old trainer
                trainer = rejectron_trainer(save_directory=f'{dataset}/rejectron_{test_seed=}_{test_samples=}_{shift=}',
                                            max_epochs=6, patience=3,
                                            run_name=f'{dataset}_rej_{test_seed=}_{test_samples=}_{shift=}',
                                            dryrun=False, gpus=[1])

                # train !
                result, _ = train_rejectors(
                    pq=pq, h=h, trainer=trainer, num_rejectors=10, patience=5,
                    logfile=f'checkpoints/{dataset}/rejectron_{test_seed=}_{test_samples=}_{shift=}',
                    make_c=make_c
                )

                result = result.iloc[-1].to_dict() | dict(test_samples=test_samples, test_seed=test_seed, shift=shift)
                df = df.append(result, ignore_index=True)
                df.to_csv(f'checkpoints/{dataset}/rejectron_results.csv', index=False)
                print(df)

# def camelyon():
#     return
#
#
# if __name__ == '__main__':
#     camelyon()
