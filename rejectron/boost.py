import pickle
import random
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.io import loadmat
from tqdm import tqdm
from multiprocessing import Pool
import os

df = loadmat('/voyager/datasets/UCI/uci_heart_processed.mat')
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'nthread': 4,
    'tree_method': 'gpu_hist'
}

bst = xgb.Booster(params)
bst.load_model('/voyager/datasets/UCI/xgb_seed=0.model')

data = {}
for key in ['train', 'iid_test', 'val', 'ood_test']:
    data[key] = xgb.DMatrix(df[f'{key}_data'], label=df[f'{key}_labels'])

record: dict[tuple, list[dict[str, Any]]] = {}
evallist = [(data['val'], 'eval')]
preds = {}


def run(seed):
    for shift in [True, False]:
        print(f'Seed {seed}, shift {shift} pid={os.getpid()}')
        test = f'{"ood" if shift else "iid"}_test_data'
        for samples in range(10, 101, 10):
            if os.path.exists(f'checkpoints/uci/detectron/{seed=}_{samples=}_{shift=}.csv') \
                    and os.path.exists(f'checkpoints/uci/detectron/preds_{seed=}_{samples=}_{shift=}.pkl'):
                print(f'Already exists: {seed=}_{samples=}_{shift=}.csv')
                continue

            rnd = random.Random(seed)
            rng = rnd.sample(range(df[test].shape[0]), samples)
            test_set = df[test][rng]
            orig_test_set = xgb.DMatrix(df[test][rng])

            n_test = samples
            test_data = xgb.DMatrix(test_set)

            test_pred = bst.predict(test_data) > 0.5
            data_ = np.concatenate([df['train_data'], test_set])
            labels = np.concatenate([df['train_labels'][0], 1 - test_pred])
            weights = np.concatenate([np.ones_like(df['train_labels'][0]), 1 / (n_test + 1) * np.ones(n_test)])

            pq_data = xgb.DMatrix(data_, label=labels, weight=weights)

            num_round = 10
            data_test = data[test.replace('_data', '')]
            record[(seed, samples, shift)] = [
                {
                    'iteration': 0,
                    'val_auc': eval(bst.eval(data['val']).split(':')[1]),
                    'test_auc': eval(bst.eval(data_test).split(':')[1]),
                    'test_reject': 0,
                }
            ]
            preds = []
            for i in (range(10)):
                params['seed'] = i
                c = xgb.train(params, pq_data, num_round, evals=evallist, verbose_eval=False)
                mask = ((c.predict(test_data) > 0.5) == test_pred)
                test_set = test_set[mask]
                preds.append(c.predict(orig_test_set))

                n_test = test_set.shape[0]
                record[(seed, samples, shift)].append(
                    dict(iteration=i + 1, val_auc=eval(c.eval(data['val']).split(':')[1]),
                         test_auc=eval(c.eval(data_test).split(':')[1]), test_reject=1 - n_test / samples, seed=seed,
                         samples=samples, shift=shift))
                if n_test == 0:
                    break

                test_data = xgb.DMatrix(test_set)
                test_pred = bst.predict(test_data) > 0.5
                data_ = np.concatenate([df['train_data'], test_set])
                labels = np.concatenate([df['train_labels'][0], 1 - test_pred])
                weights = np.concatenate([np.ones_like(df['train_labels'][0]), 1 / (n_test + 1) * np.ones(n_test)])

                pq_data = xgb.DMatrix(data_, label=labels, weight=weights)

            pd.DataFrame(record[(seed, samples, shift)]).to_csv(
                f'checkpoints/uci/detectron/{seed=}_{samples=}_{shift=}.csv', index=False)

            with open(f'checkpoints/uci/detectron/preds_{seed=}_{samples=}_{shift=}.pkl', 'wb') as f:
                pickle.dump(preds, f)


if __name__ == '__main__':
    for seed in tqdm(range(100)):
        run(seed)
    # pqdm(range(3), function=run, n_jobs=3)
