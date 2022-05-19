# Detectron: A learning based hypothesis test for covariate shift

## Setup
Full setup instructions in progress. For now `pip install -r requirements.txt`

## Train Base Classifiers
Generate initial predictors `python training/train.py [cifar | camelyon | uci]`

## Detectron
```bash
python rejectron/train.py --help

usage: train.py [-h] [--dataset {camelyon,cifar}] [--seeds SEEDS] [--test_samples TEST_SAMPLES [TEST_SAMPLES ...]] [--gpu GPU] [--num_workers NUM_WORKERS] [--batch_size BATCH_SIZE] [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES]
                [--domain_classifier DOMAIN_CLASSIFIER]

optional arguments:
  -h, --help            show this help message and exit
  --dataset {camelyon,cifar}, -d {camelyon,cifar}
                        dataset to run on
  --seeds SEEDS, -s SEEDS
                        range of seeds to use
  --test_samples TEST_SAMPLES [TEST_SAMPLES ...], -t TEST_SAMPLES [TEST_SAMPLES ...]
                        list of test samples to use
  --gpu GPU, -g GPU     cuda device to use
  --num_workers NUM_WORKERS, -w NUM_WORKERS
                        number of workers to use
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        batch size to use
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES, -ba ACCUMULATE_GRAD_BATCHES
                        batch size to use
  --domain_classifier DOMAIN_CLASSIFIER, -dc DOMAIN_CLASSIFIER
                        use domain classifier
```

Training Detectron using XGBoost can be found in `notebooks/xgb_uci.ipynb`

## Baselines
Generate baselines with `python shift_detection/experiments.py [cifar | camelyon | uci]`


## Evaluation
View results and plots in `notebooks/rej_eval.ipynb`
