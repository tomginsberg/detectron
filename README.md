# Detectron: A Learning Based Hypothesis Test for Covariate Shift

![](https://i.imgur.com/k7C9V1U.png)

## Setup

### Environment Setup

#### Pytorch

This project is built with `python 3.9.5` and `torch 1.9.0`. We suggest installing torch and its dependencies on in a
conda environment using the instructions found [here](https://pytorch.org/get-started/locally/):

```bash
conda create -n 'cov-shift' python=3.9.5
conda activate cov-shift
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

#### Other dependencies

As a pure python project all remaining dependencies are installed using `pip`.

```shell
pip install -r requirements.txt
```

#### Install datasets

We provide an interactive installer to fetch all the datasets used in our experiments. Start by specifying the
environment variable `DATA_ROOT` as the full path to the directory where you want to save the datasets, then run the
installer. This can be done in one command

```shell
DATA_ROOT=/path/to/data/root python data/install.py
```

If installing manually you must update `data/paths.json` to point to the correct paths.

## Train Base Classifiers
The work involves detecting if covariate shifts degrade the performance of classifier, 
hence the first step to establishing results is to train the base classifiers. 
We provide simple training script to reproduce our base classifiers.

Generate initial predictors `python training/train.py [cifar | camelyon | uci]`

Update `CKPT_PATH` in `models/pretrained.py` to match the full path of where checkpoints are written to by `training/train.py`. 

## Detectron

Running our experiments on Camelyon and CIFAR10 can be done using `rejectron/train.py` 
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

Training Detectron using XGBoost on the `uci_heart` dataset can be found in `notebooks/xgb_uci.ipynb`

## Baselines

Generate baselines with `python shift_detection/experiments.py [cifar | camelyon | uci]`

## Evaluation

View results and plots in `notebooks/rej_eval.ipynb`
