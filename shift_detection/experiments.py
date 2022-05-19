from data.camelyon import CamelyonModule
from data.uci_heart import UCIHeartModule
from data.cifar10 import CIFAR10DataModule
from models import pretrained
from shift_detection.shiftdetection import ShiftDetection
import numpy as np


def query_pvals(algo, samples, df):
    d = df.query(f'algorithm=="{algo}" and test_samples=={samples}')
    s = np.array(d.query('shift==True').pvals)
    n = np.array(d.query('shift==False').pvals)
    if algo == 'bbsd':
        s = np.array([min(x) for x in s])
        n = np.array([min(x) for x in n])
    return s, n


class Experiment:
    @staticmethod
    def camelyon():
        dm = CamelyonModule(test_samples='all', batch_size=1024, test_domain=5, val_samples='all')

        models, names = pretrained.all_camelyon_model(return_names=True, device='cuda:3', wilds=False)
        # models, names = None, range(10)
        em = ShiftDetection(models=models, model_names=names, datamodule=dm, df_path='tables/camelyon',
                            logit_path='logits/camelyon', load_logits=False)
        em.bbsd_sweep(test_samples=[10, 20, 50], test_seeds=range(100), shifts=(True, False))
        em.ensemble_sweep(test_samples=[10, 20, 50], test_seeds=range(100), shifts=(True, False))
        em.print_results()

    @staticmethod
    def cifar():
        dm = CIFAR10DataModule(test_samples='all', batch_size=1024, shift=True, negative_labels=False)

        models, names = pretrained.resnet18_collection_trained_on_cifar10(return_names=True, device='cuda:2',
                                                                          eval_=True)
        em = ShiftDetection(models=models, model_names=names, datamodule=dm, df_path='tables/cifar',
                            logit_path='logits/cifar', load_logits=True)
        em.bbsd_sweep(test_samples=[10, 20, 50], test_seeds=range(100), shifts=(True, False))
        em.ensemble_sweep(test_samples=[10, 20, 50], test_seeds=range(100), shifts=(True, False))
        em.print_results()

    @staticmethod
    def uci():
        rng = range(10, 101, 10)
        dm = UCIHeartModule(test_samples='all', batch_size=500, combine_val_and_test=True, shift=True,
                            negative_labels=False)
        models, names = pretrained.mlp_collection_trained_on_uci_heart(return_names=True, device='cuda:3')
        sh = ShiftDetection(models=models, model_names=names, datamodule=dm, df_path='tables/uci',
                            logit_path='logits/uci', load_logits=True, baseline_samples=250)
        sh.ensemble_sweep(test_samples=rng, test_seeds=range(1000), shifts=(True, False))
        sh.bbsd_sweep(test_samples=rng, test_seeds=range(1000), shifts=(True, False))
        sh.print_results()


if __name__ == '__main__':
    import fire
    fire.Fire(Experiment)

