from models.classifier import TorchvisionClassifier, MLP
from training.trainer import trainer
from data.camelyon import CamelyonModule
from data.uci_heart import UCIHeartModule
import wandb
import fire


#############################################################################
# Each function is a configuration for a training run.
# A function should not be changed after it's been run to ensure all code is reproducible
#############################################################################

class Train:
    @staticmethod
    def camelyon_train():
        dm = CamelyonModule(batch_size=512, train_samples='all', val_samples='all')
        for seed in range(10):
            model = TorchvisionClassifier(model='resnet18', out_features=2, pretrained=True, logger='accuracy')
            trainer(model=model, seed=seed, datamodule=dm, run_name=f'camelyon_resnet18_pretrained_seed{seed}',
                    ckpt_path=f'checkpoints/camelyon/baselines', max_epochs=5, patience=3, look_for_checkpoints=False,
                    log_every_n_steps=1)
            wandb.finish()

    @staticmethod
    def camelyon_train_small(gpu=3, samples=100000):
        dm = CamelyonModule(batch_size=512, train_samples=samples, val_samples=1000)
        seed = 0
        model = TorchvisionClassifier(model='resnet18', out_features=2, pretrained=True, logger='accuracy')
        trainer(model=model, seed=seed, datamodule=dm,
                run_name=f'camelyon_resnet18_pretrained_seed{seed}_n{samples // 1000}k',
                ckpt_path=f'checkpoints/camelyon/baselines', max_epochs=10, patience=5, look_for_checkpoints=False,
                log_every_n_steps=10, gpus=[gpu])

    @staticmethod
    def uci_mlp(gpu=3):
        dm = UCIHeartModule(batch_size=128, num_workers=8)
        for seed in range(10):
            model = MLP(input_size=9, output_size=2, hidden_layers=[256] * 3, dropout=0.3, logger='auc')
            trainer(model=model, seed=seed, datamodule=dm, run_name=f'uci_mlp_seed{seed}',
                    ckpt_path=f'checkpoints/uci/baselines2', monitor='val_auc',
                    max_epochs=1000, patience=100, look_for_checkpoints=False, log_every_n_steps=20,
                    gpus=[gpu] if isinstance(gpu, int) else None, offline=True)
            wandb.finish()


if __name__ == '__main__':
    fire.Fire(Train)
