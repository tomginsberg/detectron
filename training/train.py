from models.classifier import TorchvisionClassifier
from training.trainer import trainer
from data.camelyon import CamelyonModule
import wandb


#############################################################################
# Each function is a configuration for a training run.
# A function should not be changed after it's been run to ensure all code is reproducible
#############################################################################


def camelyon_train():
    dm = CamelyonModule(batch_size=512)
    for seed in range(10):
        model = TorchvisionClassifier(model='resnet18', out_features=2, pretrained=True, logger='accuracy')
        trainer(model=model, seed=seed, datamodule=dm, run_name=f'camelyon_resnet18_pretrained_seed{seed}',
                ckpt_path=f'checkpoints/camelyon/baselines', max_epochs=5, patience=3, look_for_checkpoints=False,
                log_every_n_steps=1)
        wandb.finish()


if __name__ == '__main__':
    camelyon_train()
