import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from glob import glob
from os.path import join


def trainer(run_name: str, ckpt_path: str, model: pl.LightningModule,
            datamodule: pl.LightningDataModule,
            seed: int = 0, look_for_checkpoints=False,
            gpus=(1,), max_epochs=100,
            monitor='val_acc', min_delta=0, patience=50, verbose=True, mode='max',
            project='detectron', offline=False,
            log_every_n_steps=10,
            ) -> pl.LightningModule:
    pl.seed_everything(seed, workers=True)

    ckpt = None
    if look_for_checkpoints:
        ckpt = glob(join(ckpt_path, run_name, '*.ckpt'))
        if len(ckpt) > 0:
            if len(ckpt) > 1:
                raise ValueError(f'Found more than one checkpoint for {ckpt_path}/{run_name}')
            print(f'Using checkpoint {ckpt[0]}')
            ckpt = ckpt[0]

    trainer = pl.Trainer(
        auto_select_gpus=True,
        gpus=gpus,
        max_epochs=max_epochs,
        deterministic=True,
        callbacks=[EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode),
                   ModelCheckpoint(dirpath=join(ckpt_path, run_name),
                                   monitor=monitor,
                                   save_top_k=1,
                                   verbose=verbose,
                                   mode=mode,
                                   ),
                   ],
        logger=WandbLogger(project=project, offline=offline, name=run_name),
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps,
    )

    # noinspection PyArgumentList
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt)

    # trainer.logger.log_hyperparams(model.hparams)
    # trainer.logger.log_hyperparams(datamodule.hparams)

    return model
