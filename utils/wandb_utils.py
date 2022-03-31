import wandb


def init_steps():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/step")
    wandb.define_metric("val/*", step_metric="val/step")
