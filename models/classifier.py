import torch
import torchvision

from .model import Model


class TorchvisionClassifier(Model):
    """
    Wrapper for torchvision classifiers.
    """

    def __init__(self, model: str = 'resnet18', out_features: int = 2, pretrained=False, fc_attr=None,
                 logger='accuracy', loss='ce', loss_params=None, optim='adam',
                 optim_params=dict(lr=1e-3, weight_decay=1e-5),
                 scheduler=None,
                 scheduler_params=None,
                 ):

        self.save_hyperparameters()

        if not fc_attr:
            if 'resnet' in model:
                fc_attr = 'fc'
            elif any([n in model for n in ('densenet', 'inception', 'vgg')]):
                fc_attr = 'classifier'
            else:
                raise ValueError(f'Model {model} not supported without explicit value for {fc_attr}')

        model = torchvision.models.__dict__[model](pretrained=pretrained)
        model.__dict__['_modules'][fc_attr] \
            = torch.nn.Linear(model.__dict__['_modules'][fc_attr].in_features, out_features)

        super().__init__(model=model, logger=logger, loss=loss, loss_params=loss_params, optim=optim,
                         optim_params=optim_params,
                         scheduler=scheduler, scheduler_params=scheduler_params)


