from glob import glob

from tqdm import tqdm

from models.classifier import TorchvisionClassifier
from models.distilbert import DistilBertClassifier
import torch
from utils.generic import key_replace

WILDS_PATH = '/voyager/projects/tomginsberg/wilds_models'
CKPT_PATH = '/voyager/projects/tomginsberg/detectron/checkpoints'


def all_camelyon_model(return_names=False, device='cuda:1', wilds=False, eval_=True):
    to_device = lambda x: x.to(device)
    if device is None:
        to_device = lambda x: x
    models = [to_device(camelyon_model(seed=s, wilds=wilds)) for s in tqdm(range(10))]
    if eval_:
        for m in models:
            m.eval()
    if not return_names:
        return models
    return models, [f'camelyon17_{"erm_densenet121" if wilds else "resnet18_pretrained"}_seed{seed}' for seed in
                    range(10)]


def camelyon_model(seed=0, wilds=False):
    """
    Loads the pretrained model from the Camelyon dataset.
    """
    if wilds:

        ckpt = f'{WILDS_PATH}/camelyon17_erm_densenet121_seed{seed}/best_model.pth'
        model = TorchvisionClassifier(
            model='densenet121',
            out_features=2,
        )
        model.load_state_dict(torch.load(ckpt)['algorithm'])
    else:
        ckpt = glob(f'{CKPT_PATH}/camelyon/baselines/camelyon_resnet18_pretrained_seed{seed}/*.ckpt')[0]
        model = TorchvisionClassifier.load_from_checkpoint(ckpt)

    return model


def distillbert_on_civilcomments(seed=0, lr='1e-6', device='cuda:1', group_by_y=False):
    """
    Loads the pretrained model from the DistillBERT on Civil Comments dataset.
    """
    model = DistilBertClassifier.from_pretrained(
        'distilbert-base-uncased', num_labels=2,
        state_dict=key_replace(
            torch.load(
                f'{WILDS_PATH}/civilcomments_distilbert_erm_{"lr" + lr if ~group_by_y else "group_by_y"}_seed{seed}/best_model.pth',
                map_location=torch.device(device))['algorithm']
            , 'model.')
    )
    return model.to(device)
