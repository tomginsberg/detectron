from models.pretrained import camelyon_model
import pandas as pd
from data.camelyon import CamelyonModule
import torch

model = camelyon_model(seed=0)
model.eval()
val_dl = CamelyonModule(val_samples=1000, batch_size=1024).val_dataloader()
torch.set_grad_enabled(False)


# Example of using the model to predict labels
# with torch.no_grad():
#     for (x, y) in val_dl:
#         yhat = model(x)
#         print('Accuracy:', (yhat.argmax(dim=1) == y).sum().item() / len(y))


def preprocess(model, val_dl) -> object:
    # TODO
    # find the important thresholds to make the method work !
    raise NotImplementedError


preprocess_info = preprocess(model, val_dl)


def is_flagged(model, x, preprocess_info) -> int:
    # TODO
    # returns 0 if not flagged, 1 if flagged
    raise NotImplementedError


def find_number_of_flagged_samples(seed, samples, shift,
                                   preprocess_info):
    dm = CamelyonModule(shift=shift,
                        negative_labels=False,
                        test_samples=test_samples,
                        test_seed=seed)
    td = dm.test_dataloader()
    flagged_samples = 0
    for (X, Y) in td:
        # X is a batch of samples, iterate over each one
        for x in X:
            flagged_samples += is_flagged(model, x, preprocess_info)

    return {'seed': seed, 'samples': samples,
            'shift': shift, 'flagged': flagged_samples}


results = []
for shift in [True, False]:
    for seed in range(10):
        for test_samples in [10, 20, 50]:
            res = find_number_of_flagged_samples(
                seed,
                test_samples,
                shift,
                preprocess_info=preprocess_info
            )
            results.append(res)

results = pd.DataFrame(results)
results.to_csv('results.csv', index=False)
