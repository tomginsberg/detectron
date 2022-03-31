from torch.utils.data import Dataset


class FlippedLabels(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index][0], -self.dataset[index][1] - 1

    def __len__(self):
        return len(self.dataset)
