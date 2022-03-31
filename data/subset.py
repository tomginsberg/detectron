from torch.utils.data import Dataset
import random


class Subsetable(Dataset):
    def __init__(self, dataset):
        self.indices = list(range(len(dataset)))
        self.dataset = dataset

    def refine_dataset(self, indices):
        """
        Refines the dataset by keeping only the indices specified in the argument.
        :param indices: list of indices to keep
        """
        self.indices = [self.indices[i] for i in indices]

    @staticmethod
    def refined(dataset, indices):
        """
        Returns a copy of the dataset with the indices refined.
        :return:
        """
        subset = Subsetable(dataset)
        subset.refine_dataset(indices)
        return subset

    def refine_to_amount(self, amount, random_seed=0, exclusion_amount=None, exclusion_seed=None):
        if exclusion_amount is None:
            rnd = random.Random(random_seed)
            self.indices = rnd.sample(self.indices, amount)
        else:
            # sample from the indices remaining without the exclusion indices
            rnd = random.Random(exclusion_seed)
            exclusion_indices = rnd.sample(self.indices, exclusion_amount)
            rnd = random.Random(random_seed)
            remaining_indices = [i for i in self.indices if i not in exclusion_indices]
            self.indices = rnd.sample(remaining_indices, amount)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]

    def __len__(self):
        return len(self.indices)

    def save_indices(self, path):
        with open(path, 'w') as f:
            for i in self.indices:
                f.write(str(i) + '\n')

    def load_indices(self, path):
        with open(path, 'r') as f:
            self.indices = [int(line) for line in f]


class DropMeta(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        return self.dataset[item][0], self.dataset[item][1]

    def __len__(self):
        return len(self.dataset)
