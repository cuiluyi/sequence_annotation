import os
import string

import torch
from torch.utils.data import Dataset
from unidecode import unidecode

data_dir = "./data/names"

lang2label = {
    file_name.split(".")[0]: torch.tensor(i, dtype=torch.long)
    for i, file_name in enumerate(os.listdir(data_dir))
}
num_langs = len(lang2label)

char2idx = {letter: i for i, letter in enumerate(string.ascii_letters + " .,:;-'")}
num_letters = len(char2idx)


# Turn a name into a <name_length x num_letters>
def name2tensor(name):
    tensor = torch.zeros(len(name), num_letters)
    for i, char in enumerate(name):
        tensor[i][char2idx[char]] = 1
    return tensor


class NameDataset(Dataset):
    def __init__(self):
        self.x, self.y = [], []
        for file in os.listdir(data_dir):
            with open(os.path.join(data_dir, file)) as f:
                lang = file.split(".")[0]
                for line in f:
                    name = unidecode(line.rstrip())
                    try:
                        self.x.append(name2tensor(name))
                        self.y.append(lang2label[lang])
                    except KeyError:
                        pass
        self.n_samples = len(self.x)

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    print(name2tensor("abc"))

    print(unidecode("Ślusàrski"))

    # create dataset
    dataset = NameDataset()

    # get first sample and unpack
    features, labels = dataset[0]
    print(features, labels)
