from pathlib import Path
import torch
from torch.utils.data import Dataset

corpus_files = ["data/train_corpus.txt", "data/test_corpus.txt"]

# construct language to label mapping
char2id = {}
for file in corpus_files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            for char in line.strip():
                if char not in char2id:
                    char2id[char] = len(char2id)
num_chars = len(char2id)

tag2label = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
}
num_labels = len(tag2label)


# Named Entity Recognition Dataset
class NERDataset(Dataset):
    def __init__(self, file_path: Path):
        self.x, self.y = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            chars, labels = [], []
            for line in f:
                if line.strip() == "":
                    self.x.append(chars)
                    self.y.append(labels)
                    chars = []
                    labels = []
                else:
                    char, label = line.strip().split()
                    try:
                        chars.append(char2id[char])
                        labels.append(tag2label[label])
                    except KeyError:
                        pass

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return torch.tensor(self.x[index]), torch.tensor(self.y[index])

    # we can call len(dataset) to return the size
    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    print(char2id["地"])

    train_file = Path("data/train.txt")
    train_dataset = NERDataset(train_file)

    features, labels = train_dataset[0]
    print(features)
    print(labels)

    # 268
    # tensor([ 0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    #         19, 20, 21, 22, 23, 13, 24, 25, 17, 26, 27, 28, 29, 30, 31, 32,  8, 33,
    #         34, 35, 36, 17, 37, 38, 39, 40, 34, 41, 42,  0, 43, 44])
    # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #         0, 0])
