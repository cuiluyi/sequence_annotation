from argparse import ArgumentParser
from pathlib import Path

import torch

from BiLSTM import BiLSTM
from ner_dataset import char2id, tag2label, num_chars, num_labels
from constants import (
    EMBED_SIZE,
    HIDDEN_SIZE,
)


def sentence2tensor(sentence: str) -> torch.Tensor:
    indices = [char2id[char] for char in sentence if char in char2id]
    return torch.tensor(indices)


def model_predict(model, sentence: str):
    label2tag = {label: tag for tag, label in tag2label.items()}
    model.eval()
    with torch.no_grad():
        inputs = sentence2tensor(sentence).unsqueeze(0)  # (batch_size, seq)
        preds = model(inputs)
    return [label2tag[pred.item()] for pred in preds[0]]


if __name__ == "__main__":
    # load model
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained model checkpoint",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model = BiLSTM(
        vocab_size=num_chars,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=num_labels,
    )
    model.load_state_dict(torch.load(model_path))

    # Model predict
    while True:
        sentence = input("Input a sentence:")
        if sentence == "quit":
            break
        pred = model_predict(model, sentence)
        print(pred)
