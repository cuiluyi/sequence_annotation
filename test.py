from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from BiLSTM import BiLSTM
from ner_dataset import NERDataset, num_chars, num_labels
from constants import (
    EMBED_SIZE,
    HIDDEN_SIZE,
    BATCH_SIZE,
    TEST_DATA_FILE,
)


def model_test(model: nn.Module, test_loader: DataLoader):
    # Accumulate totals across all batches
    num_correct = 0
    num_total = 0
    num_correct_ignore_majority = 0
    num_total_ignore_majority = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            preds = model(inputs)
            num_correct += (preds == labels).sum().item()
            num_total += labels.numel()

            for pred, label in zip(preds.view(-1), labels.view(-1)):
                if pred.item() == 0:
                    continue
                num_total_ignore_majority += 1
                if pred.item() == label.item():
                    num_correct_ignore_majority += 1

    # Safely print accuracies (avoid division-by-zero)
    if num_total_ignore_majority > 0:
        print(
            f"Accuracy (ignore majority class): {num_correct_ignore_majority / num_total_ignore_majority * 100:.4f}%"
        )
    else:
        print("Accuracy (ignore majority class): N/A (no non-majority tokens)")

    if num_total > 0:
        print(f"Accuracy: {num_correct / num_total * 100:.4f}%")
    else:
        print("Accuracy: N/A (no tokens)")


if __name__ == "__main__":
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

    # test dataset
    test_data_file = Path(TEST_DATA_FILE)
    test_dataset = NERDataset(test_data_file)

    # test dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Model test
    model_test(model, test_loader)
