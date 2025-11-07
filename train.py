from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from BiLSTM import BiLSTM
from ner_dataset import NERDataset, num_chars, num_labels


def model_train(
    model: nn.Module,
    train_loader: DataLoader,
    train_dataset: Dataset,
    criterion,
    optimizer,
    num_epochs: int,
):
    model.train()
    n_iterations = len(train_loader)
    for epoch in range(num_epochs):
        running_loss, running_corrects = 0.0, 0
        for i, (names, labels) in enumerate(train_loader):
            outputs = model(names)
            preds = torch.argmax(outputs, dim=-1)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += (preds == labels).sum().item()

            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_iterations}], Loss: {loss.item():.4f}"
            )

        epoch_loss = running_loss / n_iterations
        epoch_acc = running_corrects / len(train_dataset)
        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc*100:.4f}%"
        )
    return model


if __name__ == "__main__":
    # train dataset
    train_data_file = Path("data/train.txt")
    train_dataset = NERDataset(train_data_file)

    # train dataloader: batch_size=1 for simplicity (avoid padding and truncating)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

    # model
    model = BiLSTM(
        vocab_size=num_chars,
        embed_size=128,
        hidden_size=128,
        output_size=num_labels,
    )

    # Model train
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model_train(
        model,
        train_loader,
        train_dataset,
        criterion,
        optimizer,
        num_epochs=100,
    )

    # Save model
    FILE = Path(__file__).parent / "model.pth"
    torch.save(model.load_state_dict(), FILE)
