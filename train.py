from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from loguru import logger

from BiLSTM import BiLSTM
from ner_dataset import NERDataset, num_chars, num_labels
from constants import (
    EMBED_SIZE,
    HIDDEN_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    SAVE_DIR,
    TRAIN_DATA_FILE,
)


def model_train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    num_epochs: int,
    save_dir: Path,
):
    model.train()
    n_iterations = len(train_loader)
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            loss, preds = model(inputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (preds == labels).float().mean()

            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_iterations}], Loss: {loss.item():.4f}, Accuracy: {acc.item()*100:.4f}%"
            )

        # save model checkpoints
        torch.save(model.state_dict(), save_dir / f"model_epoch_{epoch+1}.pth")
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train dataset
    train_data_file = Path(TRAIN_DATA_FILE)
    train_dataset = NERDataset(train_data_file)

    # train dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # model
    model = BiLSTM(
        vocab_size=num_chars,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=num_labels,
    ).to(device)

    # model train
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model_train(
        model,
        train_loader,
        optimizer,
        num_epochs=NUM_EPOCHS,
        save_dir=save_dir,
    )
