import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from pathlib import Path

from rnn import RNNmodel
from name_dataset import NameDataset
from name_dataset import num_langs, num_letters


def model_train(model, criterion, optimizer, num_epochs):
    model.train()
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

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset)
        print(
            "Epoch: {} Loss: {:.4f} Acc: {:.4f}".format(
                epoch + 1, epoch_loss, epoch_acc
            )
        )
    return model


def model_test():
    num_correct = 0

    model.eval()

    with torch.no_grad():
        for names, labels in test_loader:
            output = model(names)
            pred = torch.argmax(output, dim=-1)
            num_correct += (pred == labels).sum().item()

    print(f"Accuracy: {num_correct / len(test_dataset) * 100:.4f}%")


if __name__ == "__main__":
    # Create dataset
    dataset = NameDataset()

    train_ratio = 0.9

    # option 1 (use torch.utils.data.random_split to split dataset)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # # option 2 (use sklearn.model_selection.train_test_split to split dataset)
    # train_idx, test_idx = train_test_split(
    #     range(len(dataset)),
    #     test_size=(1 - train_ratio),
    #     stratify=dataset.y,
    # )
    # train_dataset = [(dataset.x[i], dataset.y[i]) for i in train_idx]
    # test_dataset = [(dataset.x[i], dataset.y[i]) for i in test_idx]

    # Create data_loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # Create model
    model = RNNmodel(input_size=num_letters, hidden_size=256, output_size=num_langs)

    # Model train
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model_train(model, criterion, optimizer, num_epochs=100)

    # Model test
    model_test()

    # Save model
    FILE = Path(__file__).parent / "model.pth"
    torch.save(model.load_state_dict(), FILE)
