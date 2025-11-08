from ner_dataset import char2id, tag2label
import torch
from pathlib import Path
from BiLSTM import BiLSTM
from ner_dataset import num_chars, num_labels


def sentence2tensor(sentence: str) -> torch.Tensor:
    indices = [char2id[char] for char in sentence if char in char2id]
    return torch.tensor(indices)


def model_predict(model, sentence: str):
    label2tag = {label.item(): tag for tag, label in tag2label.items()}
    model.eval()
    with torch.no_grad():
        inputs = sentence2tensor(sentence).unsqueeze(0)  # (batch_size, seq)
        preds = model(inputs)
    return [label2tag[pred.item()] for pred in preds[0]]


if __name__ == "__main__":
    # load model
    FILE = Path(__file__).parent / "model.pth"
    model = BiLSTM(
        vocab_size=num_chars,
        embed_size=128,
        hidden_size=128,
        output_size=num_labels,
    )
    model.load_state_dict(torch.load(FILE))

    # Model predict
    while True:
        sentence = input("Input:")
        if sentence == "quit":
            break
        pred = model_predict(model, sentence)
        print(pred)
