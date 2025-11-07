from ner_dataset import lang2label, name2tensor
import torch
from pathlib import Path
from BiLSTM import BiLSTM
from ner_dataset import num_chars, num_labels


def model_predict(model, name: str):
    label2lang = {label.item(): lang for lang, label in lang2label.items()}
    model.eval()
    with torch.no_grad():
        # input shape: (batch_size, seq, input_size)
        name = name2tensor(name).unsqueeze(0)
        output = model(name)
        pred = torch.argmax(output, dim=-1)
    return label2lang[pred.item()]


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
