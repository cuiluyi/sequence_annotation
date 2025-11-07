import torch
import torch.nn as nn


from simple_rnn import SimpleRNN

torch.manual_seed(42)


class RNNmodel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNmodel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = SimpleRNN(input_size, hidden_size)
        # or:
        # self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # -> x needs to be: (batch_size, seq, input_size)
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])  # shape: (batch_size, output_size)
        return output

