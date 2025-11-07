import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Args:
            input_size: Size of each input vector.
            hidden_size: Size of the hidden state.
        """
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.activation = nn.Tanh()
        self.activation = nn.Sigmoid()

    def forward(self, input, hidden_state=None):
        """
        Args:
            input: (batch_size, seq_length, input_size)
            hidden_state: (batch_size, hidden_size), initial hidden state. If None, initializes to zeros.

        Returns:
            output: (batch_size, seq_length, hidden_size), all hidden states.
            hidden_state: (batch_size, hidden_size), the last hidden state.
        """
        batch_size, seq_length, _ = input.shape

        # 初始化隐藏状态
        if hidden_state is None:
            # hidden_state = torch.zeros(batch_size, self.hidden_size)
            hidden_state = nn.init.kaiming_uniform_(
                torch.empty(batch_size, self.hidden_size)
            )

        output = []  # 存储所有时间步的隐藏状态
        for t in range(seq_length):
            # 取出当前时间步输入: (batch_size, input_size)
            input_step = input[:, t, :]

            # 拼接输入与隐藏状态: (batch_size, input_size + hidden_size)
            combined = torch.cat((input_step, hidden_state), dim=1)

            # 更新隐藏状态: (batch_size, hidden_size)
            hidden_state = self.activation(self.i2h(combined))

            # 收集当前时间步的隐藏状态
            output.append(hidden_state)

        # 将隐藏状态列表堆叠成张量: (batch_size, seq_length, hidden_size)
        output = torch.stack(output, dim=1)

        return output, hidden_state


if __name__ == "__main__":
    # 模型参数
    input_size = 10
    hidden_size = 20
    batch_size = 4
    seq_length = 5

    # 创建随机输入
    x = torch.randn(batch_size, seq_length, input_size)

    # 实例化模型
    rnn = SimpleRNN(input_size, hidden_size)

    # 前向传播
    hidden_states, final_hidden_state = rnn(x)

    # 检查输出形状
    print(
        "Hidden states shape:", hidden_states.shape
    )  # (batch_size, seq_length, hidden_size)
    print(
        "Final hidden state shape:", final_hidden_state.shape
    )  # (batch_size, hidden_size)
