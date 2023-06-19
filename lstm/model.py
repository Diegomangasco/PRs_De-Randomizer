import torch.nn as nn


class ProbesEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # input_size is the number of features in each probe request
        # hidden_size and output_size are design choices
        super(ProbesEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.lstm = nn.LSTM(input_size, input_size, num_layers=2, batch_first=True, dropout=0.5)
        self.model_2 = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        x = self.model(x)
        x, _ = self.lstm(x)
        x = self.model_2(x)
        return x
