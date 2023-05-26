import torch.nn as nn


class ProbeEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # input_size is the number of features in each probe request
        # hidden_size and output_size are design choices
        super(ProbeEncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.encoder_lstm = nn.LSTM(hidden_size, output_size, num_layers=3, dropout=0.5, batch_first=True)

    def forward(self, x):
        x = self.encoder(x)
        x, _ = self.encoder_lstm(x)

        return x
