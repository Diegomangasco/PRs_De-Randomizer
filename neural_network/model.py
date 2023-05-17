import torch.nn as nn


class ProbeEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # input_size is the number of features in each probe request
        # hidden_size and output_size are design choices
        super(ProbeEncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.encoder_lstm = nn.LSTM(hidden_size, output_size, num_layers=2, batch_first=True)
        self.decoder_lstm = nn.LSTM(output_size, hidden_size, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.BatchNorm1d(input_size),
            nn.ReLU()
        )

    def forward(self, x, train=True):
        if train:
            x = self.encoder(x)
            x, _ = self.encoder_lstm(x)
            x, _ = self.decoder_lstm(x)
            x = self.decoder(x)
        else:
            x = self.encoder(x)
            x, _ = self.encoder_lstm(x)

        return x
