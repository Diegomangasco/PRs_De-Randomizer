import torch.nn as nn


class ProbeEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # input_size is the number of features in each probe request
        # hidden_size and output_size are design choices
        super(ProbeEncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),

            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.Tanh()

        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),

            nn.Linear(hidden_size, input_size),
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, input_size),
        )

    def forward(self, x, train=True):
        if train:
            x = self.encoder(x)
            x = self.decoder(x)
        else:
            x = self.encoder(x)

        return x
