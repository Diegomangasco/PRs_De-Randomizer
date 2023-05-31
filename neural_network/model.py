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
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
