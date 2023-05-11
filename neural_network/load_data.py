from numpy import array
from torch.utils.data import Dataset, DataLoader
import sys
import read_data as rd
import torch


class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


file = sys.argv[1]
pre_processing = rd.PreProcessing()
pre_processing.read_pcap(file)
inputs = pre_processing.get_features().values
labels = array(pre_processing.get_devices_IDs())
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)
dataset = CustomDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
