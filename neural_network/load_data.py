from numpy import array
from torch.utils.data import Dataset, DataLoader
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


def load_data(file_path: str, batch_size: int):
    pre_processing = rd.PreProcessing()
    pre_processing.read_pcap(file_path)
    inputs = pre_processing.get_features().values
    labels = array(pre_processing.get_devices_IDs())
    train_data = inputs[:int(0.8*len(inputs))]
    train_labels = labels[:int(0.8*len(labels))]
    validate_data = input[int(0.8*len(inputs)):]
    validate_labels = input[int(0.8*len(labels)):]
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    validate_data = torch.tensor(validate_data, dtype=torch.float32)
    validate_labels = torch.tensor(validate_labels, dtype=torch.float32)
    train_dataset = CustomDataset(train_data, train_labels)
    validate_dataset = CustomDataset(validate_data, validate_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, validate_dataloader
