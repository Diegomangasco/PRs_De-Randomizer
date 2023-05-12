from numpy import array
from torch.utils.data import Dataset, DataLoader
import read_data as rd
import torch
import logging


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
    logging.info("Creating data loaders")
    inputs = pre_processing.get_features().to_numpy()
    labels = array(pre_processing.get_devices_IDs())
    train_data = inputs[:int(0.8*len(inputs))][:]
    train_labels = labels[:int(0.8*len(labels))]
    validate_data = inputs[int(0.8*len(inputs)):][:]
    validate_labels = labels[int(0.8*len(labels)):]
    train_data = torch.tensor(train_data, dtype=torch.float32)
    validate_data = torch.tensor(validate_data, dtype=torch.float32)
    train_dataset = CustomDataset(train_data, train_labels)
    validate_dataset = CustomDataset(validate_data, validate_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, validation_dataloader
