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


def load_data(input_path: str, validate_path: str, batch_size: int):
    pre_processing_train = rd.PreProcessing()
    pre_processing_train.read_pcap(input_path)

    pre_processing_validate = rd.PreProcessing()
    pre_processing_validate.read_pcap(validate_path)

    logging.info("Creating data loaders")

    input_train = pre_processing_train.get_features().to_numpy()
    label_train = array(pre_processing_train.get_devices_IDs())

    input_validate = pre_processing_validate.get_features().to_numpy()
    label_validate = array(pre_processing_validate.get_devices_IDs())

    train_data = torch.tensor(input_train, dtype=torch.float32)
    validate_data = torch.tensor(input_validate, dtype=torch.float32)
    train_dataset = CustomDataset(train_data, label_train)
    validate_dataset = CustomDataset(validate_data, label_validate)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

    train_features = input_train.shape[1]
    validate_features = input_validate.shape[1]

    assert train_features == validate_features

    return train_dataloader, validation_dataloader, train_features


def load_test(file_path: str):
    pre_processing = rd.PreProcessing()
    pre_processing.read_pcap(file_path)
    logging.info("Creating test loader")
    inputs = torch.tensor(pre_processing.get_features().to_numpy(), dtype=torch.float32)
    labels = array(pre_processing.get_devices_IDs())
    test_dataset = CustomDataset(inputs, labels)
    test_dataloader = DataLoader(test_dataset, batch_size=labels.shape[0], shuffle=False)

    return test_dataloader
