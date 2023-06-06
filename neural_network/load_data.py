from numpy import array
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import read_data as rd
import torch
import logging


class CustomDatasetTrain(Dataset):
    def __init__(self, data):
        inputs = list()
        labels = list()
        for pr, lab in data:
            inputs.append(pr)
            labels.append(lab)
        self.inputs = torch.tensor(array(inputs), dtype=torch.float32)
        self.labels = torch.tensor(array(labels), dtype=torch.int8)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        probe = self.inputs[idx]
        label = self.labels[idx]
        return probe, label


class CustomDatasetTest(Dataset):
    def __init__(self, data):
        self.inputs = data

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


def load_data(file_path: str, batch_size: int):
    pre_processing = rd.PreProcessing()
    pre_processing.read_pcap(file_path)
    split_value = 0.2
    data = defaultdict(list)

    logging.info("Creating data loaders")

    inputs = pre_processing.get_features()
    features = list(inputs.keys())
    inputs = inputs.to_numpy()
    labels = array(pre_processing.get_devices_IDs())
    _ = [data[lab].append(inputs[i][:]) for i, lab in enumerate(labels)]

    source_category_ratios = {label: len(probes) for label, probes in data.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {label: number / source_total_examples for label, number in source_category_ratios.items()}

    val_split_length = source_total_examples * split_value  # 20% of the training split used for validation

    train_data = list()
    validate_data = list()

    for label, probes in data.items():
        split_idx = round(source_category_ratios[label] * val_split_length)
        for i, pr in enumerate(probes):
            if i > split_idx:
                train_data.append([pr, label])
            else:
                validate_data.append([pr, label])

    train_dataset = CustomDatasetTrain(train_data)
    validate_dataset = CustomDatasetTrain(validate_data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dataloader, validation_dataloader, features


def load_test(file_path: str, features: list = None):
    pre_processing = rd.PreProcessing()
    pre_processing.read_pcap(file_path)

    logging.info("Creating test loader")

    inputs = pre_processing.get_features()
    if features is not None:
        if len(inputs.keys()) > len(features):
            inputs = inputs[features]
        elif len(inputs.keys()) < len(features):
            for i, f in enumerate(features):
                if f not in inputs.keys():
                    inputs.insert(i, f, array([-1.0 for _ in range(inputs.shape[0])]))
            inputs = inputs[features]
    inputs = inputs.to_numpy()
    test_data = torch.tensor(inputs, dtype=torch.float32)

    ground_truth = pre_processing.get_devices_number()

    return test_data, ground_truth
