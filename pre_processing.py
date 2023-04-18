#  Copyright (c) 2023 Diego Gasco (diego.gasco99@gmail.com), Diegomangasco on GitHub

import scapy.all as sc
import logging
from hashlib import sha1
import numpy as np
import pandas as pd

MIN_FIELDS = 1
INPUT_DIRECTORY = "./pcap_files/"
OUTPUT_DIRECTORY = "./features_files/"
CUT_INDEX = 20


def z_normalization(features: dict) -> None:
    """
    Function that applies the Z Normalization to the dataset.

    :param pkts: a list that contains dictionaries with keys (fields) and values
    :return: the normalized dataset
    """

    # Collect the dataset values
    values = list()
    for k in features.keys():
        if k != "Device":
            values += list(features[k])

    values = np.array(values)

    # Calculate dataset mean and standard deviation
    mu = values.mean(axis=0)
    standard_dev = values.std(axis=0)

    # Apply the Z Normalization
    # The dataset mean will be 0 and its standard deviation will be 1
    for k in features.keys():
        if k != "Device":
            for i in range(len(features[k])):
                features[k][i] = (features[k][i] - mu) / standard_dev


class PreProcessing:
    """
    Class to read and clean the datas from .pcap files.

    The data structures used are:
        * self._json: a boolean flag that tells if the .json file has been already written or not.
        * self._packets = a list that contains all the read packets.
    """

    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self._packets = pd.DataFrame()

    def get_packets(self) -> pd.DataFrame:
        return self._packets.copy(deep=True)

    def read_pcap(self, file: str) -> None:
        """
        Function to read the .pcap files and store the result in a .json file.

        :param append: boolean flag, if it is True, the method append the packets in an existing .json file.
        :return None
        """

        logging.info("Analysing pcap file: " + file)

        # READ FILES

        # Read the device ID from the .txt file
        devices_list = list()
        with open(INPUT_DIRECTORY + file + ".txt", "r") as txt_reader:
            line = txt_reader.readline()
            while line:
                devices_list.append(int(line))
                line = txt_reader.readline()

        # Read from file using scapy
        capture = sc.rdpcap(INPUT_DIRECTORY + file + ".pcap")

        # tmp_packets contains all the probe requests for the selected device
        scapy_packets = [cap.show2(dump=True) for cap in capture]

        # Create a data structure for storing packets
        # Each element (probe request) has its own dictionary
        # Inside the dictionary I have a key for each layer field and the relative value
        # In particular, there are fields with the same name in different layers
        # For these cases I use the combination of layer name and field name (i.e. layer-field) as the dictionary's keys
        packets = [dict() for _ in scapy_packets]

        # PARSE AND SCALING

        # Parse every probe request
        for i, pkt in enumerate(scapy_packets):
            lines = pkt.split('\n')
            layer = None
            packets[i]["Device"] = devices_list[i]
            # Go inside the packet if is length enough
            for line in lines:
                if "###" in line and "|###" not in line:
                    # Problem: Some fields have the same name inside different layers
                    # Solution: Use the layer name to create the key for the specific field inside the dictionary
                    layer = line.strip('#[] ')
                elif '=' in line:
                    # Add the value for the specific field
                    key, val = line.split('=', 1)
                    value = val.strip().replace("'", "").replace(" ", "").lower()
                    # If there are some alpha chars, use sha1
                    if any(ch.isalpha() for ch in value):
                        value = bytes(value, 'utf-8')
                        value = sha1(value).hexdigest()[:CUT_INDEX]
                        value = float.fromhex(value)
                    else:
                        # Fields that are digits
                        value = float(value) if len(value) > 0 else 0.0
                    # Add the new value to the dictionary
                    packets[i][layer + "-" + key.strip('| ')] = value

        # MISSING VALUES

        # Take one packet with the maximum number of fields
        fields = set(max(packets, key=lambda x: len(x.keys())).keys())
        # Create a dictionary with one list for each feature
        # This will be the base for the features matrix
        features = {key: list() for key in fields}
        # Fill all the missed fields with 0.0
        for pkt in packets:
            keys = set(pkt.keys())
            # Fill the existing features
            for k in keys:
                features[k].append(pkt[k])
            # Check which are the missed features in the packet
            diff = fields.difference(keys)
            # Fill the missing fields
            for d in diff:
                features[d].append(0.0)

        # NORMALIZATION

        # Apply Z Normalization to the dataset
        #z_normalization(features)

        # DATASET STORAGE

        with open(OUTPUT_DIRECTORY + file + ".txt") as txt_writer:
            txt_writer.write(", ".join(fields))
        features = np.array(list(features.values()))
        features = features.T
        np.savetxt(OUTPUT_DIRECTORY + file + ".txt", features, fmt='%1.7f')

    def read_txt(self, file: str) -> None:
        """
        Function to read the 'packets.json' file if present.
        :return None
        """
        
        features = dict()
        # TODO pandas readtxt
        with open(OUTPUT_DIRECTORY + file + ".txt", "r") as txt_reader:
            line = txt_reader.readline()
            while line:
                if "," not in line:
                    features = None
                else:
                    features = {k: list() for k in line.split(", ")}
                line = txt_reader.readline()
        self._packets = pd.read_json(OUTPUT_DIRECTORY + "packets.json", orient='columns')
