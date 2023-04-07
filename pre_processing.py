#  Copyright (c) 2023 Diego Gasco (diego.gasco99@gmail.com), Diegomangasco on GitHub

import copy
import scapy.all as sc
import json
import logging
from hashlib import sha1
import numpy as np
import pandas as pd

MIN_FIELDS = 1
INPUT_DIRECTORY = "./pcap_files/"
OUTPUT_DIRECTORY = "./json_files/"
DEVICES = {
    "huawei": 0,
    "ipad": 1,
    "iphone6": 2,
    "iphone11": 3,
    "lenovo": 4,
    "macbookair": 5,
    "oneplus": 6,
    "samsung": 7,
    "xiaomi": 8,
}


def z_normalization(pkt_list: list[tuple[int, dict]]) -> None:
    """
    Function that applies the Z Normalization to the dataset.

    :param pkt_list: a list that contains dictionaries with keys (fields) and values
    :return: the normalized dataset
    """

    # Collect all the dataset values
    values = list()
    values += [list(pkt[1].values()) for pkt in pkt_list]

    # Transform the list in numpy array
    values = np.array(values)

    # Calculate dataset mean and standard deviation
    mu = values.mean()
    standard_dev = values.std()

    # Apply the Z Normalization
    # The dataset mean will be 0 and its standard deviation will be 1
    for pkt in pkt_list:
        for k in pkt[1].keys():
            pkt[1][k] = (pkt[1][k] - mu) / standard_dev

    # Test the normalization result
    new_values = list()
    new_values += [list(pkt[1].values()) for pkt in pkt_list]
    new_values = np.array(new_values)
    assert round(new_values.mean()) == 0.0
    assert round(new_values.std()) == 1.0


class PreProcessing:
    """
    Class to read and clean the datas from .pcap files.

    The data structures used are:
        * self._json: a boolean flag that tells if the .json file has been already written or not.
        * self._packets = a list that contains all the read packets.
    """

    def __init__(self, json_flag: bool):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self._json = json_flag
        self._packets = pd.DataFrame()

    def get_packets(self) -> pd.DataFrame:
        return self._packets.copy(deep=True)

    def read_pcap(self, append=False) -> None:
        """
        Function to read the .pcap files and store the result in a .json file.

        :param append: boolean flag, if it is True, the method append the packets in an existing .json file.
        :return None
        """

        # Set Scapy to default settings without the help of Wireshark dictionary

        # Instance of the dict and list that will contain all the probes read
        all_packets = dict()
        all_packets_tmp = list()

        # For each device present in list
        for dev in DEVICES.keys():
            logging.info("Analysing device: " + dev)
            # Read from file using scapy
            capture = sc.rdpcap(INPUT_DIRECTORY + dev + ".pcap")

            # tmp_packets contains all the probe requests for the selected device
            tmp_packets = [cap.show2(dump=True) for cap in capture]

            # Create a data structure for storing packets
            # Each element (probe request) has a dictionary
            # Inside the dictionary we have a key for each layer field and the relative value
            # In particular, there are fields with the same name in different layers, so we use the combination of
            # layer name and field name (i.e. layer-field) as the keys of the dictionary
            packets = [dict() for _ in tmp_packets]

            # PARSE AND SCALING
            # Parse every probe request
            for i, pkt in enumerate(tmp_packets):
                lines = pkt.split('\n')
                layer = None
                packets[i]["Device"] = dev
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
                        if any(ch.isalpha() for ch in value):
                            # If there are some alpha chars, use sha1
                            value = bytes(value, 'utf-8')
                            value = sha1(value).hexdigest()
                            # Reduce the number coming from sha1
                            value = float.fromhex(value) #% (10 ** 5)
                        else:
                            # Fields that are only digits
                            value = float(value) if len(value) > 0 else 0.0

                        # Add the new value to the dictionary
                        # Log scaling for the values
                        packets[i][layer + "-" + key.strip('| ')] = np.log10(value+1)

            # Set the total probes list by filtering the initial one
            all_packets_tmp += packets

        # MISSING VALUES
        # Take one packet with the maximum number of fields
        fields = set(max(all_packets_tmp, key=lambda x: len(x.keys())).keys())
        # Create one list for each feature
        features = {key: list() for key in fields}
        # Fill al the missed fields with 0.0
        for pkt in all_packets_tmp:
            keys = set(pkt.keys())
            for k in keys:
                # Fill the existing keys
                features[k].append(pkt[k])
            diff = fields.difference(keys)
            for d in diff:
                # Fill the missing fields
                features[d].append(0.0)

        # NORMALIZATION
        # Apply Z Normalization to the dataset
        #z_normalization(all_packets)

        # DATAFRAME CREATION
        df = pd.DataFrame(features, columns=list(fields))

        logging.info("Save the .pcap captures in the .json file")

        # DATAFRAME STORAGE
        df.to_json(OUTPUT_DIRECTORY + "packets.json", orient="columns")
        # Set the json flag to true
        self._json = True

    def read_json(self) -> None:
        """
        Function to read the 'packets.json' file if present.
        :return None
        """

        assert self._json is True
        self._packets = pd.read_json(OUTPUT_DIRECTORY + "packets.json", orient='columns')
