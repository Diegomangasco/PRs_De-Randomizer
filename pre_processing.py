#  Copyright (c) 2023 Diego Gasco (diego.gasco99@gmail.com), Diegomangasco on GitHub

import scapy.all as sc
import logging
from hashlib import sha1
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

INPUT_DIRECTORY = "./input_files/"
OUTPUT_DIRECTORY = "./features_files/"
CUT_INDEX = 20  # Derived from the theorem
ALPHA = 10  # Heuristic Parameter


class PreProcessing:
    """
    Class to read and clean the datas from .pcap files.

    The data structures used are:
        * self._features = a DataFrame with all the features organized in tabular format
        * self._total_devices = the number of total devices registered in the simulation
        * self._devices_list = the list of all devices that sent a probe
    """

    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self._features = None
        self._devices_IDs = None
        self._total_devices = None

    def get_features(self) -> pd.DataFrame:

        assert self._features is not None
        return self._features.copy(deep=True)

    def get_devices_IDs(self) -> list:

        assert self._devices_IDs is not None
        return self._devices_IDs.copy()
    
    def get_devices_number(self) -> int:

        assert self._total_devices is not None
        return self._total_devices

    def read_pcap(self, file: str) -> None:
        """
        Function to read the .pcap files and store the result in a .txt file.

        :param file: a string that indicates the name of the input/output files.
        :return None
        """

        # READ FILES

        logging.info("Reading .pcap and .txt files")

        # Read the device IDs from the .txt file
        self._devices_IDs = list()
        with open(INPUT_DIRECTORY + file + ".txt", "r") as txt_reader:
            line = txt_reader.readline()
            while line:
                self._devices_IDs.append(int(line))
                line = txt_reader.readline()
        self._total_devices = max(self._devices_IDs)

        # Read from file using scapy
        capture = sc.rdpcap(INPUT_DIRECTORY + file + ".pcap")

        # tmp_packets contains all the probe requests for the selected device
        scapy_packets = [cap.show2(dump=True) for cap in capture]
        print(scapy_packets[0])

        # Create a data structure for storing packets
        # Each element (probe request) has its own dictionary
        # Inside the dictionary I have a key for each layer field and the relative value
        # In particular, there are fields with the same name in different layers
        # For these cases I use the combination of layer name and field name (i.e. layer-field) as the dictionary's keys
        packets = [dict() for _ in scapy_packets]

        # PARSE

        logging.info("Parsing .pacp file")

        # Parse every probe request
        for i, pkt in enumerate(scapy_packets):
            lines = pkt.split('\n')
            layer = None
            # Go inside the packet if is length enough
            for line in lines:
                if "###" in line and "|###" not in line:
                    # Problem: Some fields have the same name inside different layers
                    # Solution: Use the layer name to create the key for the specific field inside the dictionary
                    layer = line.strip('#[] ').strip().replace(" ", "_")
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
                        value = float(value) if len(value) > 0 else None
                    # Add the new value to the dictionary
                    if value:
                        packets[i][layer + "-" + key.strip('| ')] = value

        # BUILD FEATURES MATRIX

        logging.info("Building features")

        # Take one packet with the maximum number of fields
        fields = set(max(packets, key=lambda x: len(x.keys())).keys())
        # Create a dictionary with one list for each feature
        # This will be the base for the features matrix
        features = {key: list() for key in fields}
        # Fill all the missed fields with the median
        # If some frame has more than ALPHA missing fileds, ignore it
        for pkt in packets:
            keys = set(pkt.keys())
            # Check which are the missed features in the packet
            diff = fields.difference(keys)
            # Fill the existing features
            if len(diff) < ALPHA:
                for k in keys:
                    features[k].append(pkt[k])
                # Fill the missing fields
                for d in diff:
                    features[d].append(np.nan)

        fields = list(features.keys())
        print(fields)
        features = np.array(list(features.values()))
        features = features.T

        # MISSING VALUES

        logging.info("Filling missing fields")
        
        mean_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")  # Strategy can change
        features = mean_imputer.fit_transform(features)

        # FEATURES SCALING

        logging.info("Scaling features")

        std_scaler = StandardScaler()
        features = std_scaler.fit_transform(features)

        # DATASET SAVING

        logging.info("Saving features in memory")

        self._features = pd.DataFrame(features, columns=fields)

        # DATASET STORAGE

        logging.info("Saving features inside .txt file")

        np.savetxt(OUTPUT_DIRECTORY + file + ".txt", features, fmt='%1.7f', header=" ".join(fields))

    def read_txt(self, file: str) -> None:
        """
        Function to read the 'packets.txt' file if present.
        :return None
        """
        
        self._features = pd.read_csv(OUTPUT_DIRECTORY + file + ".txt", sep = " ", comment="")
        # Read the device IDs from the .txt file
        self._devices_IDs = list()
        with open(INPUT_DIRECTORY + file + ".txt", "r") as txt_reader:
            line = txt_reader.readline()
            while line:
                self._devices_IDs.append(int(line))
                line = txt_reader.readline()
        self._total_devices = max(self._devices_IDs)
