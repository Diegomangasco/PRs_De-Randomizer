#  Copyright (c) 2023 Diego Gasco (diego.gasco99@gmail.com), Diegomangasco on GitHub

import scapy.all as sc
import pyshark
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

        # Read from file using pyshark
        pyshark_packets = pyshark.FileCapture(INPUT_DIRECTORY + file + ".pcap")

        fields = set()
        features = dict()

        # PARSE

        logging.info("Parsing .pcap file")

        # Parse every probe request
        for pkt in pyshark_packets:
            for i in range(len(pkt.layers)):
                keys = pkt.layers[i]._all_fields.keys()
                for k in keys:
                    value = pkt.layers[i]._all_fields.get(k)
                    if any(ch.isalpha() or ch == ":" for ch in value):
                        value = bytes(value, 'utf-8')
                        value = sha1(value).hexdigest()[:CUT_INDEX]
                        value = float.fromhex(value)
                    else:
                        # Fields that are digits
                        value = float(value)
                    # Add the new value to the dictionary
                    if k in features.keys():
                        features[k].append(value)
                    else:
                        fields.add(k)
                        features[k] = list()
                        features[k].append(value)

        # BUILD FEATURES MATRIX

        logging.info("Building features matrix")

        max_length = 0
        for k in fields:
            if len(features[k]) > max_length:
                max_length = len(features[k])

        for k in fields:
            if len(features[k]) != max_length:
                difference = max_length - len(features[k])
                features[k] += [np.nan for _ in range(difference)]

        features = np.array(list(features.values()))
        features = features.T

        # MISSING VALUES

        logging.info("Filling missing fields")

        mean_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")  # Strategy can change
        features = mean_imputer.fit_transform(features)

        # DATAFRAME CREATION

        logging.info("Dataframe creation")

        self._features = pd.DataFrame(features, columns=list(fields))

        # FEATURES SCALING

        logging.info("Z-score normalization")

        for column in self._features.columns:
            std = 1.0 if self._features[column].std() == 0.0 else self._features[column].std()
            self._features[column] = (self._features[column] - self._features[column].mean()) / std

        # DATASET STORAGE

        logging.info("Saving features inside .csv file")

        self._features.to_csv(OUTPUT_DIRECTORY + file + ".csv", sep=",", float_format="%.4f", index=False)

    def read_txt(self, file: str) -> None:
        """
        Function to read the 'packets.txt' file if present.
        :return None
        """
        
        self._features = pd.read_csv(OUTPUT_DIRECTORY + file + ".csv", sep=",")
        # Read the device IDs from the .txt file
        self._devices_IDs = list()
        with open(INPUT_DIRECTORY + file + ".txt", "r") as txt_reader:
            line = txt_reader.readline()
            while line:
                self._devices_IDs.append(int(line))
                line = txt_reader.readline()
        self._total_devices = max(self._devices_IDs)
