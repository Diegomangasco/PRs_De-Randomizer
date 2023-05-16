#  Copyright (c) 2023 Diego Gasco (diego.gasco99@gmail.com), Diegomangasco on GitHub
import datetime

import pyshark
import logging
from hashlib import sha1
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

OUTPUT_DIRECTORY = "./features_files/"
CUT_INDEX = 20  # Derived from the theorem


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
        with open(file + ".txt", "r") as txt_reader:
            line = txt_reader.readline()
            while line:
                self._devices_IDs.append(int(line))
                line = txt_reader.readline()
        self._total_devices = max(self._devices_IDs)

        # Read from file using pyshark
        pyshark_packets = pyshark.FileCapture(file + ".pcap")

        fields = set()
        features = dict()

        # PARSE

        logging.info("Parsing .pcap file")

        start_time = None

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
            if not start_time:
                start_time = datetime.datetime.timestamp(pkt.sniff_time)
                features["time"] = list()
            features["time"].append(datetime.datetime.timestamp(pkt.sniff_time) - start_time)

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

        logging.info("Filling missing fields with mean strategy")

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

        # logging.info("Saving features inside .csv file")
        #
        # self._features.to_csv(OUTPUT_DIRECTORY + file + ".csv", sep=",", float_format="%.4f", index=False)

    def read_csv(self, file: str) -> None:
        """
        Function to read the .csv file if present.
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
        self._total_devices = max(self._devices_IDs) + 1

    def principal_component_analysis(self, n_components=7, explainable=False) -> None:
        """
        Computes the Principal Component Analysis and projects the features over the principal components.
        """

        def explainable_principal_component_analysis(pca: PCA, feature_names: list) -> None:
            # number of components
            n_pcs = pca.components_.shape[0]

            # get the index of the most important feature on each component
            most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

            # get the names
            most_important_names = [feature_names[most_important[i]] for i in range(n_pcs)]
            names = [most_important_names[i] for i in range(n_pcs)]
            df = pd.DataFrame(names)

            print("Most relevant features for PCA")
            print(df)
            print("Features variance ratio (%)")
            print(100 * pca.explained_variance_ratio_)
            print("Features singular values")
            print(pca.singular_values_)

        pca = PCA(n_components)
        self._features = pca.fit_transform(self._features)
        if explainable:
            features_names = list(self._features.keys())
            explainable_principal_component_analysis(pca, features_names)
