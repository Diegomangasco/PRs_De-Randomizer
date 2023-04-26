import logging

from sklearn.cluster import OPTICS, DBSCAN
from pandas import DataFrame
import numpy as np
from itertools import product
from model_evaluation import *


def DBSCAN_algorithm(
        data: DataFrame,
        parameters: tuple,
        explainable: bool) -> list:
    dbscan = DBSCAN(
        min_samples=parameters[0],
        eps=parameters[1],
        metric=parameters[2],
        p=parameters[3],
        algorithm=parameters[4]
    )
    data = np.array(data.values)
    clustering = dbscan.fit(data)
    single_labels = set(clustering.labels_)
    if explainable:
        logging.info(f"Clustering with DBSCAN algorithm gives: {len(single_labels)} clusters (i.e. different devices)")

    return clustering.labels_


def OPTICS_algorithm(
        data: DataFrame,
        parameters: tuple,
        explainable: bool) -> list:
    optics = OPTICS(
        min_samples=parameters[0],
        eps=parameters[1] if parameters[6] == "dbscan" else None,
        metric=parameters[2],
        xi=parameters[3] if parameters[6] == "xi" else 0.05,
        min_cluster_size=parameters[4] if parameters[6] == "xi" else None,
        p=parameters[5],
        cluster_method=parameters[6],
        algorithm=parameters[7]
    )
    data = np.array(data.values)
    clustering = optics.fit(data)
    single_labels = set(clustering.labels_)
    if explainable:
        logging.info(f"Clustering with OPTICS algorithm gives: {len(single_labels)} clusters (i.e. different devices)")

    return clustering.labels_


def DBSCAN_cross_validation(data: DataFrame, devices_number: int):
    metrics = ["cityblock", "euclidean", "l1", "l2", "manhattan", "minkowski"]
    p = [1, 2, 3]
    eps = [0.01, 0.1, 1, 10, 100, 1000]
    algorithm = ["auto", "ball_tree", "kd_tree", "brute"]
    min_samples = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    elements = [min_samples, eps, metrics, p, algorithm]
    combinations = list(product(*elements))

    print(f"Combinations: {len(combinations)}")

    best_err = 1e10
    best_params = None
    for i, comb in enumerate(combinations):
        if (i+1) % 50 == 0:
            print(f"Combination number: {i+1}")
        res = DBSCAN_algorithm(data, comb, False)
        err = clustering_error(res, devices_number)
        if err < best_err:
            print(f"Find best in combination number: {i+1}")
            best_err = err
            best_params = comb

    return best_err, best_params


def OPTICS_cross_validation(data: DataFrame, devices_number: int):
    metrics = ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "minkowski"]
    p = [1, 2, 3]
    eps = [0.01, 0.1, 1, 10, 100, 1000]
    xi = list(range(0.0, 1.0, 0.01))
    cluster_method = ["xi", "dbscan"]
    algorithm = ["auto", "ball_tree", "kd_tree", "brute"]
    min_samples = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_cluster_size = list(range(0.0, 1.0, 0.1))
    elements = [min_samples, eps, metrics, xi, min_cluster_size, p, cluster_method, algorithm]
    combinations = list(product(*elements))

    best_err = 1e10
    best_params = None
    for comb in combinations:
        res = OPTICS_algorithm(data, comb, False)
        err = clustering_error(res, devices_number)
        if err < best_err:
            best_err = err
            best_params = comb

    return best_err, best_params
