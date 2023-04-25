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


def DBSCAN_cross_validation(data: DataFrame, labels: list):
    metrics = ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "minkowski"]
    p = [1, 2, 3]
    eps = list(range(0.1, 5, 0.1))
    algorithm = ["auto", "ball_tree", "kd_tree", "brute"]
    min_samples = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    elements = [min_samples, eps, metrics, p, algorithm]
    combinations = list(product(*elements))

    best_res = 0.0
    best_params = None
    for comb in combinations:
        res = OPTICS_algorithm(data, comb, False)
        accuracy = evaluation(res, labels)
        if accuracy > best_res:
            best_res = accuracy
            best_params = comb

    return best_res, best_params


def OPTICS_cross_validation(data: DataFrame, labels: list):
    metrics = ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "minkowski"]
    p = [1, 2, 3]
    eps = list(range(0.1, 5, 0.1))
    xi = list(range(0, 1, 0.01))
    cluster_method = ["xi", "dbscan"]
    algorithm = ["auto", "ball_tree", "kd_tree", "brute"]
    min_samples = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_cluster_size = list(range(0, 1, 0.1))
    elements = [min_samples, eps, metrics, xi, min_cluster_size, p, cluster_method, algorithm]
    combinations = list(product(*elements))

    best_res = 0.0
    best_params = None
    for comb in combinations:
        res = OPTICS_algorithm(data, comb, False)
        accuracy = evaluation(res, labels)
        if accuracy > best_res:
            best_res = accuracy
            best_params = comb

    return best_res, best_params
