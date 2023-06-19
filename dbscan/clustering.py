from sklearn.cluster import DBSCAN
import numpy as np

def dbscan(data, min_samples, eps=2.0, metric="euclidean"):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = dbscan.fit_predict(data)
    labels = np.unique(labels)
    labels = np.delete(labels, [-1], axis=0)
    result = labels.shape[0]
    return result