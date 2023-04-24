from sklearn.cluster import OPTICS
from pandas import DataFrame


def OPTICS_algorithm(data: DataFrame) -> None:
    optics = OPTICS()
    optics.fit(data.values)
    print(len(set(optics.labels_)))
