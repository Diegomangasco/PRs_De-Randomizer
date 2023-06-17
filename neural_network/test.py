from load_data import *
from experiment import *

experiment = Experiment(None, 150, 50, 2.0, 1e-4, "cpu")
experiment.load_checkpoint('./last_checkpoint.pth')
data, gt = load_test("./input/test_120_3_mins", experiment.features)
dbscan = DBSCAN(eps=2.0, min_samples=4, metric="euclidean")
dbscan.fit(data)
dbscan_result = len(set(dbscan.labels_))
print("DBSCAN", dbscan_result)
greedy = experiment.greedy_clustering(data.shape[0], 4, data)
print("Greedy", greedy)