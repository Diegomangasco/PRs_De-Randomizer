from read_data import *
from clustering import *

start_time = datetime.datetime.now().timestamp()
p = PreProcessing()
p.read_pcap("./input/test_100_3_mins")
result = dbscan(p.get_features(), min_samples=5)
print(result)
logging.info("Total time: {}".format(datetime.datetime.now().timestamp() - start_time))