from read_data import *
from clustering import *
from principal_component_analysis import *

start_time = datetime.datetime.now().timestamp()
p = PreProcessing()
p.read_pcap("./input/test_100_3_mins")
principal_component_analysis(p, components=7, explainable=True)
dbscan(p.get_features(), min_samples=5)
logging.info("Total time: {}".format(datetime.datetime.now().timestamp() - start_time))