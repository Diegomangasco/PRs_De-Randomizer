from pre_processing import *
from dimensionality_reduction import *
from clustering import *

pre_proc = PreProcessing()
#pre_proc.read_pcap("out_file")
pre_proc.read_txt("out_file")
data_frame = pre_proc.get_features()
out = principal_component_analysis(data_frame, 7, True)
print(pre_proc.get_devices_number())
OPTICS_algorithm(out, 7, True)
DBSCAN_algorithm(out, 7, True)
