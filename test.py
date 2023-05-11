from pre_processing import *
from dimensionality_reduction import *
from clustering import *
import sklearn

pre_proc = PreProcessing()
pre_proc.read_csv("out_file")
v = pre_proc.get_features().keys()
print(len(v))
data = read_principal_component_analysis_file("after_PCA_7")
# print(DBSCAN_cross_validation(data, pre_proc.get_devices_number()))
