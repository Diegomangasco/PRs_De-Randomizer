from pre_processing import *
from dimensionality_reduction import *
import numpy as np
import pandas as pd

p = PreProcessing()
p.read_pcap("out_file_")
f = p.get_features()
n = p.get_devices_IDs()
tn = p.get_devices_number()
print("DataFrame")
print(f)