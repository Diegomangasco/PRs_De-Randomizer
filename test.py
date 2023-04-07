from pre_processing import *
from dimensionality_reduction import *
import numpy as np
import pandas as pd

p = PreProcessing(True)
p.read_pcap()
p.read_json()
pkts = p.get_packets()
f = pd.DataFrame({key: value for key, value in pkts.items() if key != "Device"})
l = np.array(pkts["Device"])
principal_component_analysis(f, l, 7)
#linear_discriminant_analysis(f, l, 7)

