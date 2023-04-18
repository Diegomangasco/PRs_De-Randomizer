from pre_processing import *
from dimensionality_reduction import *
import numpy as np
import pandas as pd

p = PreProcessing()
p.read_pcap("packets")
pkts = p.get_packets()
f = pd.DataFrame({key: value for key, value in pkts.items() if key != "Device"})
l = np.array(pkts["Device"])
#out_pca = principal_component_analysis(f, 20)
#out_pca = pd.DataFrame(out_pca)
#out_lda = linear_discriminant_analysis(out_pca, l, 7)
#out_pca = principal_component_analysis(f, 7)
#print(out_pca)

#print(out_pca.shape)
#print(out_lda.shape)

