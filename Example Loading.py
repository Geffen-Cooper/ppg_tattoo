import os
import matplotlib.pyplot as plt
import sys
import csv

#Replace this path with wherever you saved the PPG Tools folder
sys.path.insert(0, r'ppgtools')

import numpy as np
from scipy import signal
from ppgtools.sigimport import importBIN, importEventMarkers, EventMarker
from ppgtools.biosignal import BioSignal

from ppgtools import sigpro, sigimport, sigplot, biometrics

import scipy.stats as stats
import copy
from sklearn.decomposition import FastICA
    
    
#%%
path = r"Data" #Put the directory you have the data in
filename = r"\artery" #Outer folder name of dataset (e.g., artery, vein, double)
devicename = "PPG Tattoo v3.2_DC.2B.5A.AE.9E.29"   #Don't change this

#Load in data
sessionData = sigimport.importTattooData(path, filename)

signals_original = sessionData[devicename]["Data"]
markers = sessionData[devicename]["Markers"]        

# sigplot.plot_biosignals(signals_original[0:8])

# plt.show()

# print(markers)

print(signals_original[0].data)