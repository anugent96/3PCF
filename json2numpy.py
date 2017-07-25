"""
Code converts JSON dataset into numpy array.
Works for 3PCF Python Code
"""

# Imports
import sys
import numpy as np
import json


# Load JSON File and the dataset
threepcf_data = str(sys.argv[1])
json_file = open(threepcf_data)
dataset = json.load(json_file)

# Extract the necessary parts of the dataset (poles)
data = dataset[u'poles'][u'__data__']

# Convert poles data to numpy arrays
d0 = np.array(data[0])
d1 = np.array(data[1])
d2 = np.array(data[2])
d3 = np.array(data[3])
d4 = np.array(data[4])
d5 = np.array(data[5])
d6 = np.array(data[6])
d7 = np.array(data[7])
d8 = np.array(data[8])
d9 = np.array(data[9])
