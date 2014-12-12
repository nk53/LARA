#!/usr/bin/env python
# This file generates histograms for the data in stats.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in the stats into a pandas DataFrame, close infile when done,
# discard filename and "type" columns
infile = "stats.txt"
with open(infile) as fh:
  summary = np.array([line.strip().split('\t') for line in fh])[:,2:]

# Setup the DataFrame
cols = summary[0]
#   summary is still in string format
data = np.array(summary[1:], dtype=float)
data = pd.DataFrame(data, columns=cols)

# Make histograms for each column as one figure
hist = data.hist(xlabelsize=10)

# We need smaller x-ticks

plt.savefig('histograms.tif')
