#!/usr/bin/env python
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

AN = pd.read_csv('AN.txt', sep='\t', index_col=0)
AH = pd.read_csv('AH.txt', sep='\t', index_col=0)
FN = pd.read_csv('FN.txt', sep='\t', index_col=0)
FH = pd.read_csv('FH.txt', sep='\t', index_col=0)

results = [AN, AH, FN, FH]

data = pd.concat(results)

treatment = data.groupby(by='subject type')
average = treatment.mean()
std = treatment.std()

# plot deltas
average['delta'].plot(kind='bar', yerr=std['delta'], color='#C0C0C0')
plt.title('Delta by Group')
plt.xlabel('Age and Treatment Group')
plt.ylabel('Selected Delta')
plt.savefig('deltas.tif')

# plot false pos/neg together
avg_f = average[['false positives', 'false negatives']]
std_f = std[['false positives', 'false negatives']]
colors_multiple = ['#C0C0C0', '#F0F0F0']
avg_f.plot(kind='bar', yerr=std_f, color=colors_multiple)
plt.title('False Categorization by Group')
plt.xlabel('Age and Treatment Group')
plt.ylabel('False Positives')
plt.savefig('falses.tif')

# plot accuracy and max accuracy together
avg_a = average[['accuracy', 'max accuracy']]
std_a = std[['accuracy', 'max accuracy']]
avg_a.plot(kind='bar', yerr=std_a, color=colors_multiple)
plt.title('Accuracy of Selected Delta VS. Maximum Theoretical Delta')
plt.ylabel('Accuracy')
plt.xlabel('Age and Treatment Group')
plt.savefig('accuracy.tif')
