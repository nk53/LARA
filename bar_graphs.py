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

# Adjust column labels so they will look nice when plotted
data.columns = pd.Index(
  ['Subject Type', 'Delta', 'Selected', 'Maximum', 'False Positives', 'False Negatives']
)

# Get relevant statistics
treatment = data.groupby(by='Subject Type')
average = treatment.mean()
std = treatment.std()

# plot deltas
average['Delta'].plot(kind='bar', yerr=std['Delta'], color='#C0C0C0')
plt.title('Delta by Group')
plt.xlabel('Age and Treatment Group')
plt.ylabel('Selected Delta')
plt.savefig('deltas.tif')

# plot false pos/neg together
avg_f = average[['False Positives', 'False Negatives']]
std_f = std[['False Positives', 'False Negatives']]
colors_multiple = ['#C0C0C0', '#F0F0F0']
avg_f.plot(kind='bar', ylim=[0, 90], yerr=std_f, color=colors_multiple)
plt.title('False Categorization by Group')
plt.xlabel('Age and Treatment Group')
plt.ylabel('False Positives')
plt.savefig('falses.tif')

# plot accuracy and max accuracy together
avges_cols = ['Selected', 'Maximum']
avg_a = average[avges_cols]
std_a = std[avges_cols]
avg_a.plot(kind='bar', yerr=std_a, color=colors_multiple)
plt.title('Accuracy of Selected Delta vs. Maximum Theoretical Accuracy')
plt.legend(bbox_to_anchor=(1, 0), loc='lower right')
plt.ylabel('Accuracy')
plt.xlabel('Age and Treatment Group')
plt.savefig('accuracy.tif')
