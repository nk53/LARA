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
#  ['Subject Type', 'Delta', 'Selected', 'Maximum', 'False Positives', 'False Negatives', 'Sensitivity', 'Specificity']
  ['Subject Type', 'Delta', 'Selected Delta', 'Delta with Maximum Accuracy', 'False Positives', 'False Negatives', 'Sensitivity', 'Specificity']
)

# Get relevant statistics
treatment = data.groupby(by='Subject Type')
average = treatment.mean()
sem = treatment.sem(ddof=0)

# plot deltas
average['Delta'].plot(kind='bar', yerr=sem['Delta'], color='#C0C0C0')
plt.xticks(rotation='horizontal')
plt.xlabel('')
plt.savefig('deltas.tif')

# plot sensitivity and specificity together
avg_f = average[['Sensitivity', 'Specificity']]
sem_f = sem[['Sensitivity', 'Specificity']]
colors_multiple = ['#C0C0C0', '#F0F0F0']
avg_f.plot(kind='bar', ylim=[0.0, 1.0], yerr=sem_f, color=colors_multiple)
plt.xticks(rotation='horizontal')
plt.xlabel('')
plt.savefig('senspec.tif')

# plot accuracy and max accuracy together
avges_cols = ['Selected Delta', 'Delta with Maximum Accuracy']
avg_a = average[avges_cols]
sem_a = sem[avges_cols]
avg_a.plot(kind='bar', ylim=[0.0, 1.0], yerr=sem_a, color=colors_multiple)
plt.legend(bbox_to_anchor=(1, 0), loc='lower right')
plt.xticks(rotation='horizontal')
plt.xlabel('')
plt.savefig('accuracy.tif')
