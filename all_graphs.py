#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from pdutils import normed_hist, normed_hist2

def get_binsize(bin_range, nbins):
  return (np.diff(bin_range) / float(nbins))[0]

AN = pd.read_csv('AN.txt', sep='\t', index_col=0)
AH = pd.read_csv('AH.txt', sep='\t', index_col=0)
FN = pd.read_csv('FN.txt', sep='\t', index_col=0)
FH = pd.read_csv('FH.txt', sep='\t', index_col=0)

results = [AN, AH, FN, FH]

category_data = pd.concat(results)
all_data = pd.read_csv('stats.txt', sep='\t')

# Adjust column labels so they will look nice when plotted
category_data.columns = pd.Index(
  ['Subject Type', 'Delta', 'Selected', 'Maximum', 'False Positives', 'False Negatives']
)

# Get relevant statistics
treatment = category_data.groupby(by='Subject Type')
average = treatment.mean()
sem = treatment.sem(ddof=0)

# plot deltas bar graph and histogram on the same figure
fig_delta, (axd_bar, axd_hist) = plt.subplots(2, 1, sharex=True, sharey=False)
average['Delta'].plot(kind='barh', xerr=sem['Delta'], ax=axd_bar, color='#C0C0C0')
axd_bar.set_ylabel('')

hist_args = {'bins': 10}
plot_args = {
  'linewidth': 3,
  'ax': axd_hist,
  'yticks': np.arange(0, .31, .10),
}
normed_hist2(all_data['delta'], hist_args, plot_args)
plt.savefig('delta.tif')

# plot false pos/neg together
#plt.figure(2)
#plt.subplot(121)
fig_falses, (axf_bar, axf_hist) = plt.subplots(2, 1, sharex=True, sharey=False)
avg_f = average[['False Positives', 'False Negatives']]
sem_f = sem[['False Positives', 'False Negatives']]
colors_multiple = ['#C0C0C0', '#F0F0F0']
avg_f.plot(kind='barh', xerr=sem_f, ax=axf_bar, color=colors_multiple)
axf_bar.set_ylabel('')
hist_args = {'bins': 10}
plot_args = {
  'linewidth': 3,
  'ax': axf_hist,
  'xticks': np.arange(0, 91, 20),
  'yticks': np.arange(0.05, 0.5, 0.1),
}
normed_hist2(all_data['false positives'], hist_args, plot_args)
normed_hist2(all_data['false negatives'], hist_args, plot_args)
axf_hist.legend(['False Positives', 'False Negatives'])
plt.savefig('falses.tif')

# plot accuracy and max accuracy together
fig_accuracies, (axa_bar, axa_hist) = plt.subplots(2, 1, sharex=True, sharey=False)
avges_cols = ['Selected', 'Maximum']
avg_a = average[avges_cols]
sem_a = sem[avges_cols]
avg_a.plot(kind='barh', xerr=sem_a, ax=axa_bar, color=colors_multiple)
#axa_bar.legend(bbox_to_anchor=(0, 0), loc='lower left')
axa_bar.set_ylabel('')
hist_args = {'bins': 10}
plot_args = {
  'linewidth': 3,
  'ax': axa_hist,
  'xticks': np.arange(0.2, 1.1, 0.2),
  'yticks': np.arange(0.05, 0.36, 0.1),
}
normed_hist2(all_data['accuracy'], hist_args, plot_args)
normed_hist2(all_data['max accuracy'], hist_args, plot_args)
axa_hist.legend(['Selected', 'Maximum'])
#axa_hist.legend(['Selected', 'Maximum'], bbox_to_anchor(0, 1), loc='upper left')
plt.savefig('accuracy.tif')

# Bin size statistics
ac_xrange = [0.2, 1.0]
binsize = get_binsize(ac_xrange, 10)
print "Accuracies Binsize", binsize
