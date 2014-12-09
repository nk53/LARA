#!/usr/bin/env python
#import numpy as np
#import scipy
#import sys
#import os, errno
#import matplotlib.patches as patches
#import pandas as pd
#import matplotlib.pyplot as plt
#import time as t
#from numpy import NaN, Inf, arange, isscalar, isnan, asarray, array, newaxis
#from scipy import spatial, signal, fft, arange
#from pandas import Series, DataFrame
#from pandas.tools.plotting import autocorrelation_plot
#from PIL import Image
#from pyeeg import * 

import pdutils, pdmath, pdlearn
import os.path
import matplotlib.pyplot as plt
from numpy import array
from pandas import DataFrame

print "Notebook initalized"    

# Loop over each folder in fpaths.txt
Base_path = os.path.abspath('..')
fpaths = [line.strip() for line in open("fpaths.txt")]

for fpath in fpaths:
  File_path = "%s/%s" % (Base_path, fpath)
  print "Starting %s" % (File_path)
  
  loaded_stats = pdutils.load_files(File_path)
  
  data_orignal, data_edit, roi_param_original, roi_param_edit, im, roi_loc_lcpro, \
    roi_x_lcpro, roi_y_lcpro , roi_loc_orignal, roi_x_orignal, roi_y_orignal, events_x, \
    events_y = loaded_stats

  #handy way to math check the frame rate, since users can't be trusted to know it. 
  rate = data_orignal.index[1]-data_orignal.index[0]

  #transformation
  sg_setting = 11
  data_smooth = DataFrame(index = data_orignal.index)
  for label, column in data_orignal.iteritems():
      temp_list = column.tolist()
      temp_list_smooth = pdlearn.savitzky_golay(temp_list, sg_setting, 4)
      data_smooth[label] = temp_list_smooth  
  
  #graph a random Roi to check if the sav golay settings were ok.
  print "Plotting Roi11: Are the Savitzky-Golay settings OK?"
  label = 'Roi11'
  plt.plot(data_orignal.index, data_orignal[label], label = 'original', color = 'r')
  plt.plot(data_orignal.index, data_smooth[label], label = 'smooth', color = 'b')
  plt.plot(events_x[label], events_y[label], marker = "^", color="g", linestyle= "None")
  plt.title(label)
  plt.show()
  
  ##this is the tuner
  #print "Delta Tuner 1"
  #results_average, results_num = pdlearn.delta_tuner(data_smooth, 50, rate)
  #
  ##attempts to fit a curve to the '# of ROIs with events' plot
  ##ra_index = array(results_average.index)
  #
  ##ip is the inflection point; ip_x and ip_y are its x and y coordinates, respectively
  ##roi_num_fit, ip_x, ip_y = pdlearn.cubicRegression(ra_index, results_num)
  ##print ip_x, ip_y
  #
  ##delta tuner graphs
  #plt.subplot(211)
  #plt.title('Delta Tuner results')
  #plt.plot(results_average.index, results_average)
  #plt.ylabel('Average # of Events/ROI')
  #
  #plt.subplot(212)
  #plt.plot(results_average.index, results_num, linestyle='-')
  ##plt.plot(results_average.index, roi_num_fit, linestyle='--')
  ##plt.axvline(ip_x, linewidth=2, color='r')
  ##plt.plot(ip_x, ip_y, 'ko')
  ##plt.legend(['# ROIs', 'Curve Fit', 'Inflection Point', 'Delta=%.3f, %.0f ROIs' % (ip_x, ip_y)])
  #plt.ylabel('# of ROIs with events')
  #plt.xlabel('Delta')
  #plt.show()

  #this is the second tuner, with CHI built in
  epsilon = 50
  #generates the chi_table for the full epsilon set of deltas
  print "Delta Tuner 2"
  results_average, results_num, chi_table = pdlearn.delta_tuner2(data_smooth, epsilon, rate, events_x, loaded_stats)
  print "Fitting cubic spline"
  #graph using the call below
  chi_table.to_csv(r'%s/chi_table_tuner_epsilon-%s.csv'%(File_path, epsilon))
  results_average.to_csv(r'%s/Average_num_events_per_roi_tuner_epsilon-%s.csv'%(File_path, epsilon))
  results_num.to_csv(r'%s/Number_positive_ROIs_tuner_epsilon-%s.csv'%(File_path, epsilon))
  
  plt.subplot(411)
  plt.title('Delta Tuner results')
  plt.plot(results_average.index, results_average)
  plt.ylabel('Average # of Events/ROI')
  
  plt.subplot(412)
  plt.plot(results_average.index, results_num)
  #plt.axvline(ip_x) # approximate inflection point
  plt.ylabel('# of ROIs with events')
  
  plt.subplot(413)
  plt.plot(chi_table.index, chi_table.True_Positive, label = "True Positive", c = '#332288')
  plt.plot(chi_table.index, chi_table.True_Negative, label = "True Negative", c = '#88CCEE')
  plt.plot(chi_table.index, chi_table.False_Positive, label = "False Positive", c = '#999933')
  plt.plot(chi_table.index, chi_table.False_Negative, label = "False Negative", c = '#AA4499')
  plt.legend(loc=1,prop={'size':10})
  plt.ylabel('# of ROIs with events')
  #plt.xlabel('Delta')
  
  plt.subplot(414)
  trues    = chi_table.True_Positive + chi_table.True_Negative
  falses   = chi_table.False_Positive + chi_table.False_Negative
  sums     = trues + falses
  accuracy = trues / sums # type conversion not necessary
  plt.plot(chi_table.index, accuracy)
  #plt.axvline(ip_x)
  plt.ylabel('Accuracy')
  plt.xlabel('Delta')
  
  plt.show()
