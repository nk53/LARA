#!/usr/bin/env python

from pandas import Series, DataFrame
import pdutils, pdmath, pdlearn
import os, os.path
import matplotlib.pyplot as plt
from numpy import array
from pandas import DataFrame

print "Notebook initalized"    

# This line gets printed first in each stats_spec file
stats_header = '\t'.join(
  ['file', 'subject type', 'delta', 'accuracy',
   'max accuracy', 'false positives', 'false negatives']
)

# Where we write errors
logfile = "errlog.txt"
# The newline separator
nl = os.linesep
# Data superdirectory
Base_path = os.path.abspath('../data')

# Loop over all data, record stats by subject_type/injection
subject_types = ['AH', 'AN', 'FH', 'FN']
injections = ['Control', 'FPL', 'FPL & NIF']

stats_all = open('stats.txt', 'w')
stats_all.write(stats_header + nl)
for subject_type in subject_types:
  # open file for stats output, line buffered
  stats_spec = open('%s.txt' % (subject_type), 'w', 0)
  stats_spec.write(stats_header + nl)
  for injection in injections:
    type_path = os.path.join(Base_path, subject_type, injection, 'Analyzed')
    data_dirs = os.listdir(type_path)
    for data_dir in data_dirs:
      File_path = os.path.join(type_path, data_dir)
      print "Starting %s" % (File_path)
      
      # Sometimes the "ROI normalized.txt" file is missing
      try:
        loaded_stats = pdutils.load_files(File_path)
      except IOError as e:
        errstr = "I/O error({0}): {1} for file: {2}{3}".format(e.errno, e.strerror, File_path, nl)
        print errstr
        log_fh = open(logfile, 'a')
        log_fh.write(errstr)
        continue
      
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
      
      ##graph a random Roi to check if the sav golay settings were ok.
      #print "Plotting Roi11: Are the Savitzky-Golay settings OK?"
      #label = 'Roi11'
      #plt.plot(data_orignal.index, data_orignal[label], label = 'original', color = 'r')
      #plt.plot(data_orignal.index, data_smooth[label], label = 'smooth', color = 'b')
      #plt.plot(events_x[label], events_y[label], marker = "^", color="g", linestyle= "None")
      #plt.title(label)
      #plt.show()
      
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
      results_average, results_num, chi_table = \
        pdlearn.delta_tuner2(data_smooth, epsilon, rate, events_x, loaded_stats)
      print "Fitting cubic spline"
      
      smoothed_spline, second_derivative, inflection_point = \
        pdlearn.splineSmooth(array(results_num.index), array(results_num))
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
      plt.plot(smoothed_spline['xs'], smoothed_spline['ys'])
      plt.axvline(inflection_point['x'])
      plt.legend(['# ROIs', 'spline fit'])
      plt.ylabel('# of ROIs with events')
      
      plt.subplot(413)
      plt.plot(chi_table.index, chi_table.True_Positive, label = "True Positive", c = '#332288')
      plt.plot(chi_table.index, chi_table.True_Negative, label = "True Negative", c = '#88CCEE')
      plt.plot(chi_table.index, chi_table.False_Positive, label = "False Positive", c = '#999933')
      plt.plot(chi_table.index, chi_table.False_Negative, label = "False Negative", c = '#AA4499')
      plt.axvline(inflection_point['x'])
      plt.legend(loc=1,prop={'size':10})
      plt.ylabel('# of ROIs with events')
      
      plt.subplot(414)
      trues    = chi_table.True_Positive + chi_table.True_Negative
      falses   = chi_table.False_Positive + chi_table.False_Negative
      sums     = trues + falses
      accuracy = trues / sums # type conversion not necessary
      plt.plot(chi_table.index, accuracy)
      plt.axvline(inflection_point['x'])
      plt.ylabel('Accuracy')
      plt.xlabel('Delta')
      
      #plt.show()
      plt.savefig('plots/delta-%s.tif' % (data_dir))
      
      # Skip the rest if there was no inflection point
      if inflection_point['index'] == None:
        errstr = "Skipping %s" % (File_path)
        print errstr
        log_fh = open(logfile, 'a')
        log_fh.write(errstr)
        continue
      
      # we'll want this later
      max_accuracy = max(map(float, array(accuracy)[1:]))

      # run event detection using the delta value at the inflection point
      delta = inflection_point['x']
      print "Chosen delta:", delta
      peak_amp_temp, peak_sets_temp_x, peak_sets_temp_y = pdlearn.event_detection(data_smooth, delta, rate)
      
      # event summary table
      event_summary = DataFrame(index = data_orignal.columns)
      event_summary['RAIN'] = peak_amp_temp.loc['count']
      lcpro_all = Series()
      lcpro_edit = Series()
      for key, events in events_x.iteritems():
          number_events = len(events)
          if key in data_edit.columns:
              lcpro_edit[key] = number_events
          lcpro_all[key] = number_events
      event_summary['LCPro, All'] = lcpro_all
      event_summary['LCpro, select'] = lcpro_edit
      event_summary = event_summary.fillna(0)
      # event_summary contains the number of events in each ROI, according to RAIN, LCPro,
      # and according to post-processed LCPro ("LCPro, select")
      print "Total number of ROIs for each type of detection"
      print event_summary[event_summary>=1].count()
      
      chi_table = DataFrame(index = ['True', 'False'], columns= ['Positive', 'Negative'])
      
      #true positive
      temp = event_summary[event_summary['RAIN']>=1]
      true_positive = len(temp[temp['LCpro, select']>=1])
      chi_table.Positive['True'] = true_positive 
      
      #true negative
      temp = event_summary[event_summary['RAIN']<1]
      temp = temp.fillna(0)
      true_negative = len(temp[temp['LCpro, select']==0])
      chi_table.Negative['True'] = true_negative
      
      #false positive
      temp = event_summary[event_summary['RAIN']>=1]
      temp = temp.fillna(0)
      false_positive = len(temp[temp['LCpro, select']==0])
      chi_table.Positive['False'] = false_positive
      
      #false negative
      temp = event_summary[event_summary['RAIN']<1]
      false_negative = len(temp[temp['LCpro, select']>=1])
      chi_table.Negative['False'] = false_negative
      
      # "accuracy" for this delta
      delta_accuracy = (true_positive + true_negative) / \
        (true_positive + true_negative + false_positive + false_negative)
      
      print "Chi table:"
      print chi_table
      
      # colocalization plot
      print "Generating colocalization plot"
      
      pdlearn.coloc_2d(roi_loc_orignal, event_summary, im, s = 50,
        filename='plots/coloc-%s.tif' % (data_dir))
      
      ## individual line plot
      #label = 'Roi31'
      #
      #plt.plot(data_orignal.index, data_orignal[label], label = 'original', color = 'r')
      #plt.plot(data_orignal.index, data_smooth[label], label = 'smooth', color = 'b')
      #plt.plot(events_x[label], events_y[label], marker = "^", color="g", linestyle= "None")
      #plt.plot(peak_sets_temp_x[label], peak_sets_temp_y[label], marker = "^", color="y", linestyle= "None")
      #plt.title(label)
      #plt.show()
      
      # generate all line plots; saves out automatically
      print "Generating/saving line plots"
      pdutils.line_plots(data_orignal, data_smooth, events_x, events_y, \
                         peak_sets_temp_x, peak_sets_temp_y, event_summary,File_path)
      
      # save tables
      print "Saving tables"
      event_summary.to_csv(r'%s/Event_summary_table_delta-%s.csv'%(File_path, delta))
      chi_table.to_csv(r'%s/chi_table_delta-%s.csv'%(File_path, delta))
      
      # Write out some stats locally
      output = '\t'.join(map(str, [data_dir, subject_type, delta, delta_accuracy,
        max_accuracy, false_positive, false_negative]))
      stats_spec.write(output + nl)
      stats_all.write(output + nl)
