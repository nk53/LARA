#!/usr/bin/env python

from sys import exit
from os import symlink
from os.path import join as path_join
from os.path import abspath, sep
from glob import glob
from pandas import read_csv

super_dir=abspath(path_join('..', 'data'))
treatments = ['AH', 'AN', 'FH', 'FN']
groups = ['Control', 'FPL', 'FPL & NIF']
event_summary_pattern = 'Event_summary_table*csv'
output_dir = "falses"

skipped_dirs = []
for treatment in treatments:
  for group in groups:
    # empty string forces path_join to end with separator
    # e.g. "path/to/file/", rather than "path/to/file"
    data_dirs = glob(path_join(super_dir, treatment, group, 'Analyzed', '') + '*')
    for data_dir in data_dirs:
      dd_short = data_dir.split(sep)[-1]
      event_summary_file = glob(data_dir + sep + event_summary_pattern) # an array
      if len(event_summary_file) > 1:
        exit("More than one file matched %s" % event_summary_pattern)
      if len(event_summary_file) == 0:
        skipped_dirs.append(data_dir) # log the ones we skip
        continue
      event_summary_file = event_summary_file[0]
      
      print "Reading", event_summary_file
      event_summary = read_csv(event_summary_file, index_col=0)
      
      positives = event_summary[event_summary['RAIN']>=1]
      negatives = event_summary[event_summary['RAIN']<1]
      
      fp_list = positives[positives['LCpro, select']==0]
      fn_list = negatives[negatives['LCpro, select']>=1]
      
      false_rois = fp_list.index.union(fn_list.index)
      
      for roi in false_rois:
        roi_fname = roi + '.pdf'
        link_name = '_'.join([dd_short, roi_fname])
        source = path_join(data_dir, 'plots', roi_fname)
        destination = path_join(output_dir, link_name)
        try:
          symlink(source, destination)
        except OSError as e:
          if e.errno == 17:
            errmsg = "Error: remove all symbolic links in the '%s' directory to continue" % output_dir
            exit(errmsg)

sdl = len(skipped_dirs)
if sdl > 0:
  print "Skipped the following (%d) directories:" % sdl
  for sd in skipped_dirs:
    print sd
else:
  print "All directories processed"
