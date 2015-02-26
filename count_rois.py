#!/usr/bin/env python

from sys import exit
from os import symlink, stat
from os.path import join as path_join
from os.path import abspath, sep
from string import find
from glob import glob
from pandas import read_csv
from numpy import loadtxt

super_dir=abspath(path_join('..', 'data'))
treatments = ['AH', 'AN', 'FH', 'FN']
groups = {'Ctrl':'Control',
          'ctrl':'Control',
          '_FPL_':'FPL',
          'FPL-NIF':'FPL & NIF'}
dirs = loadtxt('filename_list.txt', dtype=str, delimiter='\t')

count = 0
for dirname in dirs:
  # determine path to dirname
  for treatment in treatments:
    if find(dirname, treatment) >= 0:
      break
  for group_search, group_dir in groups.iteritems():
    if find(dirname, group_search) >= 0:
      break
  path_to_dir = path_join(super_dir, treatment, group_dir, 'Analyzed',
                          dirname, 'plots', '')
  # check if path exists; dies with OSError if it fails
  stat(path_to_dir) # conveniently tells us which dir fails
  # count the number of Roi*pdf files
  rois = glob(path_to_dir + 'Roi*pdf')
  count += len(rois)

print "There are", count, "ROIs in", len(dirs), "directories"
