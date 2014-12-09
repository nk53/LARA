# This file contains general utilities, not directly related to machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os, errno
import matplotlib.patches as patches
from pandas import DataFrame
from PIL import Image
from pyeeg import * 
from numpy import NaN, Inf, arange, isscalar, asarray, array 

def load_files(folder):
    '''
    this function takes a path to where all of the LC_pro saved files are. There should be 3 files:
    'ROI normailed.text' - the ROI intensity time series data. Units should be in time (not frame) and relative intensity
    'Parameter List_edit.txt' - this is the events' information file. Duplicate ROIs are expected (since an ROI can have multipule events). The orignal LC_pro file can be loaded, as long as the name is changed to match. 
    'rbg.png' - A still of the video, must be .png. If it is a .tif, it will load, but it will be pseudo colored. it can be just a frame or some averaged measures.
    
    if the files are not named properly or the path is wrong, it will throw a file not found error.
    '''
    data = pd.read_csv(r'%s/ROI normalized.txt' %(folder), index_col= 'time(s)', sep='\t') #load the intensity time series for each roi. should be a text file named exactly 'ROI normalized.txt'
    print "Loaded 'ROI normalized.txt'"
      
    roi_param_edit = pd.read_csv(r'%s/Parameter List_edit.txt' %(folder), index_col='ROI', sep='\t')#load the parameter list.
    print "Loaded 'Parameter List_edit.txt'"

    roi_param_original = pd.read_csv(r'%s/Parameter List .txt' %(folder), index_col='ROI', sep='\t')#load the full parameter list.
    print "Loaded 'Parameter List_edit.txt'"
    
    im = Image.open(r'%s/rgb.png' %(folder)) #MUST BE RBG and .png. seriously, I'm not kidding.
    print "Loaded 'rgb.png'"
    
    del data['Unnamed: 0'] #lc_pro outputs a weird blank column named this everytime. I don't know why, but it does. this line deletes it safely.
    roi_loc_lcpro, roi_x_lcpro, roi_y_lcpro, data_edit = lcpro_param_parse(roi_param_edit, data , original=False) #use the parameter list to get the x and y location for each ROI

    roi_loc_orignal, roi_x_orignal, roi_y_orignal, data_orignal  = lcpro_param_parse(roi_param_original, data , original=True) #use the parameter list to get the x and y location for each ROI
    
    print "Configured Data"
    
    events_x, events_y = get_events(data = data_orignal, roi_param = roi_param_original) #use the parameter list to get the location and amplitude of each event for every ROI
    print "LCPro events extracted"
    
    path = folder +"/plots"
    mkdir_p(path) #makes a plots folder inside the path where the data was loaded from
    print "Made plots folder"
    
    return data_orignal, data_edit, roi_param_original, roi_param_edit, im, roi_loc_lcpro, roi_x_lcpro, roi_y_lcpro , roi_loc_orignal, roi_x_orignal, roi_y_orignal, events_x, events_y
    
def lcpro_param_parse(roi_param, data , original = True):
    '''
    This function takes the Dataframe created by opening the 'Parameter List.txt' from LC_Pro.
    It returns the location data as both a concise list datafram of only locations (roi_loc), an x and y list (roi_x, roi_y). 
    It also changes the names in the roi_loc file to be the same as they are in the data dataframe, which is 
    '''
    roi_loc = roi_param[['X', 'Y']] #make a new dataframe that contains only the x and y coordinates
    roi_loc.drop_duplicates(inplace= True) #roi_param has duplicate keys (rois) because the parameters are based on events, which lc_pro detects. a single roi can have many events. doing it in place like this does cause an error, but don't let it both you none.
    roi_x = roi_loc['X'].tolist() #extract the x column as an array and store it as a value. this is handy for later calculations
    roi_y = roi_loc['Y'].tolist() #extract the y column as an array and store it as a value. this is handy for later calculations
    new_index = [] #make an empty temp list
    for i in np.arange(len(roi_loc.index)): #for each index in roi_loc
        new_index.append('Roi'+str(roi_loc.index[i])) #make a string from the index name in the same format as the data
    roi_loc = DataFrame({'x':roi_x, 'y':roi_y}, index= new_index) #reassign roi_loc to a dataframe with the properly named index. this means that we can use the same roi name to call from either the data or location dataframes
    
    if len(data.columns) != len(new_index) and original == True: #if the number of roi's are the same AND we are using the original file (no roi's have been romved from the edited roi_param)
        sys.exit("The number of ROIs in the data file is not equal to the number of ROIs in the parameter file. That doesn't seem right, so I quit the function for you. Make sure you are loading the correct files, please.")
    
    if original == False: #if it is not the original, then use the roi_loc index to filter only edited roi's.
        data = data[roi_loc.index]
    
    truth = (data.columns == roi_loc.index).tolist() #a list of the bool for if the roi indexes are all the same.
    
    if truth.count(True) != len(data.columns): #all should be true, so check that the number of true are the same.
        sys.exit("The names on data and roi_loc are not identical. This will surely break everything later, so I shut down the program. Try loading these files again.")
    
    return roi_loc, roi_x, roi_y, data

def get_events(data, roi_param):
    '''
    extract the events from the roi_parameter list. It returns them as a pair of dictionaries (x or y data, sored as floats in a list) that use the roi name as the key. 
    duplicate events are ok and expected.
    '''
    
    new_index = [] #create a new, empty list
    
    for i in np.arange(len(roi_param.index)): #for each index in the original roi_param list, will include duplicates
        new_index.append('Roi'+str(roi_param.index[i])) #reformat name and append it to the empty index list
    roi_events = DataFrame(index= new_index) #make an empty data frame using the new_index as the index
    roi_events_time = roi_param['Time(s)'].tolist() #convert time (which is the x val) to a list
    roi_events_amp = roi_param['Amp(F/F0)'].tolist() #conver amplitude (which is the y val) to a list
    roi_events['Time'] = roi_events_time #store it in the events dataframe
    roi_events['Amp'] = roi_events_amp #store is in the events dataframe
    
    events_x = {} #empty dict
    events_y = {} #empty dict
    
    for label in data.columns: #for each roi name in data, initalize the dict by making an empty list for each roi (key) 
        events_x[label] = []
        events_y[label] = []

    for i in np.arange(len(roi_events.index)): #for each event
        key = roi_events.index[i] #get the roi name
        events_x[key].append(roi_events.iloc[i,0]) #use the name to add the event's time data point to the dict
        events_y[key].append(roi_events.iloc[i,1]) #use the name to add the event's amplitude data point to the dict
        
    return events_x, events_y #return the two dictionaries

def mkdir_p(path):
    '''
    This function creates a folder at the end of the specified path, unless the folder already exsists. 
    '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def coloc_2d(roi_loc_orignal, event_summary, im, s = 6):
    '''
    show all roi's. if detected by LCpro, in blue, if detected by RAIN, yellow. green for overlap.
    '''
    
    rain_events_list = event_summary[event_summary['RAIN'] >= 1].index.tolist() #list of only roi's found by RAIN
    lcpro_events_select_list = event_summary[event_summary['LCpro, select'] >= 1].index.tolist() #list of only roi's found by RAIN
    
    fig, ax = plt.subplots()
    
    for roi in roi_loc_orignal.index: #for each ROI from the original LCpro output
        
        col = ax.scatter(roi_loc_orignal.loc[roi].x, roi_loc_orignal.loc[roi].y,  s = s, edgecolor = 'k', linewidth ='1',color= 'w' , marker ="o", alpha = 0.5) #the col thing was an attempt to make this figure interactive with onpick, but it didn't work. this is therefore an artifacct that i am too lazy to get rid of
        
        if roi in rain_events_list: #if this object was detected by rain, plot it in yellow
            plt.scatter(roi_loc_orignal.loc[roi].x, roi_loc_orignal.loc[roi].y,  s = s, edgecolor = 'k', linewidth ='1',color= 'y' , marker ="o", alpha = 0.5)
        
        if roi in lcpro_events_select_list: #if this object was detected by LCPro, plot it in blue
            plt.scatter(roi_loc_orignal.loc[roi].x, roi_loc_orignal.loc[roi].y,  s = s, edgecolor = 'k', linewidth ='1',color= 'b' , marker ="o", alpha = 0.5)
        
        if roi in lcpro_events_select_list and roi in rain_events_list: #if this object was detected by both, overlay in magenta! ooo, magenta
            plt.scatter(roi_loc_orignal.loc[roi].x, roi_loc_orignal.loc[roi].y,  s = s, edgecolor = 'k', linewidth ='1',color= 'm' , marker ="o", alpha = 0.75)
    
    #fig.canvas.mpl_connect('pick_event', onpick)    #this was that artifact from the col = ax. line above. this is the depths of my laziness. and i'm in a rush.
    
    width, height = im.size #get the size of the image that the plot will be over
    plt.xlim(xmin=0, xmax=width) #set x limit to be the size of the image
    plt.ylim(ymin = 0, ymax = height) #set y limit to be the size of the image
    plt.title('Event detection overlap (Magenta!)') 
    plt.imshow(im)

    plt.show()
    
def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    #mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                #mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
 
    return array(maxtab)#, array(mintab)

def line_plots(data_orignal, data_smooth, events_x, events_y, peak_sets_temp_x, peak_sets_temp_y, event_summary,folder):
    '''
    creates the plots with two lines: original data and smoothed data. it also overlays the events from LCpro and RAIN.
    '''
    lcpro_events_select_list = event_summary[event_summary['LCpro, select'] >= 1].index.tolist() #list of only roi's found by RAIN
    
    for label, column in data_orignal.iteritems():
        
        plt.figure()
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity')
        plt.title(label)
        plt.ylim(ymin = min(data_orignal.min()), ymax = max(data_orignal.max()))
        plt.xlim(xmin = data_orignal.index[0], xmax = data_orignal.index[-1])
        
        plt.plot(data_orignal.index, data_orignal[label], label = 'original', color = 'r')
        plt.plot(data_orignal.index, data_smooth[label], label = 'smooth', color = 'b')
        if label in data_orignal.columns:     
            plt.plot(events_x[label], events_y[label], marker = "^", color="r", linestyle= "None")
        if label in lcpro_events_select_list:
            plt.plot(events_x[label], events_y[label], marker = "^", color="g", linestyle= "None")
        if label in peak_sets_temp_x.keys():
            plt.plot(peak_sets_temp_x[label], peak_sets_temp_y[label], marker = "^", color="y", linestyle= "None")
        plt.savefig(r'%s/plots/%s.pdf' %(folder,label))
        plt.close()
