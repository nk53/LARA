# This file contains utilities specifically related to machine learning
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdmath
from operator import itemgetter
from pandas import Series, DataFrame
from numpy import NaN, Inf, arange, isscalar, asarray, array
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from scipy.interpolate import UnivariateSpline


def delta_tuner(dataframe, epsilon, rate): #choose which data to use to tune. can be either selected list or full ist. AD reccoments full list.
    '''
    this function takes a dataframe of time series data and runs peak detection iteritvely. 
    since peak detection always has the the range of delta values of 0 to the max of stack,
    epsiolon is used to be the number of divisions of that range to test. 1 being the minimum for epsilon, which is the exact middle of the range.
    the function will return a results table (average # of events) and (# ROIs with events>1) on their own axis.
    the graph will be click able, as to obtain the delta value that generated that point.
    data results are not saved.
    '''
    
    range_array = np.linspace(0, max(dataframe.max())/2, num = epsilon) #create the array of which delta values to test. the range is from zero (although zero is not used) to half of the max value from the entire data frame. epsilon is used to determine the number of slices to make
    
    results_average = Series(index = range_array) #the empty series to store results
    results_num = Series (index = range_array) #an empty series to store results
    #results_perc = Series (index = range_array)
    
    for delta in range_array[1:]: #for each delta value in the array
        
        peak_amp_temp, peak_sets_temp_x, peak_sets_temp_y = event_detection(dataframe,delta, rate) #perform event detection with the delta

        event_counts = peak_amp_temp.loc['count'] #count the number of events, which is a row in the peak_amp_temp array
        average_num_events = event_counts.mean() #average the counts, to obtain (average # events/roi)
        num_roi = event_counts[event_counts>=1].count() #count the number of ROIs with more than one event
        
        #perc_roi = num_roi/len(data_smooth.columns)

        results_average[delta] = average_num_events 
        results_num[delta] = num_roi
        #results_perc[delta]= perc_roi
    return results_average, results_num

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def coloc_2d(roi_loc_orignal, event_summary, im, s = 6, filename=None):
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
    
    if filename == None:
      plt.show()
    else:
      plt.savefig(filename)
    
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

def event_detection(data, delta, rate):
    '''
    do peak detect on a dataframe. takes a delta value and the rate.
    A point is considered a maximum peak if it has the maximal value, and was preceded (to the left) by a value lower by DELTA.
    '''
    
    #results storage
    peak_amp_temp = DataFrame()
    rr_int_temp = DataFrame()
    peak_sets_temp_x = {}
    peak_sets_temp_y = {}

    for label, column in data.iteritems(): #for each column in the data frame
        time = column.index.tolist() #time array
        col = column.tolist() #amplitude array

        #peakdet
        maxtab = peakdet(col, delta,None)

        maxtab = np.array(maxtab)

        if maxtab.size == 0: #how to handle empty np array, which occurs if no events are detected
            maxptime = []
            maxpeaks = []

        else:
            maxptime = maxtab[:,0] #all of the rows and only the first column are time
            maxpeaks = maxtab[:,1] #all of the rows and only the second column are amp.

        maxptime_true = (np.multiply(maxptime,rate) + time[0]) #get the real time for each peak, since the peak is given in index not time
        peak_sets_temp_x[label] = maxptime_true #store array of event time in the dictionary with the ROI name as the key

        peak_sets_temp_y[label] = maxpeaks #store array of events in the dictionary with the ROI name as the key

        #RR = rrinterval(maxptime_true)
        peak_amp_temp[label] = Series(data = maxpeaks, index=maxptime_true).describe() #store summary data series in the summary dataframe
        #rr_int_temp[label] = Series(data = RR, index=maxptime_true[:-1]).describe()

    return peak_amp_temp, peak_sets_temp_x, peak_sets_temp_y

def delta_tuner2(dataframe, epsilon, rate, events_x, loaded_stats): #choose which data to use to tune. can be either selected list or full ist. AD reccoments full list.
    '''
    this function takes a dataframe of time series data and runs peak detection iteritvely. 
    since peak detection always has the the range of delta values of 0 to the max of stack,
    epsiolon is used to be the number of divisions of that range to test. 1 being the minimum for epsilon, which is the exact middle of the range.
    the function will return a results table (average # of events) and (# ROIs with events>1) on their own axis.
    the graph will be click able, as to obtain the delta value that generated that point.
    data results are not saved.
    this also generates the dataframe that contains the chi table information iteratively, which is then, later, graphable. 
    '''
    
    data_orignal, data_edit, roi_param_original, roi_param_edit, im, roi_loc_lcpro, \
      roi_x_lcpro, roi_y_lcpro , roi_loc_orignal, roi_x_orignal, roi_y_orignal, events_x, \
      events_y = loaded_stats

    range_array = np.linspace(0, max(dataframe.max())/2, num = epsilon) #create the array of which delta values to test. the range is from zero (although zero is not used) to half of the max value from the entire data frame. epsilon is used to determine the number of slices to make
    
    results_average = Series(index = range_array) #the empty series to store results
    results_num = Series (index = range_array) #an empty series to store results
    chi_table = DataFrame(index = range_array, columns=['True_Positive', 'True_Negative', 'False_Positive', 'False_Negative'])
    #results_perc = Series (index = range_array)
    
    for delta in range_array[1:]: #for each delta value in the array
        
        peak_amp_temp, peak_sets_temp_x, peak_sets_temp_y = event_detection(dataframe,delta, rate) #perform event detection with the delta

        event_counts = peak_amp_temp.loc['count'] #count the number of events, which is a row in the peak_amp_temp array
        average_num_events = event_counts.mean() #average the counts, to obtain (average # events/roi)
        num_roi = event_counts[event_counts>=1].count() #count the number of ROIs with more than one event
        
        #perc_roi = num_roi/len(data_smooth.columns)

        results_average[delta] = average_num_events 
        results_num[delta] = num_roi
        #results_perc[delta]= perc_roi
        
        #event summary table
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
        
        #true positive
        temp = event_summary[event_summary['RAIN']>=1]
        true_positive = len(temp[temp['LCpro, select']>=1])
        chi_table.True_Positive[delta] = true_positive 

        #true negative
        temp = event_summary[event_summary['RAIN']<1]
        temp = temp.fillna(0)
        true_neg = len(temp[temp['LCpro, select']==0])
        chi_table.True_Negative[delta] = true_neg

        #false positive
        temp = event_summary[event_summary['RAIN']>=1]
        temp = temp.fillna(0)
        false_positive = len(temp[temp['LCpro, select']==0])
        chi_table.False_Positive[delta] = false_positive

        #false negative
        temp = event_summary[event_summary['RAIN']<1]
        false_negative = len(temp[temp['LCpro, select']>=1])
        chi_table.False_Negative[delta] = false_negative
        
    return results_average, results_num, chi_table

def cubicRegression(training_set, target):
    '''
    expects training_set and target each to be 1-dimensional
    NaNs are discarded
    '''
    
    ## clean data of NaNs
    #data = vstack([training_set, target]).T
    #data = data[~isnan(data).any(axis=1)]
    #fixed_set, fixed_target = data[:,0], data[:,1]
    
    # define cubic regression model
    model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                      ('linear', LinearRegression(fit_intercept=False))])
    
    # do the regression
    model = model.fit(fixed_set[:, newaxis], fixed_target)
    coefficients = model.named_steps['linear'].coef_
    
    # output y values for curve
    #curve = 0
    #for index, c in enumerate(coefficients):
    #    curve += c * training_set ** index
    curve = polyn_value(coefficients, training_set)
    
    # find the inflection point
    d2 = derivative(derivative(coefficients))
    inflection_point_x = -1 * d2[0] / d2[1]
    inflection_point_y = polyn_value(coefficients, inflection_point_x)
    
    return curve, inflection_point_x, inflection_point_y

def splineSmooth(x, y):
    '''
    Returns a smoothed line from the (x,y) data given via UnivariateSpline
    with an automatically-determined smoothing factor
    '''
    ## clean NaNs from (x,y)
    #data = vstack([x, y]).T
    #data = data[~isnan(data).any(axis=1)]
    #fx, fy = data[:,0], data[:,1]
    x = x[1:] # the NaNs are consistently in
    y = y[1:] # the 0th index position

    # perform smoothing
    s = UnivariateSpline(x, y, s=len(x)*2)
    xs = np.linspace(x[0], x[-1], len(x)*10)
    ys = s(xs)
    
    # get 2nd derivative
    derivative2 = s.derivative(n=2)
    yd = derivative2(xs)
    coeffs = derivative2.get_coeffs()
    knots  = derivative2.get_knots()
    
    # next part won't work if arrays aren't of equal size
    if len(coeffs) != len(knots):
      print "Lengths do not match"; quit()
    
    # Get the lines describing the 2nd derivative and the points where y=0
    lines = []
    zeros = []
    for i in range(len(coeffs) - 1):
      x1, x2 = knots[i], knots[i+1]
      y1, y2 = coeffs[i], coeffs[i+1]
      line = pdmath.linef2pts([x1, y1], [x2, y2], solve="x")
      line_zero = pdmath.linzero(line, x1, x2)
      lines.append(line)
      if line_zero != None:
        zeros.append(line_zero)
    
    # We want the inflection point that's closest to the median
    zeros = array(zeros)
    med = np.median(x)
    diffs = abs(zeros - med)
    ipoint_index = min(enumerate(diffs), key=itemgetter(1))[0]
    ipoint = zeros[ipoint_index]
    
    smoothed_spline  = {'xs' : xs, 'ys' : ys}
    second_derivative   = {'xs' : xs, 'ys' : yd}
    inflection_point = {'index' : ipoint_index,
                            'x' : ipoint,
                            'y' : yd[ipoint_index]}

    return smoothed_spline, second_derivative, inflection_point
