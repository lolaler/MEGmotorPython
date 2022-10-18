#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:55:56 2022

@author: lolaler

This script contains functions to help with csv loop file calculations.
Must be in the same directory as the sub_sessions_loop.py file in order to work as intended.

"""

import mne, numpy as np
from math import sqrt
import os.path, csv
import matplotlib.pyplot as plt

from mne.channels.layout import _pair_grad_sensors, _merge_ch_data
from mne.io import read_raw_fif
from mne.preprocessing import compute_proj_eog




### functions to get the lowest point ###

def _get_peak(data, times, tmin=None, tmax=None, mode='abs'):
    """
    Get feature-index and time of maximum signal from 2D array.
    Note. This is a 'getter', not a 'finder'. For non-evoked type
    data and continuous signals, please use proper peak detection algorithms.

    Parameters
    ----------
    data : instance of numpy.ndarray (n_locations, n_times)
        The data, either evoked in sensor or source space.
    times : instance of numpy.ndarray (n_times)
        The times in seconds.
    tmin : float | None
        The minimum point in time to be considered for peak getting.
    tmax : float | None
        The maximum point in time to be considered for peak getting.
    mode : {'pos', 'neg', 'abs'}
        How to deal with the sign of the data. If 'pos' only positive
        values will be considered. If 'neg' only negative values will
        be considered. If 'abs' absolute values will be considered.
        Defaults to 'abs'.

    Returns
    -------
    max_loc : int
        The index of the feature with the maximum value.
    max_time : int
        The time point of the maximum response, index.
    max_amp : float
        Amplitude of the maximum response.

    """

    if tmin is None:
        tmin = times[0]
    if tmax is None:
        tmax = times[-1]

    if tmin < times.min():
        raise ValueError('The tmin value is out of bounds. It must be '
                          'within {} and {}'.format(times.min(), times.max()))
    if tmax > times.max():
        raise ValueError('The tmax value is out of bounds. It must be '
                          'within {} and {}'.format(times.min(), times.max()))
    if tmin > tmax:
        raise ValueError('The tmin must be smaller or equal to tmax')

    time_win = (times >= tmin) & (times <= tmax)
    mask = np.ones_like(data).astype(bool)
    mask[:, time_win] = False

    maxfun = np.argmin
    if mode == 'pos':
        if not np.any(data > 0):
            raise ValueError('No positive values encountered. Cannot '
                              'operate in pos mode.')
    elif mode == 'neg':
        if not np.any(data < 0):
            raise ValueError('No negative values encountered. Cannot '
                              'operate in neg mode.')
        maxfun = np.argmin

    masked_index = np.ma.array(np.abs(data) if mode == 'abs' else data,
                                mask=mask)

    max_loc, max_time = np.unravel_index(maxfun(masked_index), data.shape)

    return max_loc, max_time, data[max_loc, max_time]

def get_min(self, ch_type=None, tmin=None, tmax=None,
                  mode='abs', time_as_index=False, merge_grads=False,
                  return_amplitude=False):
        """

        Get location and latency of peak amplitude.

        Parameters
        ----------
        ch_type : str | None
            The channel type to use. Defaults to None. If more than one sensor
            Type is present in the data the channel type has to be explicitly
            set.
        tmin : float | None
            The minimum point in time to be considered for peak getting.
            If None (default), the beginning of the data is used.
        tmax : float | None
            The maximum point in time to be considered for peak getting.
            If None (default), the end of the data is used.
        mode : {'pos', 'neg', 'abs'}
            How to deal with the sign of the data. If 'pos' only positive
            values will be considered. If 'neg' only negative values will
            be considered. If 'abs' absolute values will be considered.
            Defaults to 'abs'.
        time_as_index : bool
            Whether to return the time index instead of the latency in seconds.
        merge_grads : bool
            If True, compute peak from merged gradiometer data.
        return_amplitude : bool
            If True, return also the amplitude at the maximum response.
            .. versionadded:: 0.16

        Returns
        -------
        ch_name : str
            The channel exhibiting the maximum response.
        latency : float | int
            The time point of the maximum response, either latency in seconds
            or index.
        amplitude : float
            The amplitude of the maximum response. Only returned if
            return_amplitude is True.
            .. versionadded:: 0.16

        """  # noqa: E501

        types_used = self.get_channel_types(unique=True, only_data_chs=True)


        if ch_type is not None and ch_type not in types_used:
            raise ValueError('Channel type `{ch_type}` not found in this '
                              'evoked object.'.format(ch_type=ch_type))

        elif len(types_used) > 1 and ch_type is None:
            raise RuntimeError('More than one sensor type found. `ch_type` '
                                'must not be `None`, pass a sensor type '
                                'value instead')

        if merge_grads:
            if ch_type != 'grad':
                raise ValueError('Channel type must be grad for merge_grads')
            elif mode == 'neg':
                raise ValueError('Ei ole negatiivisia arvoja')

        meg = eeg = misc = seeg = dbs = ecog = fnirs = False
        picks = None
        if ch_type in ('mag', 'grad'):
            meg = ch_type
        elif ch_type == 'eeg':
            eeg = True
        elif ch_type == 'misc':
            misc = True
        elif ch_type == 'seeg':
            seeg = True
        elif ch_type == 'dbs':
            dbs = True
        elif ch_type == 'ecog':
            ecog = True

        if ch_type is not None:
            if merge_grads:
                picks = _pair_grad_sensors(self.info, topomap_coords=False)
            else:
                picks = mne.pick_types(self.info, meg=meg, eeg=eeg, misc=misc,
                                    seeg=seeg, ecog=ecog, ref_meg=False,
                                    fnirs=fnirs, dbs=dbs)
        data = self.data
        ch_names = self.ch_names

        if picks is not None:
            data = data[picks]
            ch_names = [ch_names[k] for k in picks]

        if merge_grads:
            data, _ = _merge_ch_data(data, ch_type, [])
            ch_names = [ch_name[:-1] + 'X' for ch_name in ch_names[::2]]

        ch_idx, time_idx, max_amp = _get_peak(data, self.times, tmin,
                                              tmax, mode)

        out = (ch_names[ch_idx], time_idx if time_as_index else
                self.times[time_idx])

        if return_amplitude:
            out += (max_amp,)

        return out
    

def filterChannels(side):
    """ 

    Get channels based on which hand's finger was moved. These channels have been manually selected, 
    so they can be changed freely.
        
    Parameters
    ----------
    side: str
        Either 'left' or 'right'.

    Returns
    -------
    channels: list(str)
        An array containing the channels.

    """
    
    # left hand finger moving 
    if side == 'left':
        channels =   ['MEG1312', 'MEG1313', 'MEG1322', 'MEG1323', 'MEG1443', 'MEG1442',
                      'MEG1143', 'MEG1142', 'MEG1132', 'MEG1133', 'MEG1343', 'MEG1342', 'MEG1332', 'MEG1333', 
                      'MEG2222', 'MEG2223', 'MEG2413', 'MEG2412', 'MEG2422', 'MEG2423',
                      'MEG2443', 'MEG2442', 'MEG2432', 'MEG2433']
    
    #right hand finger moving
    else:
        channels = ['MEG0423', 'MEG0422', 'MEG0412', 'MEG0413', 'MEG0223', 'MEG0222', 'MEG0212', 'MEG0213', 'MEG0433', 'MEG0432', 'MEG0442', 
                    'MEG0443', 'MEG0233', 'MEG0232', 'MEG0242', 'MEG0243', 'MEG1823', 'MEG1822', 'MEG1812', 'MEG1813',
                    'MEG1623', 'MEG1622', 'MEG1612', 'MEG1613', 'MEG1633', 'MEG1632', 'MEG1642', 'MEG1643', 'MEG1842', 'MEG1843']

    return channels



def peak_vectorsum(evokedArray, channel):
    """

    A function to get the vector sum of gradiometer pairs. This is done using the method get_peak from the mne.Evoked class. 
    The mode is set to 'pos', but in some cases there are only negative values, which is the reason for the try-except structure.

    Parameters
    ----------
    evokedArray: EvokedArray (Evoked object from numpy array)
        An array of evoked data. Has attributes such as ch_names, data, tmax, tmin, info and so on.

    Returns
    -------
    channel: str
        The channel pair with the highest vector sum. 

    maxtime: int
        The time at which the maximum amplitude is reached.

    maxamplitude: int
        The maximum amplitude.

    """
    other = channel[:-1]
    if   (channel[-1] == '2'):  other += '3'
    elif (channel[-1] == '3'):  other += '2'

    # ---------- REMEMBER TO CHANGE TIME INTERVAL IN WHICH THE MAX VALUE IS ALLOWED TO BE -----------
    try:
        copy = evokedArray.copy().pick_channels([channel, other])
        channel, maxtime, maxamplitude = copy.get_peak(  ch_type = 'grad', tmin = 0.5, tmax = 2.3, mode = 'pos', 
                                                                time_as_index = False, merge_grads = True, return_amplitude = True)
    except ValueError:
        channel = "No positive values"
        maxtime = 0
        maxamplitude = 0

    return channel, maxtime, maxamplitude






def maxdiff(evokedArray):
    """

    A function to find the channel with the biggest difference between its maximum and minimum amplitudes.
    
    Parameters
    ----------
    evokedArray: EvokedArray (Evoked object from numpy array)
        An array of evoked data. Has attributes such as ch_names, data, tmax, tmin, info and so on.

    Returns
    -------
    max_diff: int
        The difference between the maximum and minimum amplitudes.

    max_channel: st
        The channel with the biggest difference.
    
    """

    max_diff = 0
    max_channel = 'null'

    # we loop through the channels one at a time to find their maximum and minimum amplitudes
    for kanava in evokedArray.ch_names:
        forList = list()
        forList.append(kanava)

        # we have to create a copy of the array because the pick_channel function is permanent. 
        # we then use the pick_channel method to only find the peak i.e. maximum of the current channel
        copy = evokedArray.copy()           

        try: 
            kanavamax, aikamax, amplitudimax = copy.pick_channels(forList, ordered = False).get_peak(ch_type = 'grad', tmin = 0, tmax = None, 
                                                              mode = 'pos', time_as_index = False, merge_grads = False, return_amplitude = True)
        
        # in case there is a channel with only negative values
        except ValueError:
            kanavamax = None
            aikamax = 0
            amplitudimax = 0

        # we then use the pick_channel method to only find the minimum of the current channel
        kanavamin, aikamin, amplitudimin = get_min((copy.pick_channels(forList, ordered = False)), ch_type='grad', tmin = 0, tmax = None, 
                                                    mode = 'neg', time_as_index = False, merge_grads = False, return_amplitude = True)

        # check if the difference is bigger than the previous max difference. if yes => update the value and the max channel
        if (kanavamax != None and amplitudimax - amplitudimin > max_diff):
            max_diff = amplitudimax - amplitudimin
            max_channel = kanava

    # plot if needed
    # mne.viz.plot_evoked_topo([copy], merge_grads = False, title = 'Suurin ero max ja min välillä: ' + str(max_diff) + ' Kanava: ' + max_channel)
    return max_diff, max_channel





def singlemaxmin(evokedArray, subjectNo, sessionNo, side):
    """ 
    
    A function to find the channel with the maximum amplitude as well as the channel with the minimum amplitude.
    These are meant to be two different channels.

    Parameters
    ----------
    evokedArray: EvokedArray
        See above for thorough description.

    subjectNo: str
        The subject number as a string (easier to use when setting the path for each patient). The format is 'sub-0x' or 'sub-xx'.
        Used in the title for the plot.

    sessionNo: int
        The session number.
        Used in the title for the plot.

    side: str
        The side, either 'left' or 'right'.
        Used in the title for the plot. 

    Returns
    -------

    channelmax: str
        The name of the channel with the highest amplitude.
    
    maxtime: int
        The time at which the max amplitude was reached.

    maxamplitude: int
        The maximum amplitude.

    channelmin: str
        The name of the channel with the lowest amplitude.
    
    mintime: int
        The time at which the min amplitude was reached.

    minamplitude: int
        The minimum amplitude.

    """

    # finding the channel with highest max amplitude
    channelmax, maxtime, maxamplitude = evokedArray.get_peak(ch_type = 'grad', tmin = 0.5, tmax = None, mode = 'pos', 
                                                             time_as_index = False, merge_grads = False, return_amplitude = True)

    # finding the channel with the lowest min amplitude, using a copy of the array just in case
    evokedcopy = evokedArray.copy()
    channelmin, mintime, minamplitude = get_min((evokedcopy.pick_channels(evokedcopy.ch_names, ordered = False)), ch_type='grad', 
                                                tmin = 0, tmax = 1, mode = 'neg', time_as_index = False, merge_grads = False, return_amplitude = True)
    
    # visualization
    # mne.viz.plot_evoked_topo([evokedArray], merge_grads = False, title = 'Subject: ' + subjectNo + ' Session: ' + str(sessionNo) + ' Side: ' + side 
    #                                                                                                        + ' Max: ' + channelmax + ' Min: ' + channelmin)
    
    return channelmax, maxtime, maxamplitude, channelmin, mintime, minamplitude
    




def correctAndPlot(evokedArray, tmin, tmax, plot):
    """

    A function that applies the mne baseline_correction function to the array of evoked data and plots the results.
    The baseline correction is based on the mean from the interval (tmin, tmax).

    Parameters
    ----------
    evokedArray: EvokedArray
        See above for thorough description.

    tmin: int
        The lower limit of the interval used for calculating the mean.

    tmax: int
        The upper limit of the interval used for calculating the mean.

    plot: Boolean
        True if the user wants a plot of the results. If false, no plot.

    Returns
    -------

    (a plot)

    evokedArray: EvokedArray
        The baseline corrected array.

    """
    
    # baseline correction
    evokedArray = evokedArray.apply_baseline(baseline = (tmin, tmax), verbose = None)

    if (plot):
        mne.viz.plot_evoked_topo([evokedArray], merge_grads = False)
    
    return evokedArray


def vectorSumTime(evokedArray, vectorpair, refchannel):
    """

    A function that finds the vector sum of the chosen vector pair.

    The function first gets the time and minimum amplitude of the reference channel, and uses that time = t to find the amplitude
    of the other gradiometer channel at the same time = t.

    The purpose is to find the value of the vector sum of the (negative valued) suppression phase.

    Parameters
    ----------
    evokedArray: EvokedArray
        See above for thorough description.

    vectorpair: str
        The gradiometer pair for which we want to calculate the vector sum. E.g. 'MEG144X'

    refchannel: str
        One of the channels in the gradiometer pair we are looking at. Could be either the channel that ends in 2 or 3, so either 'MEG1443' or 'MEG1442'

    Returns
    -------
    vectorsum(refamplitude, othamplitude): float
        The Root Mean Square (RMS) value of the two amplitudes. 

    reftime: float
        The time at which we checked the two channels' amplitudes. Expressed in seconds.


    """

    # create copy of evoked
    refcopy = evokedArray.copy()

    # we pick only our reference channel(s)
    refcopy.pick_channels([refchannel])

    other = vectorpair[:-1]

    # if our reference channel is eg. 'MEG1343', its pair is 'MEG1342'. this if lets us get the name of the pair
    if (refchannel[-1] == '2'): other += '3'
    else:                       other += '2'

    # create copy of evoked
    othcopy = evokedArray.copy()

    # ...and pick the other channel
    othcopy.pick_channels([other])

    # get the time at which the chosen reference channel reaches its minimum
    channel, reftime, refamplitude = get_min(refcopy, ch_type='grad', tmin = 0, tmax = 0.5, mode = 'neg', time_as_index = False, 
                                                                                merge_grads = False, return_amplitude = True)
    
    # we now have the index for the value that we want: time*1000 to get an index (e.g. 0.435s --> index 435). + 500 because time starts at -0.5 s
    index = int(reftime * 1000) + 500

    # using the index that we got from reftime, we get the amplitude of the other channel at that time
    othamplitude = othcopy.data[0][index]

    # return the amplitude 
    return vectorsum(refamplitude, othamplitude), reftime     


def vectorsum(first, second):
    """ 

    Pretty self-explanatory. Returns the vector sum (Root Mean Square, RMS) of the two numbers given as parameters.
    
    """

    return sqrt((1/2) * (pow(first, 2) + pow(second, 2)))


def readminchannel(filename):
    """
    
    A function that reads the visually selected gradiometer pairs and their minimum reference channels from a spreadsheet (in .csv-format). 
    Returns a dictionary (key -> value) with the relevant info.

    Parameters
    ----------
    filename: str
        The name of the .csv file from which the channels are to be read.

    Returns
    -------
    dictionary: Dictionary
        A dictionary with the following info in keys and values: (subject, session, side) -> (channelpair, reference channel)

    """

    # open the file to be read
    with open(filename, 'r') as file:
        csvreader = csv.reader(file)

        # initialize an empty dictionary
        dictionary = {}

        # skip the first row because it's just headers
        next(csvreader)

        # loop through the rest of the file and get each row's subject, session, side, chosen channel pair and reference channel
        for row in csvreader:
            sub = row[0]
            ses = row[1]
            side = row[2]
            channelpair = row[5]
            refchannel = row[6]

            # add said values to dictionary
            dictionary[(sub, ses, side)] = (channelpair, refchannel)
    
    return dictionary

def readchanneltime(filename):    
    """
    
    A function that reads the visually selected gradiometer pairs and the upper limit for a time interval used to calculate the average
    value of the suppression phase. Read from a spreadsheet (in .csv-format). 
    Returns a dictionary (key -> value) with the relevant info.

    Parameters
    ----------
    filename: str
        The name of the .csv file from which the channels are to be read.

    Returns
    -------

    dictionary: Dictionary
        A dictionary with the following info in keys and values: (subject, session, side) -> (channelpair, tmax)

    """

    # open the file to be read
    with open(filename, 'r') as file:
        csvreader = csv.reader(file)

        # initialize an empty dictionary
        dictionary = {}
        next(csvreader)

        # loop through the rest of the file and get each row's subject, session, side, chosen channel pair and tmax
        for row in csvreader:
            sub = row[0]
            ses = row[1]
            side = row[2]
            channelpair = row[4]
            tmax = row[5]

            # add said values to dictionary
            dictionary[(sub, ses, side)] = (channelpair, tmax)
    
    return dictionary


def avg(some_list):
    """ 

    Returns the average of all values in some list.

    """
    return (sum(some_list)/ len(some_list))


def timebasedAvg(evoked, channel2, tmin, tmax):
    """

    A function that calculates the average amplitude of a gradiometer pair within the chosen timeinterval tmin <= t <= tmax.

    Parameters
    ----------
    evokedArray: EvokedArray
        See above (line 248) for thorough description.

    channel2: str
        The channel of the gradiometer pair that ends in the number 2. So 'MEG1442', 'MEG0212', and so on.

    tmin: float
        The beginning of the chosen interval.

    tmax: float
        The end of the chosen interval.

    Returns
    -------
    vectorsum(avg2, avg3): float
        Calls the vectorsum-function and returns the vector sum / RMS of the two averages.

    """

    # create copies of evokedArray
    # here copy2 means the channel that ends in 2, copy3 the channel that ends in 3. 
    copy2 = evoked.copy()
    copy3 = evoked.copy()
    channel3 = channel2[:-1] + '3'

    # filter the channels so we're only left with the channels of the current gradiometer pair
    copy2.pick_channels([channel2])
    copy3.pick_channels([channel3])

    # times values multiplied by 1000 to get the indexes
    tmin *= 1000
    tmax *= 1000

    # indexes + 1000 because the epoch starts at -1 second
    sliceobj = slice(1000 + int(tmin), 1000 + int(tmax))
    
    # get the value lists for our channels
    datalist2 = copy2.data[0]
    datalist3 = copy3.data[0]
    
    # slice the lists so we're only left with values whose indexes are within our time interval, and find the avg values
    avg2 = avg(datalist2[sliceobj])
    avg3 = avg(datalist3[sliceobj])

    # lastly, return the vector sum of the two averages
    return vectorsum(avg2, avg3)

def findhalfmax(evoked, maxamp, refchannel, peaktime):
    """ 
    
    A function to find the times when 50 % of the max amplitude of a channel (rebound) has been reached. The aim is to find two
    time points, one on each "side" of the highest peak.

    Parameters
    ----------
    evokedArray: EvokedArray
        See above (line 248) for thorough description.

    maxamp: float
        The maximum amplitude of the refchannel.

    refchannel: str
        The channel that we are investigating. The max amp comes from this channel, 
        and we want to find tmin and tmax where 0.5*maxamp has been reached.

    peaktime: float
        The time (e.g. 0.435 s) when maxamp was reached.

    Returns
    -------
    halfmax: float
        The value of half of the max amplitude, i.e. 0.5*maxamp.

    (tmin - 1000) / 1000: float
        When going through the loop, the times are used ase indexes, so 0.435 s --> 435. in order to remove the first second
        before the stimulus and to return the time back into seconds we subtract the time before the stimulus (1 s = 1000)
        and divide it by 1000 to get it into milliseconds.

    (tmax - 1000) / 1000: float
        Same procedure as with tmin.
    
    """

    # copy the data from the evoked Array
    copy = evoked.copy()

    # filter the array so we're only left with the reference channel
    copy.pick_channels([refchannel])
    
    # create variables where we store the results later on
    tmin = 0
    tmax = 0

    # because we are looking at discrete time points, we have to set a confidence interval.
    ci = 0.25 * pow(10, -13)

    # set the smallest difference between data point and ci to be really big => 100.
    mindifference = 100

    # set the value for 50 % of the max amplitude
    halfmax = 0.5 * maxamp

    # we start looking for the tmin ca 0.3 s before the peak
    i = int(peaktime * 1000 - 300)

    # loop through the data from the channel. as long as the answer isn't equal to half of 
    # the max amp, we continue (unless we reach the end)
    while  i < len(copy.data[0]):

        # amplitude at index i
        ampl = copy.data[0][i]

        # set interval for where the value should be
        plus = ampl + ci
        minus = ampl - ci

        # if the amplitude at index i is within our confidence interval and the difference is smaller than the smallest so far, update t-values
        if  ((halfmax - ci <= ampl <= halfmax + ci) and (plus < mindifference or abs(minus) < mindifference)):

            # if we're looking at times before the peak, update tmin
            if (i < peaktime):
                tmin = i

            # else we update tmax
            else:
                tmax = i

        # update i to go to the next index
        i += 1

    print('Results: tmin - ' + str(tmin) + ' tmax - ' + str(tmax))

    return halfmax, (tmin - 1000) / 1000, (tmax - 1000) / 1000


def getEvokeds(filepath, tmin, tmax, stimchannel):
    """
    
    A function that reads a raw .fif file and outputs an Evoked Array.

    Parameters
    ----------

    filepath: str
        The path to the raw .fif file.

    tmin: float
        The tmin for the epoch.

    tmax: float
        The tmax for the epoch.

    stimchannel: str
        The stim channel, e.g. 'STI001'.

    Returns
    -------

    evoked: Evoked Array
        An Evoked Array with attributes such as data, info etc.

    """

    # let's explore some frequency bands
    iter_freqs = [('Beta', 13, 25),]

    # we create average power time courses for each frequency band and set epoching parameters
    baseline = None
    
    # get the header to extract events
    raw = read_raw_fif(filepath,preload=False)   

    # add stim_channel
    events = mne.find_events(raw, stim_channel=stimchannel)       

    frequency_map = list()

    for band, fmin, fmax in iter_freqs:

        # (re)load the data to save memory
        raw = mne.io.read_raw_fif(filepath, preload = True)

        # we're only looking at gradiometers
        raw.pick_types(meg = 'grad', eeg = False, eog = True)

        # remove eye blinks with PCA
        projs,eog_events = compute_proj_eog(raw, tmin = -0.2, tmax = 0.2, n_mag = 2, n_grad = 2, n_eeg = 2,
                                            average = True, eog_h_freq = 20, eog_l_freq = 1, ch_name = None, n_jobs = 1)
        raw.add_proj(projs, remove_existing=True)

        # bandpass filter and compute Hilbert
        raw.filter(fmin, fmax, n_jobs=  4,  # use more jobs to speed up.
                    l_trans_bandwidth = 1,  # make sure filter params are the same
                    h_trans_bandwidth = 1,  # in each band and skip "auto" option.
                    fir_design = 'firwin')
        raw.apply_hilbert(n_jobs=4, envelope=False)
        epochs=mne.Epochs(raw, events, event_id=5, tmin=tmin, tmax=tmax, proj=True, baseline=baseline, preload=True, reject=dict(grad=4000e-13))

        # remove evoked response and get analytic signal (envelope)
        epochs.subtract_evoked()
        epochs = mne.EpochsArray(data=np.abs(epochs.get_data()), info=epochs.info, tmin=epochs.tmin) 

        # now average and move on
        frequency_map.append(((band, fmin, fmax), epochs.average()))
        evoked=epochs.average() 

    return evoked

def getRawPath(sub, ses, task):

    """
    
    A function that returns the paths to two files based on the subject's number, session number and task.

    Parameters
    ----------

    sub: int
        The subject number as an integer.

    ses: int
        The session number as an integer.

    task: str
        The task, either 'evoked' or 'motor'.

    Returns
    -------

    (left side path, right ride path): str tuple
        A tuple of the file paths for both sides, left side first.

    """


    # check whether we need to add a zero before the subject number
    if sub >= 10:
        path = 'somepath' + 'sub-' + str(sub)
        subject = 'sub-' + str(sub)
    else:
        path = 'somepath' + 'sub-0' + str(sub)
        subject = 'sub-0' + str(sub)

    # formatting the session to be 'ses-0j'
    session = 'ses-0' + str(ses)

    # get end of filename based on the task (standardies end of filename)
    if (task == 'evoked'):
        end = 'CKCevoked_proc-raw_meg_tsss_mc.fif'
    else:
        if (sub == 9):
            end = 'motor_ann_proc-raw_meg_tsss_mc.fif'
        else:
            end = 'motor_proc-raw_meg_tsss_mc.fif'

    # set the base of the filename 
    root =  "some base filename"

    # combine and return filenames for both sides
    return (subject, session, (root + 'left' + end, root + 'right' + end))


def main_function(subject, session, type, plot, read_channels, readf, writef, c_names):

    # create range object from either one or many subjects
    if isinstance(subject, int): 
        subj_range = range(subject, subject + 1) 
    else:  
        subj_range = range(subject[0], subject[1])

    # create range object from either one or many sessions
    if isinstance(session, int): 
        ses_range = range(session, session + 1) 
    else:  
        ses_range = range(session[0], session[1])

    if read_channels:
        sessionDict = readchanneltime(readf)

    tmin = -1
    tmax = 6
    basecorlim = 0.1

    if (type == 'evoked'):
        tmin = -0.5
        tmax = 2.5
    
    # open the file we are going to write in. the file closes automatically with this 'with x as y' structure
    with open(writef, "w") as file:

        # create a csv writer object
        writefile = csv.writer(file, dialect = 'excel') 
    
        # write column names into the .csv file
        writefile.writerow(c_names)

        for i in subj_range:

            for j in ses_range:

                # get raw pathnames for both left and right side
                subject, session, raw_paths = getRawPath(i, j, type)

                # go thru both sides
                for raw_path in raw_paths:

                    if 'left' in raw_path:
                        side = 'left'
                    else:
                        side = 'right'

                    # check whether the file exists, i.e. whether the session for this subject has even happened or 
                    # the file has been named correctly (exclude subject 6)
                    if os.path.isfile(raw_path) and not (i == 6):

                        # initiate empty list and add the subject number, the session number and which side we're looking at
                        result_list = list()
                        result_list.extend([subject, str(j), side])

                        # get evoked data
                        evoked = getEvokeds(raw_path, tmin, tmax, 'STI001')

                        # baseline correct and plot if True
                        correctAndPlot(evoked, -1, -0.1, plot)

                        if read_channels:
                            # get the chosen channel based on the subject, session and side
                            chosen_channel, nothing = sessionDict[(subject, str(j), side)]

                            # if we have chosen a channel for this session and side...
                            if (chosen_channel):
                                MEG_channel = 'MEG' + chosen_channel

                                # get evoked data
                                evoked = getEvokeds(raw_path, -1, 6, 'STI001')

                                # baseline correct and plot if True
                                correctAndPlot(evoked, tmin, basecorlim, plot)
                                
                                try:
                                    # ... we filter the evoked data so we're left with only the data from our chosen channel
                                    evoked.pick_channels([MEG_channel])
                                
                                except ValueError:
                                    print("You might want to check your csv-file, the channels may have been interpreted as numbers (causing missing zeros).\n Adding a zero this time automatically. :)\n")
                                    MEG_channel = 'MEG0' + chosen_channel
                                    evoked.pick_channels([MEG_channel])


                                # downsample to make the data fit into a spreadsheet
                                data = mne.filter.resample(evoked.data[0], down = 10)

                                # add the data to  list
                                result_list.extend(data)

                            else:
                                result_list.extend(["No data for this session."])

                            # write entire list into the spreadsheet (.csv file)
                        writefile.writerow(result_list)

                    else:
                        # if the session could not be found, print it in the console and the .csv file. 
                        print("Could not find " + str(session) + " for subject " + subject + ".\n")
                        writefile.writerow(["Session " + str(j) + " for subject " + subject + " not found."])

    # ---- by this point, the file we wrote into has been closed. ----

    # let user know that we are done and which file we wrote into
    print("All files have been processed and all info has been written to file " + writef + ".")
