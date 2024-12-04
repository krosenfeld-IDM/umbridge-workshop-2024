import pywt
import os
import sys
import sciris as sc
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import statsmodels.api as sm

MAX_PERIOD = 7*26 # in bi-weeks

def pad_data(x):
    """
    Pad data to the next power of 2
    """
    nx = len(x) # number of samples
    nx2 = (2**np.ceil(np.log(nx)/np.log(2))).astype(int) # next power of 2
    x2 = np.zeros(nx2, dtype=x.dtype) # pad to next power of 2
    offset = (nx2-nx)//2 # offset
    x2[offset:(offset+nx)] = x # copy
    return x2

def log_transform(x, debug=1):
    """
    Log transform for case data
    """ 
    # add one and take log
    x = np.log(x+1)
    # set mean=0 and std=1
    m = np.mean(x)
    s = np.std(x)
    x = (x - m)/s
    return x

def calc_Ws(cases):
    # transform case data
    log_cases = pad_data(log_transform(cases))

    # setup and execute wavelet transform
    # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#morlet-wavelet
    wavelet = pywt.ContinuousWavelet('cmor2-1')

    dt = 1 # 2 weeks
    widths = np.logspace(np.log10(1), np.log10(MAX_PERIOD), int(MAX_PERIOD))
    [cwt, frequencies] = pywt.cwt(log_cases, widths, wavelet, dt)

    # Number of time steps in padded time series
    nt = len(cases)
    # trim matrix
    offset = (cwt.shape[1] - nt) // 2
    cwt = cwt[:, offset:offset + nt]

    return cwt, frequencies

def main(data, distances, sim_output, do_plot=False):

    # data = sc.load(os.path.join("data","londondata.sc"))
    # distances = np.load(os.path.join("data","londondist.npy"))

    # identify which locations are within 30km of London
    ref_city = "London"
    j = data.placenames.index(ref_city)
    ref_cwt, _ = calc_Ws(sim_output[:, 1, data.placenames.index(ref_city)].flatten())
    x = []; y = [];
    for i, city in enumerate(data.placenames):
        distance = distances[i,j]
        if i == j:
            continue
        cwt, frequencies = calc_Ws(sim_output[:, 1, i].flatten())
        
        diff = np.conjugate(ref_cwt)*cwt
        ind = np.where(np.logical_and(frequencies < 1/(1.5 * 52), frequencies > 1 / (3 * 52)))
        diff = diff[ind[0], :]

        # # and by time
        # ind = np.where(data.reports > 50)[0]
        # diff = diff[:, ind]

        x.append(distance)
        y.append(np.angle(np.mean(diff)))

    london_x = np.array(x); london_y = np.array(y)

    def estimate_slope(x,y):
        X = sm.add_constant(x[:, np.newaxis])
        model = sm.OLS(y, X)
        results = model.fit()
        return results.params, results.bse

    result_dict = dict()

    ind = np.isfinite(london_y)
    london_x = london_x[ind]
    london_y = london_y[ind]
    if ind.sum() > 2:
        p,pe = estimate_slope(london_x, 180/np.pi*london_y)
    else:
        p = [-np.inf, -np.inf]
        pe = [np.inf, np.inf]
    result_dict['London_m'] = (p[1],pe[1])
    result_dict['London_b'] = (p[0],pe[0])

    if do_plot:
        plt.figure(figsize=(5,4))
        gs = gridspec.GridSpec(1, 1)

        ax0 = plt.subplot(gs[0])
        sns.regplot(x=london_x, y=180/np.pi*london_y, ax=ax0)
        ax0.set_xlabel("Distance from London (km)")
        ax0.set_ylabel("Phase diff from London")
        plt.savefig("phase_diff.png")

    return p

if __name__ == "__main__":
    main()