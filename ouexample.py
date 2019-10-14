# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:57:48 2019

@author: Rob
"""

import numpy as np
import scipy.stats
import scipy.linalg
import scipy.integrate
import matplotlib.pyplot as plt
import os

def makedata(deltat = .1, T = 1000, samples = 100):
    n = int(T/deltat)
    result = np.zeros((samples, n))
    scale = np.exp(-deltat)
    noise = np.sqrt(1 - np.exp(-2*deltat))
    now = np.random.normal(samples)
    for i in range(n):
        if i % 1000 == 0:
            print(i)
        result[:, i] = now
        now *= scale
        now += np.random.normal(scale = noise, size = samples)
    return(result)

def makecov(data, basisSize = 20, top = 30):
    # indicator function basisSize
    quantiles = scipy.stats.norm.ppf(np.linspace(1/basisSize, 1, basisSize))
    # inner products with x
    correct = np.zeros(basisSize)
    correct[0] = scipy.integrate.quad(lambda x: x*np.exp(-.5*np.square(x))/np.sqrt(2*np.pi), -np.inf, quantiles[0])[0]
    for i in range(basisSize - 1):
        correct[i + 1] = scipy.integrate.quad(lambda x: x*np.exp(-.5*np.square(x))/np.sqrt(2*np.pi), quantiles[i], quantiles[i+1])[0]
    # go through the data
    print(correct)
    samples = data.shape[0]
    errors = np.zeros((samples, top))
    for s in range(samples):
        print(s)
        sample = data[s, :]
        rep = np.zeros((sample.size, basisSize))
        # indicator function basis
        for i in range(basisSize):
            rep[:, i] = (sample < quantiles[i])
        rep[:, 1:] = rep[:, 1:] - rep[:, :-1]
        # diagonal inner product matrix
        cov = np.matmul(np.transpose(rep), rep)/sample.size
        # numerical errors
        for i in range(1, top):
            lagcov = np.matmul(np.transpose(rep[i:, :]), rep[:-i, :]) /sample.size
            w, v = scipy.linalg.eigh(lagcov, cov)
            estimate = v[:, basisSize - 2]
            errors[s, i] = np.sqrt(1 - np.square(np.dot(estimate, correct))*basisSize/np.dot(estimate, estimate))
    return(errors)

# Experiments
deltat = .01
T = 1000
samples = 20
data = makedata(deltat, T, samples)
basisSize = 30
top = 200
errors = makecov(data, basisSize, top)

# Plotting
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 2
plot = plt.figure(constrained_layout = True)
ax = plot.gca()
ax.semilogx(deltat*np.arange(1, top), np.mean(errors, 0)[1:], color = 'black', linewidth = 2)
ax.set_ylim([.1, .2])
ax.set_xlim([.01, 1])
ax.set_xlabel('Time Lag', font)
ax.set_ylabel('Error', font)
ax.tick_params(width = 2)
plot.savefig('errorplot.png', bbox_inches = 'tight', dpi = 200)
