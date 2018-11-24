#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:32:25 2017

@author: nawalgao
"""


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk", font_scale=1.4)
  

def plot_hmc_chains(m, samples):
    """
    Inputs:
    m: gpflow model object
    samples: gpflow model object
    """
    lengthscales =m.kern.lengthscales.get_samples_df(samples)
    variance = m.kern.variance.get_samples_df(samples)
    plt.plot(lengthscales)
    plt.title('lengthscales')
    #overall_variance = m.pref_var.get_samples_df(samples)
    plt.figure()
    plt.plot(variance)
    plt.title('signal variance')
    return

