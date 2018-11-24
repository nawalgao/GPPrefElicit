#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 10:43:39 2018

@author: nimishawalgaonkar
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk", font_scale=1.4)


from ..SynOccupant import datagen
from ..GPUnimodalPref.predict import Predict

def visualize_utility(Xgrid, Mgrid, Vargrid):
    """
    Visualize 1D utility funciton values
    Xgrid : grid states
    Mgrid : mean of GP at those finite grid points
    Vargrid : variance of GP at those finite grid points
    """
    Stdgrid = np.sqrt(Vargrid)
    lower = Mgrid - 2*Stdgrid
    upper = Mgrid + 2*Stdgrid
    line, = plt.plot(Xgrid, Mgrid, lw = 2)
   
    plt.fill_between(Xgrid[:,0], lower, upper,
                     color = line.get_color(), alpha = 0.25)
    plt.xlabel('Temperature degC')
    plt.ylabel('Utility')
    plt.title('Utility at different temp values')
    return


def diff_ut(iter_num, trial_num, 
            m, samples, Xgridnorm, Xgrid,
            savefig = True, num_gps = 100):
    """
    Different utilities along with the associated uncertainities
    """
    PREDICT = Predict(m)
    meangrid, vargrid = PREDICT.u_mcmc(samples, Xgridnorm)
    plt.figure(figsize=(12,8))
    np.savez('../GPFlowPref/data/results/T' + str(trial_num) +
                    '/utility_samples/Ih' + str(iter_num) + 'D' + '.npz',
                    meangrid = meangrid, vargrid = vargrid)
    for i in xrange(10,10 + num_gps):
        visualize_utility(Xgrid,meangrid[i,:],vargrid[i,:])
    if savefig:
        plt.savefig('../GPFlowPref/data/results/T' + str(trial_num) +
                    '/utility_samples/Ih' + str(iter_num) + 'D' + str(i) + '.png', dpi = 600)
        plt.close()

def diff_g(iter_num, trial_num, 
            m, samples, Xgridnorm, Xgrid,
            savefig = True, num_gps = 100):
    """
    Different utilities along with the associated uncertainities
    """
    
    plt.figure(figsize=(12,8))
    for i in xrange(10,10 + num_gps):
        m.set_state(samples[i,:])
        g = m.predict_g_samples(Xgridnorm, 1)
        plt.plot(Xgridnorm, norm.cdf(g[0,:,:]), 'b', lw=2, alpha = 0.05)
    if savefig:
        plt.savefig('../GPFlowPref/data/results/T' + str(trial_num) +
                    '/utility_samples/Ihg' + str(iter_num) + 'D' + str(i) + '.png', dpi = 600)
        plt.close()



def visualize_one_hyper_utility_samples(iter_num, trial_num,
                                  m, samples, Xgridnorm, Xgrid,
                                  savefig = True, num_utility = 100):
    """
    Visualize different utility samples given fixed hyperparameters
    """
    
    for s in samples:
        m.set_state(s)
        ftrain = m.predict_f_samples(Xgridnorm, num_utility)
    plt.figure(figsize = (12,8))
    for i in xrange(num_utility):
        plt.plot(Xgrid,ftrain[i,:,0], c = 'b' ,alpha = 0.25)
    #plt.ylim(-8,8)
    plt.title('Utility samples ; Iter : ' + str(iter_num))
    if savefig:
        plt.savefig('../GPFlowPref/data/results/T' + str(trial_num) +
                    '/latent_f/Ih' + str(iter_num) + 'D' + str(i) + '.png', dpi = 600)
        plt.close()

def visualize_utility_train(iter_num, trial_num,
                            m, samples, X, config_file,
                            savefig = True, num_utility = 100):
    """
    Posterior utility function values at training datapoints
    """
    num_feat = X.shape[1]/2
    Xn = np.vstack([X[:,num_feat:], X[:,:num_feat]])
    Xnorm = datagen.ThermalPrefDataGen(config_file).normalize1D(Xn)
    for s in samples:
        m.set_state(s)
        ftrain = m.predict_f_samples(Xnorm, num_utility)
    for i in xrange(samples.shape[1]/2 + 1): 
        plt.figure(figsize = (12,8))
        plt.plot(ftrain[:,i,0])
        plt.ylim(-8,8)
        plt.title(' Training datapoints; Data num : ' + str(i))
        if savefig:
            plt.savefig('../GPFlowPref/data/results/T' + str(trial_num) +
                        '/latent_f_train/It' + str(iter_num) + 'D' + str(i) + '.png', dpi = 600)
        plt.close()
    
def visualize_EUI(Xgrid, mean_imp, iter_num, trial_num, savefig = True):
    plt.figure(figsize = (12,8))
    plt.plot(Xgrid, mean_imp)
    plt.ylabel('Expected Improvement')
    plt.xlabel('State Values')
    plt.title('UEI ; Iter:' + str(iter_num))
    plt.ylim(0, .3)
    if savefig:
        plt.savefig('../GPFlowPref/data/results/T' + str(trial_num) +
                    '/exp_imp_plots/' + str(iter_num) + '.png', dpi = 600)
    plt.close()

def visualize_latent_v(samples,iter_num, trial_num, savefig = True):
    for i in xrange(samples.shape[1]/2 + 1): 
        plt.figure(figsize = (12,8))
        plt.plot(samples[:,i])
        plt.title('Latent v ; Data num : ' + str(i))
        plt.ylim(-3,3)
        if savefig:
            plt.savefig('../GPFlowPref/data/results/T' + str(trial_num) +
                        '/latent_v/It' + str(iter_num) + 'D' + str(i) + '.png', dpi = 600)
        plt.figure(figsize = (12,8))
        plt.hist(samples[:,i])
        plt.xlim(-3,3)
        plt.title('Latent v ; Data num : ' + str(i))
        if savefig:
            plt.savefig('../GPFlowPref/data/results/T' + str(trial_num) +
                        '/latent_v/Ih' + str(iter_num) + 'D' + str(i) + '.png', dpi = 600)
        plt.close()
            
def autocorr(samples,iter_num, trial_num, savefig = True):
    for i in range(samples.shape[1]):  
        fig, ax = plt.subplots()
        ax.acorr(samples[:, i], detrend=plt.mlab.detrend_mean, maxlags=200)
        ax.set_xlim(0, 200)
        ax.set_ylabel('Acorr')
        if savefig:
            fig.savefig('../GPFlowPref/data/results/T' + str(trial_num) +
                        '/autocorr/V' + str(iter_num) + 'D' + str(i) + '.png', dpi = 600)
    return


def hmc_chains(samples, iter_num, trial_num, savefig = True):
    plt.figure(figsize=(16, 4))
    plt.plot(samples)
    if savefig:
            plt.savefig('../GPFlowPref/data/results/T' + str(trial_num) +
                        '/hmc_chains/V' + str(iter_num) + 'D' + '.png',
                        dpi = 600)
    
    