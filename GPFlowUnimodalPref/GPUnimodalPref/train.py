#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:11:18 2018

@author: nimishawalgaonkar
"""

import numpy as np
import json # Needed for config file
import os
import gpflow
 
from .unimodal_gpmc import UnimodalPrefGPMC
from .unimodal_gpmc import UnimodalHiLoPrefGPMC, UnimodalHiLoPrefGPMC1
from .unimodal_gpmc import UnimodalHiLoPrefGPMC2
from .unimodal_gpmc import UnimodalHiLoPrefGPMC3

class Train(object):
    """
    Train the preference learning model using the pairwise comparisons
    Train a GP Preference Learning Model with unimodality constraints
    Inputs:
        X : previous feat columnwise contatenated with current feat X = [X_pre, X_current]
        Y : binary variable indicating whether current or previous state is preferred 
        Y = 1 if current is prefered ; 0 if previous is preferred 
        config_file : visual preferences related configuration settings  
    """
    def __init__(self, X, Y, Yder, X_prime, config_file, model_num = 1):
        self.X = X
        self.Yder = Yder.astype(float)
        self.Y = Y.astype(float)
        self.X_prime = X_prime
        self.config_file = config_file
        
        # sanity check
        assert os.path.isfile(self.config_file)
        num_feat = self.X.shape[1]/2
        self.Xn = np.vstack([self.X[:,num_feat:], self.X[:,:num_feat]])
     
        if model_num == 1:
            # Model Configuration
            m = UnimodalPrefGPMC(self.Xn, self.Y, self.X_prime)
            print '-' * 40
            print 'Model is 1'
            print '-' * 40
            m.kern_f.lengthscale.prior = gpflow.priors.Gamma(1., 1.)
            m.kern_f.signal_variance.prior = gpflow.priors.Gamma(1.,1.)
            m.kern_g.lengthscale.prior = gpflow.priors.Gamma(1., 1.)
            m.kern_g.signal_variance.prior = gpflow.priors.Gamma(1.,1.)
        if model_num == 2:
            # Model Configuration
            m = UnimodalHiLoPrefGPMC(self.Xn, self.Y, self.X_prime)
            print '-' * 40
            print 'Model is 2'
            print '-' * 40
            m.kern_f.lengthscale.prior = gpflow.priors.Gamma(1., 1.)
            m.kern_f.signal_variance.prior = gpflow.priors.Gamma(1.,1.)
            m.kern_g.lengthscale.prior = gpflow.priors.Gamma(1., 1.)
            m.kern_g.signal_variance.prior = gpflow.priors.Gamma(1.,1.)
        if model_num == 3:
            # Model Configuration
            m = UnimodalHiLoPrefGPMC1(self.Xn, self.Y, self.Yder, self.X_prime)
            print '-' * 40
            print 'Model is 3'
            print '-' * 40
            m.kern_f.lengthscale.prior = gpflow.priors.Gamma(1., 1.)
            m.kern_f.signal_variance.prior = gpflow.priors.Gamma(1.,1.)
            m.kern_g.lengthscale.prior = gpflow.priors.Gamma(1., 1.)
            m.kern_g.signal_variance.prior = gpflow.priors.Gamma(1.,1.)
        
        if model_num == 4:
            # Model Configuration
            m = UnimodalHiLoPrefGPMC2(self.Xn, self.Yder, self.X_prime)
            print '-' * 40
            print 'Model is 4'
            print '-' * 40
            m.kern_f.lengthscale.prior = gpflow.priors.Gamma(1., 1.)
            m.kern_f.signal_variance.prior = gpflow.priors.Gamma(1.,1.)
            m.kern_g.lengthscale.prior = gpflow.priors.Gamma(1., 1.)
            m.kern_g.signal_variance.prior = gpflow.priors.Gamma(1.,1.)
        
        if model_num == 5:
            # Model Configuration
            m = UnimodalHiLoPrefGPMC2(self.Xn, self.Yder, self.X_prime)
            print '-' * 40
            print 'Model is 5'
            print '-' * 40
            #print m
            m.kern_f.lengthscale.prior = gpflow.priors.Gamma(1., 1.)
            m.kern_f.signal_variance.prior = gpflow.priors.Gamma(1.,1.)
            #m.kern_f.signal_variance = 1.
            #m.kern_f.signal_variance.fixed =  True
            m.kern_g.lengthscale.prior = gpflow.priors.Gamma(1., 1.)
            m.kern_g.signal_variance.prior = gpflow.priors.Gamma(1.,1.)
        
        if model_num == 6:
            # Model Configuration
            m = UnimodalHiLoPrefGPMC3(self.Xn, self.Yder, self.X_prime)
            print '-' * 40
            print 'Model is 6'
            print '-' * 40
            #print m
            m.kern_f.lengthscale.prior = gpflow.priors.Gamma(1., 1.)
            m.kern_f.signal_variance.prior = gpflow.priors.Gamma(1.,1.)
            #m.kern_f.signal_variance = 1.
            #m.kern_f.signal_variance.fixed =  True
            m.kern_g.lengthscale.prior = gpflow.priors.Gamma(1., 1.)
            m.kern_g.signal_variance.prior = gpflow.priors.Gamma(1.,1.)
        
        self._m = m
        
    def mcmc(self, config_file):
        
        with open(config_file, 'r') as fd:
            config = json.loads(fd.read())
        
        
        # HMC sampling
        MAP_optimize_maxiter = config['MCMC']['MAP_optimize_maxiter']
        num_samples = config['MCMC']['num_samples']
        thin = config['MCMC']['thin']
        burn = config['MCMC']['burn']
        epsilon = config['MCMC']['epsilon']
        Lmax = config['MCMC']['Lmax']
        verbose = eval(config['MCMC']['verbose'])
        
        self._m.optimize(maxiter= MAP_optimize_maxiter) # start near MAP
        samples = self._m.sample(num_samples, verbose= verbose,
                           epsilon= epsilon, thin = thin, burn = burn, Lmax=Lmax)
        
        return self._m, samples
    
    def maxpost(self, config_file):
        with open(config_file, 'r') as fd:
            config = json.loads(fd.read())
        
        # MAP settings
        max_iter =  config['MAP']['max_iter']
        self._m.optimize(maxiter= max_iter) # start near MAP
        
        return self._m