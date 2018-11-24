#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:29:40 2018

@author: nimishawalgaonkar
"""

import numpy as np

class Predict(object):
    """
    Preference Model Prediction class
    """
    
    def __init__(self, m):
        self.m = m
        
    def u_mcmc(self, samples,  Xnorm):
        """
        Utility posterior function values for normalized state matrix X_norm
        Inputs:
            Xnorm (N X 1) when only 1D (VI) feature vector or
            Xnorm (N X 2) when 2D (VI + SP) feature matrix
        Ouput:
            Posterior mean and variance of GP (for each hyperparameter sample)
        """
        meanmat = np.zeros(shape = (samples.shape[0], Xnorm.shape[0]))
        varmat = np.zeros(shape = (samples.shape[0], Xnorm.shape[0]))
        for i, s in enumerate(samples):
            self.m.set_state(s)
            mean, var = self.m.predict_f(Xnorm)
            meanmat[i,:] = mean[:,0]
            #print 'var shape is '
            #print var.shape
            varmat[i,:] = np.diag(var[:,:,0])
        
        return meanmat, varmat
    
    def g_mcmc(self, samples,  Xnorm):
        """
        Utility posterior function values for normalized state matrix X_norm
        Inputs:
            Xnorm (N X 1) when only 1D (VI) feature vector or
            Xnorm (N X 2) when 2D (VI + SP) feature matrix
        Ouput:
            Posterior mean and variance of GP (for each hyperparameter sample)
        """
        meanmat = np.zeros(shape = (samples.shape[0], Xnorm.shape[0]))
        varmat = np.zeros(shape = (samples.shape[0], Xnorm.shape[0]))
        for i, s in enumerate(samples):
            self.m.set_state(s)
            mean, var = self.m.predict_g(Xnorm)
            meanmat[i,:] = mean[:,0]
            #print 'var shape is '
            #print var.shape
            varmat[i,:] = np.diag(var[:,:,0])
        
        return meanmat, varmat
    
    def u_test_train_mcmc(self, samples, Xtrainnorm, Xtestnorm):
        """
        Posterior GP mean and variance at training and testing state
        for each hyperparameter sample
        Inputs:
            Xtrainnorm : Normalized training state values
            Xtestnorm : Normalized testing state values
        Outputs:
            mtrainmat : GP mean values at training points
            vartrainmat : Variance of GP at training points
            mtestmat : GP mean at testing points | GP at training points
            vartestmat : Variance of GP at testing points | GP at training points
        """
        num_feat = Xtrainnorm.shape[1]/2
        Xtrain_seccol_norm = Xtrainnorm[:,num_feat:]
        mtrainmat, vartrainmat = self.u_mcmc(samples, Xtrain_seccol_norm)
        mtestmat, vartestmat = self.u_mcmc(samples, Xtestnorm)
        
        return mtrainmat, vartrainmat, mtestmat, vartestmat 
    
    def u_map(self, Xnorm):
        """
        Inputs:
            Xnorm (N X 1) when only 1D (VI) feature vector or
            Xnorm (N X 2) when 2D (VI + SP) feature matrix
        Outputs:
           Posterior mean and variance of GP (for MAP hyperparameter sample) 
        """
        mean, var = self.m.predict_f(Xnorm)
        
        return mean, var
    
    def u_test_train_map(self, Xtrainnorm, Xtestnorm):
        """
        Posterior GP mean and variance at training and testing state
        for each hyperparameter sample
        Inputs:
            Xtrainnorm : Normalized training state values
            Xtestnorm : Normalized testing state values
        Outputs:
            mtrain : GP mean value at training points
            vartrain : Variance of GP at training points
            mtest : GP mean at testing points | GP at training points
            vartest : Variance of GP at testing points | GP at training points
        """
        mtrain, vartrain = self.u_map(Xtrainnorm)
        mtest, vartest = self.u_map(Xtestnorm)
        
        return mtrain, vartrain, mtest, vartest