#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 11:36:54 2018

@author: nimishawalgaonkar
"""

import numpy as np
import tensorflow as tf
import gpflow
from gpflow.mean_functions import Zero
from gpflow.param import AutoFlow, DataHolder
from gpflow._settings import settings
float_type = settings.dtypes.float_type

from .ker_w_der import ExtendRBF1D
from .unimodal_like import UnimodalLikelihood

class UnimodalGP(gpflow.model.Model):
    
    def __init__(self, name = 'unimodal_model'):
        """
        X_concat is a data vector, size (N + M) x 1
        X_concat = (X, X_der_loc)
        Y is a data matrix, size N x 1 
    
        This is a vanilla implementation of a GP with unimodal contraints.
        
        Refer:
        https://bayesopt.github.io/papers/2017/9.pdf
        """
        
        # Initialize the model
        gpflow.model.Model.__init__(self, name)
        # Zero mean function for now
        self.mean_function = Zero()
        
#        # Initialize data
#        if isinstance(X_concat, np.ndarray):
#            #: X is a data matrix; each row represents one instance
#            X_concat = DataHolder(X_concat)
#        if isinstance(Y, np.ndarray):
#            #: Y is a data matrix, rows correspond to the rows in X, columns are treated independently
#            Y = DataHolder(Y)
        
        # Define Kernels
        self.kern_f = ExtendRBF1D()
        self.kern_g = ExtendRBF1D()
        
        # Define Likelihood 
        #self.likelihood = UnimodalLikelihood()
        
        #self.likelihood._check_targets(Y.value)
        
#        # Initialize
#        self.Y = Y
#        self.X_concat = X_concat
    
    def build_predict(self, *args, **kwargs):
        raise NotImplementedError

    @AutoFlow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict(Xnew)

    @AutoFlow((float_type, [None, None]))
    def predict_fder(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_fder(Xnew)
    
    @AutoFlow((float_type, [None, None]))
    def predict_g(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_g(Xnew)

    @AutoFlow((float_type, [None, None]))
    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        return self.build_predict(Xnew)

    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict(Xnew)
        jitter = tf.eye(tf.shape(mu)[0], dtype=float_type) * settings.numerics.jitter_level
        samples = []
        for i in range(self.num_latent):
            L = tf.cholesky(var[:, :, i] + jitter)
            shape = tf.stack([tf.shape(L)[0], num_samples])
            V = tf.random_normal(shape, dtype=settings.dtypes.float_type)
            samples.append(mu[:, i:i + 1] + tf.matmul(L, V))
        return tf.transpose(tf.stack(samples))
    
    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def predict_g_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict_g(Xnew)
        jitter = tf.eye(tf.shape(mu)[0], dtype=float_type) * settings.numerics.jitter_level
        samples = []
        for i in range(self.num_latent):
            L = tf.cholesky(var[:, :, i] + jitter)
            shape = tf.stack([tf.shape(L)[0], num_samples])
            V = tf.random_normal(shape, dtype=settings.dtypes.float_type)
            samples.append(mu[:, i:i + 1] + tf.matmul(L, V))
        return tf.transpose(tf.stack(samples))

    @AutoFlow((float_type, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)


    
    