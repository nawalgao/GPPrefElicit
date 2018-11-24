#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 11:40:39 2018

@author: nimishawalgaonkar
"""

import gpflow
import tensorflow as tf
from gpflow.param import AutoFlow
from gpflow._settings import settings
float_type = settings.dtypes.float_type

class ExtendRBF1D(gpflow.kernels.Kern):
    """
    Kernel for monotonicity models
    """
    def __init__(self):
        gpflow.kernels.Kern.__init__(self, input_dim = 1, active_dims= [0])
        self.lengthscale = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)
        self.signal_variance = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)
    
    def K(self, X1, X2):
        """
        Normal RBF kernel
        Covariance between function  values at X1 and function values at X2
        """
        cov = self.signal_variance * tf.exp(-1./(2 * tf.square(self.lengthscale)) * tf.square(X1 - tf.transpose(X2)))
        return cov

    def Kd(self, X1, X2):
        """
        Covariance between gaussian process at X2 and its derivative at X1
        X1, X2 order matters in this scenario
        Please be careful
        """
        cov = (self.signal_variance * 
               tf.exp(-1./(2 * tf.square(self.lengthscale)) * tf.square(X1 - tf.transpose(X2))) *
              (1./tf.square(self.lengthscale) * (X1 - tf.transpose(X2))))
        return cov
    
    def Kdd(self, X1, X2):
        """
        Covariance between derivatives of gaussian process at X1 and X2
        """
        cov = (self.signal_variance * 
               tf.exp(-1./(2 * tf.square(self.lengthscale)) * tf.square(X1 - tf.transpose(X2)))* 
               1./tf.square(self.lengthscale) *
              (1. - 1./tf.square(self.lengthscale) * tf.square(X1 - tf.transpose(X2))))
        return cov
    
    def Kj(self, X, X_prime):
        """
        Covariance matrix for joint Normal distribution over GP and its derivative
        """
        K_f_f = self.K(X, X)
        K_f_fprime = self.Kd(X, X_prime)
        K_fprime_f = tf.transpose(K_f_fprime)
        K_fprime_fprime = self.Kdd(X_prime, X_prime)
        Knew1 = tf.concat([K_f_f, K_f_fprime], 1)
        Knew2 = tf.concat([K_fprime_f, K_fprime_fprime], 1)
        K_joint = tf.concat([Knew1, Knew2], 0)
        return K_joint 
    
    def Kmn(self, X_new, X, X_prime):
        """
        Concatenate covariance tensor Kmn = [K(X, Xnew), Kd(X_prime, Xnew)]
        """
        K1 = self.K(X, X_new)
        K2 = tf.transpose(self.Kd(X_new, X_prime))
        Kc = tf.concat([K1, K2], 0)
        return Kc

    def Kgh(self, X_new, X, X_prime):
        """
        Concatenate covariance tensor Kgh = [Kd(Xnew, X), Kdd(X_prime, Xnew)]
        """
        K1 = tf.transpose(self.Kd(X_new, X))
        K2 = tf.transpose(self.Kdd(X_new, X_prime))
        Kc = tf.concat([K1, K2], 0)
        return Kc
        
    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def compute_K(self, X, Z):
        return self.K(X, Z)
    
    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def compute_Kd(self, X, Z):
        return self.Kd(X, Z)
    
    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def compute_Kdd(self, X, Z):
        return self.Kdd(X, Z)
    
    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def compute_Kj(self, X, X_prime):
        return self.Kj(X, X_prime)
    
    @AutoFlow((float_type, [None, None]),
              (float_type, [None, None]), (float_type, [None, None]))
    def compute_Kmn(self, Xnew, X, X_prime):
        return self.Kmn(Xnew, X, X_prime)
    
    @AutoFlow((float_type, [None, None]),
              (float_type, [None, None]), (float_type, [None, None]))
    def compute_Kgh(self, Xnew, X, X_prime):
        return self.Kgh(Xnew, X, X_prime)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

if __name__ == '__main__':
    
    import numpy as np
    a = np.arange(100)
    b = np.arange(100)
    c = np.array([6])
    K = ExtendRBF1D()
    #kmn1 = K.compute_K(a[:,None], c[:,None])
    #kmn2 = K.compute_Kd(c[:,None], a[:,None])
    #kgh1 = K.compute_Kd(c[:,None], a[:,None])
    #kgh2 = K.compute_Kdd(c[:,None], a[:,None])
    kmn = K.compute_Kmn(c[:,None], a[:,None], b[:,None])
    kgh = K.compute_Kgh(c[:,None], a[:,None], b[:,None])
    #kmn = K.compute_Kj(a[:,None], b[:,None])
    
  
    