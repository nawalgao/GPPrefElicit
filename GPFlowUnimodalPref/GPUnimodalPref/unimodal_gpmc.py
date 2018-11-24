#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:49:21 2018

@author: nimishawalgaonkar
"""

import numpy as np
import tensorflow as tf
from gpflow.priors import Gaussian
from gpflow.param import AutoFlow, DataHolder, Param
from gpflow._settings import settings
float_type = settings.dtypes.float_type
from .unimodal_gp import UnimodalGP
from .unimodal_conditional import monotone_conditional, grad_monotone_conditional, grad_monotone_conditional1, grad_monotone_conditional_der
from .unimodal_like import UnimodalPrefLikelihood, UnimodalHiLoPrefLike1, UnimodalHiLoPrefLike, UnimodalHiLoPrefLike2

class UnimodalGPMC(UnimodalGP):
    def __init__(self, X, Y, X_prime):
        """
        X is a data vector, size N x 1
        X_prime is a vector, size M x 1
        Y is a data vector, size N x 1 

        This is a vanilla implementation of a GP with unimodality contraints and HMC sampling
        Refer:
        https://bayesopt.github.io/papers/2017/9.pdf
        """
        X_concat = np.vstack([X, X_prime])
        X_concat = DataHolder(X_concat, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        UnimodalGP.__init__(self)
        self.X_concat = DataHolder(X_concat)
        self.Y = DataHolder(Y)
        self.X = DataHolder(X)
        self.X_prime = DataHolder(X_prime)
        self.num_data = X_concat.shape[0]
        self.num_x_points = X.shape[0]
        self.num_der_points = X_prime.shape[0]
        self.num_latent = Y.shape[1]
        
        self.Vf = Param(np.zeros((self.num_data, self.num_latent)))
        self.Vf.prior = Gaussian(0., 1.)
        
        self.Vg = Param(np.zeros((2*self.num_der_points, self.num_latent)))
        self.Vg.prior = Gaussian(0., 1.)
        
    
    def compile(self, session = None, graph = None, optimizer = None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X_concat.shape[0]:
            #print "wink wink"
            self.num_data = self.X_concat.shape[0]
            self.num_x_points = self.X.shape[0]
            self.num_der_points = self.X_prime.shape[0]
            self.Vf = Param(np.zeros((self.num_data, self.num_latent)))
            self.Vf.prior = Gaussian(0., 1.)
            self.Vg = Param(np.zeros((2*self.num_der_points, self.num_latent)))
            self.Vg.prior = Gaussian(0., 1.)
        
        return super(UnimodalGPMC, self).compile(session = session,
                                                 graph = graph,
                                                 optimizer = optimizer)
    def build_likelihood(self):
        Kfjoint = self.kern_f.Kj(self.X, self.X_prime)
        Kgjoint = self.kern_g.Kj(self.X_prime, self.X_prime)
        
        Lf = tf.cholesky(Kfjoint + tf.eye(tf.shape(self.X_concat)[0], dtype=float_type)*
                        settings.numerics.jitter_level)
        Lg = tf.cholesky(Kgjoint + tf.eye(2*tf.shape(self.X_prime)[0], dtype=float_type)*
                        settings.numerics.jitter_level)
        
        F_concat = tf.matmul(Lf, self.Vf)
        F, F_prime = tf.split(F_concat, [self.num_x_points, self.num_der_points])
        
        G_concat = tf.matmul(Lg, self.Vg)
        G, G_prime = tf.split(G_concat, [self.num_der_points, self.num_der_points])
        log_like = self.likelihood.logp(self.Y, F, F_prime, G, G_prime)
        return log_like
    
    def build_predict(self, Xnew):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at X.

        """
        
        #V, V_prime = tf.split(self.V, [self.num_x_points, self.num_der_points])
        mu_f, var_f = monotone_conditional(Xnew, self.X,self.X_prime, self.kern_f, self.Vf, 
                                       whiten=True)
        return mu_f, var_f
    
    def build_predict_g(self, Xnew):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(G* | (G=LgVg) )

        where G* are points on the GP at Xnew, G=LgVg are points on the GP at X.

        """
        
        #V, V_prime = tf.split(self.V, [self.num_x_points, self.num_der_points])
        mu_g, var_g = monotone_conditional(Xnew, self.X_prime,self.X_prime,
                                           self.kern_g, self.Vg, 
                                       whiten=True)
        return mu_g, var_g

class UnimodalPrefGPMC(UnimodalGPMC):
    def __init__(self, X, Y, X_prime):
        """
        X is a data vector, size 2N x 1
        X_prime is a vector, size M x 1
        Y is a data vector, size N x 1 consisting of ones and zeros
        y_i = 1 : current is preferred over previous
        y_i = 0 : previous is preferred over current

        This is a vanilla implementation of a GP preference model with 
        unimodality contraints and HMC sampling
        Refer:
        https://bayesopt.github.io/papers/2017/9.pdf
        """
        
        UnimodalGPMC.__init__(self, X, Y, X_prime)
        self.likelihood = UnimodalPrefLikelihood()
    
    def build_likelihood(self):
        Kfjoint = self.kern_f.Kj(self.X, self.X_prime)
        Kgjoint = self.kern_g.Kj(self.X_prime, self.X_prime)
        
        Lf = tf.cholesky(Kfjoint + tf.eye(tf.shape(self.X_concat)[0], dtype=float_type)*
                        settings.numerics.jitter_level)
        Lg = tf.cholesky(Kgjoint + tf.eye(2*tf.shape(self.X_prime)[0], dtype=float_type)*
                        settings.numerics.jitter_level)
        
        F_concat = tf.matmul(Lf, self.Vf)
        F, F_prime = tf.split(F_concat, [self.num_x_points, self.num_der_points])
        
        G_concat = tf.matmul(Lg, self.Vg)
        G, G_prime = tf.split(G_concat, [self.num_der_points, self.num_der_points])
        log_like = self.likelihood.logp(self.Y, F, F_prime, G, G_prime)
        return log_like

class UnimodalHiLoPrefGPMC(UnimodalPrefGPMC):
     """
      X is a data vector, size 2N x 1
      X_prime is a vector, size M x 1
      Y (hi_low_pref) : what do you prefer (high or low)
      high : 1
      low : -1
      This is a vanilla implementation of a GP preference model with 
      unimodality contraints and HMC sampling
      when information about the gradient of utility is available    
      For example:
            x1     x2      what do you want? High or low?
            20     24      want high then we get y = 1 and z(24) = 1 
            24     27      want low then we get y = 0 and z(24) = -1
            27     26      want low then we get y = 0 and z(26) = -1
      Hence we need to add one more component to our log-like
      Refer pg. 16 FALL 18 Notability notes
      Refer:
      https://bayesopt.github.io/papers/2017/9.pdf
     """
     def __init__(self, X, Yder, X_prime):
         X_cur = X[:X.shape[0]/2, :]
         X_prime_concat = np.vstack([X_cur, X_prime])
         self.X_prime_concat = DataHolder(X_prime_concat, on_shape_change='recompile')
         X_concat = np.vstack([X, X_prime_concat])
         X_concat = DataHolder(X_concat, on_shape_change='recompile')
         #Yder = DataHolder(Yder, on_shape_change='recompile')
         UnimodalGP.__init__(self)
         
         self.X_concat = X_concat
         self.X = DataHolder(X)
         self.X_cur = DataHolder(X_cur)
         self.X_prime = DataHolder(X_prime)
         
         self.num_data = X_concat.shape[0]
         self.num_xcur_points = X_cur.shape[0]
         self.num_x_points = X.shape[0]
         self.num_der_points = X_prime.shape[0]
         self.num_latent = Yder.shape[1]
         
         self.Vf = Param(np.zeros((self.num_data, self.num_latent)))
         self.Vfdd = Param(np.zeros((self.X_prime_concat.shape[0],
                                     self.num_latent)))
         self.Vf.prior = Gaussian(0., 1.)
        
         self.Vg = Param(np.zeros((2*self.num_der_points, self.num_latent)))
         self.Vg.prior = Gaussian(0., 1.)
         
         self.likelihood = UnimodalHiLoPrefLike1()
         
         if isinstance(Yder, np.ndarray):
            #: Y is a data matrix, rows correspond to the rows in X, columns are treated independently
            Yder = DataHolder(Yder)
         self.Yder = Yder
     def build_likelihood(self):
         Kfjoint = self.kern_f.Kj(self.X, self.X_prime_concat)
         Kgjoint = self.kern_g.Kj(self.X_prime, self.X_prime)
         self.Kf_comp = self.kern_f.compute_Kj(self.X, self.X_prime_concat)
         
         Lf = tf.cholesky(Kfjoint + tf.eye(tf.shape(self.X_concat)[0], dtype=float_type)*
                          settings.numerics.jitter_level)
         Lg = tf.cholesky(Kgjoint + tf.eye(2*tf.shape(self.X_prime)[0], dtype=float_type)*
                          settings.numerics.jitter_level)
         
         F_concat = tf.matmul(Lf, self.Vf)
         F, F_prime_z_obs, F_prime = tf.split(F_concat, [self.num_x_points,self.num_xcur_points, self.num_der_points])
        
         G_concat = tf.matmul(Lg, self.Vg)
         G, G_prime = tf.split(G_concat, [self.num_der_points, self.num_der_points])
         log_like = self.likelihood.logp(F_prime, G, G_prime, self.Yder, F_prime_z_obs)
         return log_like

class UnimodalHiLoPrefGPMC1(UnimodalHiLoPrefGPMC):
    def __init__(self, X, Y, Yder, X_prime):
        UnimodalHiLoPrefGPMC.__init__(self, X, Yder, X_prime)
        self.likelihood = UnimodalHiLoPrefLike()
        if isinstance(Yder, np.ndarray):   
            Y = DataHolder(Y)
        self.Y = Y
        self.jitter = 10e-4
    
    def build_likelihood(self):
         Kfjoint = self.kern_f.Kj(self.X, self.X_prime_concat)
         Kgjoint = self.kern_g.Kj(self.X_prime, self.X_prime)
        
         Lf = tf.cholesky(Kfjoint + tf.eye(tf.shape(self.X_concat)[0], dtype=float_type)*
                          self.jitter)
         Lg = tf.cholesky(Kgjoint + tf.eye(2*tf.shape(self.X_prime)[0], dtype=float_type)*
                          self.jitter)
         
         F_concat = tf.matmul(Lf, self.Vf)
         F, F_prime_z_obs, F_prime = tf.split(F_concat, [self.num_x_points,self.num_xcur_points, self.num_der_points])
        
         G_concat = tf.matmul(Lg, self.Vg)
         G, G_prime = tf.split(G_concat, [self.num_der_points, self.num_der_points])
         log_like = self.likelihood.logp(self.Y, F, F_prime, G, G_prime, self.Yder, F_prime_z_obs)
         return log_like
    
    def build_predict(self, Xnew):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at X.

        """
        
        #V, V_prime = tf.split(self.V, [self.num_x_points, self.num_der_points])
        mu_f, var_f = grad_monotone_conditional(Xnew, self.X, self.X_prime_concat,
                                                self.kern_f, self.Vf, self.jitter,
                                                whiten=True)
        return mu_f, var_f
    
    def build_predict_g(self, Xnew):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(G* | (G=LgVg) )

        where G* are points on the GP at Xnew, G=LgVg are points on the GP at X.

        """
        
        #V, V_prime = tf.split(self.V, [self.num_x_points, self.num_der_points])
        mu_g, var_g = monotone_conditional(Xnew, self.X_prime,self.X_prime,
                                           self.kern_g, self.Vg, 
                                       whiten=True)
        return mu_g, var_g
    
    
class UnimodalHiLoPrefGPMC2(UnimodalGP):
    def __init__(self, X_total, Yder, X_prime):
        """
        Modeling assumption : We just observe gradient values.
        For example, we have
        x    x    yder          y
        20   24    1 (high)     1
        24   22    1 (high)     not sure
        22   26    -1 (low)     not sure
        
        X_total is a data vector, size 2N x 1
        X_prime is a vector, size M x 1
        Y (hi_low_pref) : what do you prefer (high or low)
        high : 1
        low : -1
        This is a vanilla implementation of a GP preference model with 
        unimodality contraints and HMC sampling
        when information about the gradient of utility is available 
        """
        UnimodalGP.__init__(self)
        
        X_cur = X_total[:X_total.shape[0]/2, :]
        X_prev = X_total[X_total.shape[0]/2:, :]
        
        X_prime_concat = np.vstack([X_cur, X_prime])
    
        # come up with X_rel
        cur_greater_prev = 1*(X_cur > X_prev)
        #print 'shape 1'
        #print cur_greater_prev.shape
        prev_greater_cur = 1*(np.logical_not(cur_greater_prev))
        #print 'shape 2'
        #print prev_greater_cur.shape
        pref_high = 1*(Yder == 1.)
        #print 'shape 3'
        #print pref_high.shape
        pref_low = 1*(Yder == -1)
        #print 'shape 4'
        #print pref_low.shape
        bool1 = cur_greater_prev*pref_high
        #print 'shape 5'
        #print bool1.shape
        bool2 = prev_greater_cur*pref_low
        #print 'shape 5'
        #print bool2.shape
        bool_add = bool1 + bool2
        #print 'shape 6'
        #print bool2.shape
        #print X_cur.shape
        #print 'bool add'
        #print bool_add.shape 
        rel_X_cur = X_cur[bool_add == 1]
        #print 'shape 7'
        #print rel_X_cur.shape
        rel_X_prev = X_prev[bool_add == 1]
        #print 'shape 8'
        #print rel_X_prev.shape
        X_rel = np.vstack([rel_X_cur[:,None], rel_X_prev[:,None]])
        rel_Yder = Yder[bool_add == 1] 
       
        
        
        self.X_prime_concat = DataHolder(X_prime_concat, on_shape_change='recompile')
        X_concat = np.vstack([X_rel, X_prime_concat])
        X_concat = DataHolder(X_concat, on_shape_change='recompile')
        
        self.X_concat = X_concat
        self.X_total = DataHolder(X_total)
        self.X_rel = DataHolder(X_rel)
        self.X_cur = DataHolder(X_cur)
        self.X_prime = DataHolder(X_prime)
      
        self.Yder = DataHolder(Yder)
        
        self.num_data = X_concat.shape[0]
        self.num_xcur_points = X_cur.shape[0]
        self.num_x_points = X_rel.shape[0]
        self.num_der_points = X_prime.shape[0]
        self.num_latent = Yder.shape[1]
        
        self.Vf = Param(np.zeros((self.num_data, self.num_latent)))
        self.Vfdd = Param(np.zeros((self.X_prime_concat.shape[0],
                                     self.num_latent)))
        self.Vf.prior = Gaussian(0., 1.)
        
        self.Vg = Param(np.zeros((2*self.num_der_points, self.num_latent)))
        self.Vg.prior = Gaussian(0., 1.)
        
        self.likelihood = UnimodalHiLoPrefLike1()
        self.jitter = 10e-4
    
    def build_likelihood(self):
        Kfjoint = self.kern_f.Kj(self.X_rel, self.X_prime_concat)
        Kgjoint = self.kern_g.Kj(self.X_prime, self.X_prime)
        
        Lf = tf.cholesky(Kfjoint + tf.eye(tf.shape(self.X_concat)[0], dtype=float_type)*
                          self.jitter)
        Lg = tf.cholesky(Kgjoint + tf.eye(2*tf.shape(self.X_prime)[0], dtype=float_type)*
                          self.jitter)
        
        
        F_concat = tf.matmul(Lf, self.Vf)
        F, F_prime_z_obs, F_prime = tf.split(F_concat, [self.num_x_points,self.num_xcur_points, self.num_der_points])
        
        G_concat = tf.matmul(Lg, self.Vg)
        G, G_prime = tf.split(G_concat, [self.num_der_points, self.num_der_points])
        log_like = self.likelihood.logp(F, F_prime, G, G_prime, self.Yder, F_prime_z_obs)
        
        return log_like
    
    def build_predict(self, Xnew):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at X.

        """
        
        #V, V_prime = tf.split(self.V, [self.num_x_points, self.num_der_points])
        mu_f, var_f = grad_monotone_conditional(Xnew, self.X_rel, self.X_prime_concat,
                                                self.kern_f, self.Vf, self.jitter,
                                                whiten=True)
        return mu_f, var_f

    def build_predict_fder(self, Xnew):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at X.

        """
        
        #V, V_prime = tf.split(self.V, [self.num_x_points, self.num_der_points])
        mu_f, var_f = grad_monotone_conditional_der(Xnew, self.X_rel, self.X_prime_concat,
                                                self.kern_f, self.Vf, self.jitter,
                                                whiten=True)
        return mu_f, var_f
    
    def build_predict_g(self, Xnew):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(G* | (G=LgVg) )

        where G* are points on the GP at Xnew, G=LgVg are points on the GP at X.

        """
        
        #V, V_prime = tf.split(self.V, [self.num_x_points, self.num_der_points])
        mu_g, var_g = monotone_conditional(Xnew, self.X_prime,self.X_prime,
                                           self.kern_g, self.Vg, 
                                       whiten=True)
        return mu_g, var_g
    
    

class UnimodalHiLoPrefGPMC3(UnimodalHiLoPrefGPMC2):
    def __init__(self, X_total, Yder, X_prime):
        UnimodalGP.__init__(self)
        X_cur = X_total[:X_total.shape[0]/2, :]
        X_prime_concat = np.vstack([X_cur, X_prime])
        
        self.num_data = X_prime_concat.shape[0]
        self.num_der_points = X_prime.shape[0]
        self.num_latent = Yder.shape[1]
        self.num_xcur_points = X_cur.shape[0]
        
        self.Yder = DataHolder(Yder)
        self.X_prime = DataHolder(X_prime)
        self.X_prime_concat = DataHolder(X_prime_concat, on_shape_change='recompile')
        
        self.Vf = Param(np.zeros((self.num_data, self.num_latent)))
        self.Vf.prior = Gaussian(0., 1.)
        self.Vg = Param(np.zeros((2*self.num_der_points, self.num_latent)))
        self.Vg.prior = Gaussian(0., 1.)
        
        self.likelihood = UnimodalHiLoPrefLike2()
        self.jitter = 10e-4
    
    def build_likelihood(self):
        Kfprime = self.kern_f.Kdd(self.X_prime_concat, self.X_prime_concat)
        Kgjoint = self.kern_g.Kj(self.X_prime, self.X_prime)
        
        Lf = tf.cholesky(Kfprime + tf.eye(tf.shape(self.X_prime_concat)[0], dtype=float_type)*
                          self.jitter)
        Lg = tf.cholesky(Kgjoint + tf.eye(2*tf.shape(self.X_prime)[0], dtype=float_type)*
                          self.jitter)
        
        F_concat = tf.matmul(Lf, self.Vf)
        F_prime_z_obs, F_prime = tf.split(F_concat, [self.num_xcur_points, self.num_der_points])
        
        G_concat = tf.matmul(Lg, self.Vg)
        G, G_prime = tf.split(G_concat, [self.num_der_points, self.num_der_points])
        
        log_like = self.likelihood.logp(F_prime, G, G_prime, self.Yder, F_prime_z_obs)
        
        return log_like
    
    def build_predict(self, Xnew):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at X.

        """
        
        #V, V_prime = tf.split(self.V, [self.num_x_points, self.num_der_points])
        mu_f, var_f = grad_monotone_conditional1(Xnew, self.X_prime_concat,
                                                self.kern_f, self.Vf, self.jitter,
                                                whiten=True)
        return mu_f, var_f
        
       
        
    
        
        
         
    
        
    
        