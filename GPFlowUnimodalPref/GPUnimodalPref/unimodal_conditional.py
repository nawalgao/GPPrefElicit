#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 12:54:24 2018

@author: nimishawalgaonkar
"""

import tensorflow as tf
from gpflow.scoping import NameScoped
from gpflow._settings import settings
float_type = settings.dtypes.float_type

@NameScoped("monotone_conditional")
def monotone_conditional(Xnew, X, X_prime, kern, f_concat, whiten=False):
    """
    Given F, representing the GP at the points X, and
    F_prime representing dervative values of F at X_prime points,
    produce the mean and (co-)variance of the GP at the points Xnew.

    Additionally, the GP may have been centered (whitened) so that
        p(v) = N( 0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case 'f' represents the values taken by v.

    The method can either return the diagonals of the covariance matrix for
    each output of the full covariance matrix (full_cov).

     - Xnew is a data matrix, size P x 1
     - X are data points, size N x 1
     - X_prine are datapoints, size M x 1
     
     
     - kern is a GPflow kernel which supports monotonocity (eg. ExtendRBF1D())
     - f is a data matrix, N x K, representing the function values at X, for K functions. K = 1 for now
     - q_sqrt (optional) is a matrix of standard-deviations or Cholesky
       matrices, size M x K or M x M x K
     - whiten (optional) is a boolean: whether to whiten the representation
       as described above.
    """
    
    # compute kernel stuff
    X_concat = tf.concat([X, X_prime], 0)
    num_data = tf.shape(X_concat)[0]
    num_func = 1 # only one output GP
    
    Kmm = kern.Kj(X, X_prime) + tf.eye(num_data, dtype=float_type) * settings.numerics.jitter_level
    Kmn = kern.Kmn(Xnew, X, X_prime)
    Lm  = tf.cholesky(Kmm)
    
    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower = True)
    
    # compute the covariance due to the conditioning
    fvar = kern.K(Xnew, Xnew) - tf.matmul(A, A, transpose_a=True)
    shape = tf.stack([num_func, 1, 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # K x N x N or K x N
    
    # another backsubstitution in the unwhitened case
    if not whiten:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f_concat, transpose_a=True)

    fvar = tf.transpose(fvar)  # N x K or N x N x K

    return fmean, fvar

@NameScoped("grad_monotone_conditional")
def grad_monotone_conditional(Xnew, X, X_prime_concat, kern, f_concat, jitter, whiten=False):
    """
    Given F, representing the GP at the points X, and
    F_prime representing dervative values of F at X_prime points,
    produce the mean and (co-)variance of the GP at the points Xnew.

    Additionally, the GP may have been centered (whitened) so that
        p(v) = N( 0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case 'f' represents the values taken by v.

    The method can either return the diagonals of the covariance matrix for
    each output of the full covariance matrix (full_cov).

     - Xnew is a data matrix, size P x 1
     - X are data points, size N x 1
     - X_prine are datapoints, size M x 1
     
     
     - kern is a GPflow kernel which supports monotonocity (eg. ExtendRBF1D())
     - f is a data matrix, N x K, representing the function values at X, for K functions. K = 1 for now
     - q_sqrt (optional) is a matrix of standard-deviations or Cholesky
       matrices, size M x K or M x M x K
     - whiten (optional) is a boolean: whether to whiten the representation
       as described above.
    """
    
    # compute kernel stuff
    X_concat = tf.concat([X, X_prime_concat], 0)
    num_data = tf.shape(X_concat)[0]
    num_func = 1 # only one output GP
    
    Kmm = kern.Kj(X, X_prime_concat) + tf.eye(num_data, dtype=float_type) * jitter
    Kmn = kern.Kmn(Xnew, X, X_prime_concat)
    Lm  = tf.cholesky(Kmm)
    
    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower = True)
    
    # compute the covariance due to the conditioning
    fvar = kern.K(Xnew, Xnew) - tf.matmul(A, A, transpose_a=True)
    shape = tf.stack([num_func, 1, 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # K x N x N or K x N
    
    # another backsubstitution in the unwhitened case
    if not whiten:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f_concat, transpose_a=True)

    fvar = tf.transpose(fvar)  # N x K or N x N x K

    return fmean, fvar


@NameScoped("grad_monotone_conditional_der")
def grad_monotone_conditional_der(Xnew, X, X_prime_concat, kern, f_concat, jitter, whiten=False):
    """
    Given F, representing the GP at the points X, and
    F_prime representing dervative values of F at X_prime points,
    produce the mean and (co-)variance of the GP at the points Xnew.

    Additionally, the GP may have been centered (whitened) so that
        p(v) = N( 0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case 'f' represents the values taken by v.

    The method can either return the diagonals of the covariance matrix for
    each output of the full covariance matrix (full_cov).

     - Xnew is a data matrix, size P x 1
     - X are data points, size N x 1
     - X_prine are datapoints, size M x 1
     
     
     - kern is a GPflow kernel which supports monotonocity (eg. ExtendRBF1D())
     - f is a data matrix, N x K, representing the function values at X, for K functions. K = 1 for now
     - q_sqrt (optional) is a matrix of standard-deviations or Cholesky
       matrices, size M x K or M x M x K
     - whiten (optional) is a boolean: whether to whiten the representation
       as described above.
    """
    
    # compute kernel stuff
    X_concat = tf.concat([X, X_prime_concat], 0)
    num_data = tf.shape(X_concat)[0]
    num_func = 1 # only one output GP
    
    Kmm = kern.Kj(X, X_prime_concat) + tf.eye(num_data, dtype=float_type) * jitter
    Kmn = kern.Kgh(Xnew, X, X_prime_concat)
    Lm  = tf.cholesky(Kmm)
    
    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower = True)
    
    # compute the covariance due to the conditioning
    fvar = kern.Kdd(Xnew, Xnew) - tf.matmul(A, A, transpose_a=True)
    shape = tf.stack([num_func, 1, 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # K x N x N or K x N
    
    # another backsubstitution in the unwhitened case
    if not whiten:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f_concat, transpose_a=True)

    fvar = tf.transpose(fvar)  # N x K or N x N x K

    return fmean, fvar


@NameScoped("grad_monotone_conditional1")
def grad_monotone_conditional1(Xnew, X_prime_concat, kern, f_concat, jitter, whiten=False):
    """
    Given F, representing the GP at the points X, and
    F_prime representing dervative values of F at X_prime points,
    produce the mean and (co-)variance of the GP at the points Xnew.

    Additionally, the GP may have been centered (whitened) so that
        p(v) = N( 0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case 'f' represents the values taken by v.

    The method can either return the diagonals of the covariance matrix for
    each output of the full covariance matrix (full_cov).

     - Xnew is a data matrix, size P x 1
     - X are data points, size N x 1
     - X_prine are datapoints, size M x 1
     
     
     - kern is a GPflow kernel which supports monotonocity (eg. ExtendRBF1D())
     - f is a data matrix, N x K, representing the function values at X, for K functions. K = 1 for now
     - q_sqrt (optional) is a matrix of standard-deviations or Cholesky
       matrices, size M x K or M x M x K
     - whiten (optional) is a boolean: whether to whiten the representation
       as described above.
    """
    
    num_data = tf.shape(X_prime_concat)[0]
    num_func = 1 # only one output GP
    
    Kmm = kern.Kdd(X_prime_concat, X_prime_concat) + tf.eye(num_data, dtype=float_type) * jitter
    Kmn = tf.transpose(kern.Kd(Xnew, X_prime_concat)) # here you might eed to transpose
    Lm  = tf.cholesky(Kmm)
    
    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower = True)
    
    # compute the covariance due to the conditioning
    fvar = kern.K(Xnew, Xnew) - tf.matmul(A, A, transpose_a=True)
    shape = tf.stack([num_func, 1, 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # K x N x N or K x N
    
    # another backsubstitution in the unwhitened case
    if not whiten:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)
        
    # construct the conditional mean
    fmean = tf.matmul(A, f_concat, transpose_a=True)

    fvar = tf.transpose(fvar)  # N x K or N x N x K

    return fmean, fvar
    