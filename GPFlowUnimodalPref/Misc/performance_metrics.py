# -*- coding: utf-8 -*-
"""
Created on Sun May 28 21:52:15 2017

@author: nawalgao
"""
import numpy as np
from scipy.spatial import distance


class PerformanceMetrics(object):
    
    def __init__(self, p_pred_mat, y_actual):
        """
        p_pred_mat: N x S matrix (N: total number of duels, S: number of posterior samples)
        y_act : actual preferences (1 or 0)
        """
        self.y_actual = y_actual
        self.p_pred_mat = p_pred_mat
        self.p_pred_mean = np.mean(self.p_pred_mat, axis = 1)
        
    def hit_rate(self):
        """
        Hit rate accuracy 
        (refer to our Purdue conference preference elicitation paper)
        """
        pred = self.p_pred_mean > 0.5
        diff = pred.astype(int)- self.y_actual
        test_acc = 1. -np.sum(np.abs(diff)).astype(float)/diff.shape[0]
        return test_acc
    
    def euclidean(self):
         """
         Euclidean distance
         (refer to our Purdue conference preference elicitation paper)
         """
         dst = distance.euclidean(self.p_pred_mean, self.y_actual)
         return dst
     
    def out_of_sample_deviance(self):
        """
        Calculates the out-of sample deviance (deviance of the testing set) when we know the actual preference probability
        Inputs:
            p_pred_mat: N x S matrix (N: total number of duels, S: number of posterior samples)
            p_act: actual preference probability
    
        Outputs:
            out-of sample deviance
        """
        p_diff = self.p_pred_mat - self.y_actual
        p_diff_abs = np.abs(p_diff)
        log_ev = np.log(1 - p_diff_abs + 1e-10)
        dev = -2 * log_ev
        dev_sum = np.sum(dev, axis = 1)
        dev_mean = np.mean(dev_sum)
        return dev_mean