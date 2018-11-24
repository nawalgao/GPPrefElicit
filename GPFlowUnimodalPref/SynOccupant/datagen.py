"""
Synthetic Occupant Data Generation Schemes 1D and 2D
"""


import numpy as np
from scipy.stats import norm 
import json

import objfunc


def probit_link_func(x):
    """
    Probit link function
    Output : CDF of x 
    """
    return norm.cdf(x)

def gen_output(u1, u2):
    """
    Inputs:
    u1 : first column
    u2 : second column
    Outputs:
    y : binary variable (1 or 0) ; 1 if x2 is preferred ; 0 if x1 is preferred
    p: actual preference probability
    """
    # utility function value generation
    diff = u2 - u1
    y = 1*(diff > 0.)
    #p = probit_link_func(3*diff)
    #y = bernoulli.rvs(p)
    return y


class ThermalPrefDataGen(object):
    def __init__(self, config_file):
        """
        Synthetic occupant pairwise comparison data generation
        We assume that there is a synthetic occupant who has a specific utility.
        This class is used to generate 1D and 2D feature state based pairwise
        comparisons. We assume that the visual state is defined by two features,
        Operating temp and relative humidity. If you want to add more feature
        to define the state of the room, you need to work on adding method to this
        class which would take care of 3D feature pairwise comparison data.
        """
        # Read configuration file
        with open(config_file, 'r') as fd:
            config = json.loads(fd.read())
        
        # Grid parameters
        self.Gridmin1 = config['Grid']['min1']
        self.Gridmax1 = config['Grid']['max1']
        self.temp_diff = config['Grid']['temp_diff1']
        
        # Thermal utility function parameters 1D
        self.a = config['Beta_Objective_func']['a']
        self.b = config['Beta_Objective_func']['b']
        self.l = config['Beta_Objective_func']['l']
    
    def normalize1Dpairwise(self, X):
        """
        Normalize vertical illuminance feature duels
        Inputs: 
            X (N x 2) : pairwise comparison data [X_prev, X_current]
        Outputs:
            X_norm : normalized pairwise comparison data
        """
        X_norm = 2./(self.Gridmax1 - self.Gridmin1)*(X - self.Gridmin1) - 1.
        
        return X_norm
    
    def normalize1D(self, X):
        """
        Normalize 1 D feature vector 
        Input:
            X (N x 1) : 1D feature vector
        Output: 
            X_norm : normalized 1D feature vector
        """
        
        return self.normalize1Dpairwise(X)
    
    def feat1_grid_duels(self, num_points):
        """
        Grid for duels : only vertical illuminance as a feature
        This grid is required for calculating preference probabilities
        Outputs:
            Xgrid, Xgridnorm, num_points (number of grid points)
        """
        # Normalization of features
        Grid = np.linspace(self.Gridmin1, self.Gridmax1, num_points)
        # Grid configuration 1D
        Xtt1, Xtt2 = np.meshgrid(Grid, Grid)
        Xtt1_ravel = np.ravel(Xtt1)
        Xtt2_ravel = np.ravel(Xtt2)
        # Grid points defined
        Xgrid = np.zeros(shape = (Xtt1_ravel.shape[0], 2))
        Xgrid[:,0] = Xtt1_ravel
        Xgrid[:,1] = Xtt2_ravel
        # Normalization of grid points
        Xgridnorm = self.normalize1Dpairwise(Xgrid)
        
        return Xgrid, Xgridnorm
    
    def feat1_nxt_point_grid(self, num_points):
        """
        Grid for selection of next datapoint:
        if feat = 1 then we will have a 1D grid so as to 
        calculate the posterior utility at those points
        """
        # Grid Points
        #vecforut = np.linspace(self.Gridmin1, self.Gridmax1, num_points)
        vecforut = np.arange(self.Gridmin1, self.Gridmax1, self.temp_diff)
        print vecforut
        vec_norm = self.normalize1D(vecforut)
        
        return vecforut[:,None], vec_norm[:,None]
    
    def response_gen1D(self, X):
        """
        Given duels matrix 1D features, generate response of synthetic occ
        Inputs:
            X (N x 2) : pairwise comparion data [X_prev, X_current]
        Outputs:
            Y (N x 1) : yi vector, yi = 1 if xcurrent is preferred, 0 otherwise
        """
        x1 = X[:,0]
        x2 = X[:,1]
        u1, u1_der = objfunc.beta_utility_gen(x1, self.l,
                                             self.a, self.b)
        u2, u2_der = objfunc.beta_utility_gen(x2, self.l,
                                             self.a, self.b)
        
        y_pr = gen_output(u1, u2)
        return y_pr
    
    def response_grad_gen1D(self, X):
        x1 = X[:,0]
        x2 = X[:,1]
        u1, u1_der = objfunc.beta_utility_gen(x1, self.l,
                                             self.a, self.b)
        u2, u2_der = objfunc.beta_utility_gen(x2, self.l,
                                             self.a, self.b)
        y_pr = np.zeros(u2_der.shape[0])
        y_pr[u2_der > 0] = 1.
        y_pr[u2_der < 0] = -1.
        return y_pr
        
    
    def duels_gen(self, num_feat, num_datapoints):
        """
        Generation of training duels and associated utilities
        """
        #n = 14 # number of datapoints
        #x1 = np.linspace(self.Gridmin1, self.Gridmax1, n)[:, None] # Operating temp.
        x1 = np.arange(self.Gridmin1, self.Gridmax1, self.temp_diff)[:,None]
        
         # sampling from indexes
        indexes = np.arange(x1.shape[0])
        
        ind_samp1 = np.random.choice(indexes, size = 1)
        x_samp1_01 = x1[ind_samp1]
        
        ind_samp2_all1 = np.random.choice(indexes, size = num_datapoints)
        x_samp21 = x1[ind_samp2_all1]
        x_samp1_n1 = x_samp21[:-1]
        x_samp11 = np.append(x_samp1_01, x_samp1_n1)[:,None]
    
        u1, u1_der = objfunc.beta_utility_gen(x_samp11, self.l,
                                      self.a, self.b)
        u2, u2_der = objfunc.beta_utility_gen(x_samp21, self.l,
                                      self.a, self.b) 

        return (x_samp11, x_samp21, u1, u2)
    
    def gradient_duels_gen(self, num_feat, num_datapoints):
        """
        Generation of training duels and associated utilities
        """
        #n = 14 # number of datapoints
        #x1 = np.linspace(self.Gridmin1, self.Gridmax1, n)[:, None] # Operating temp.
        x1 = np.arange(self.Gridmin1, self.Gridmax1, self.temp_diff)[:,None]
        
         # sampling from indexes
        indexes = np.arange(x1.shape[0])
        
        ind_samp1 = np.random.choice(indexes, size = 1)
        x_samp1_01 = x1[ind_samp1]
        
        ind_samp2_all1 = np.random.choice(indexes, size = num_datapoints)
        x_samp21 = x1[ind_samp2_all1]
        x_samp1_n1 = x_samp21[:-1]
        x_samp11 = np.append(x_samp1_01, x_samp1_n1)[:,None]
    
        u1, u1_der = objfunc.beta_utility_gen(x_samp11, self.l,
                                      self.a, self.b)
        u2, u2_der = objfunc.beta_utility_gen(x_samp21, self.l,
                                      self.a, self.b) 
        
        u2_der = u2_der[:,0] 
        y_der = np.zeros(u2_der.shape[0])
        y_der[u2_der > 0] = 1.
       
        y_der[u2_der < 0] = -1.

        return (x_samp11, x_samp21, y_der, u1, u2)
    
    def pairwise1D(self, num_datapoints, save_file_name, save_file = False):
        """
        Generate 1D pairwise preferences dataset
        One feature values
        """
        num_feat = 1
        (x_samp11,
         x_samp21, u1, u2) = self.duels_gen(num_feat, num_datapoints)
        y_pr = gen_output(u1, u2)
        #print x_samp11.shape[0]
        #print x_samp21.shape[0]
        X = np.hstack([x_samp11, x_samp21])
        if save_file: 
            np.savez(save_file_name, X = X, Y = y_pr)
        return X, y_pr
    
    def gradientpairwise1D(self, num_datapoints, save_file_name, save_file = False):
        """
        Generate 1D pairwise preferences dataset
        One feature values
        """
        num_feat = 1
        (x_samp11,
         x_samp21, y_der,
         u1, u2) = self.gradient_duels_gen(num_feat, num_datapoints)
        y_pr = gen_output(u1, u2)
        X = np.hstack([x_samp11, x_samp21])
        if save_file: 
            np.savez(save_file_name, X = X, Y = y_pr, Y_der = y_der)
        return X, y_pr, y_der
    
    
class ReachableStates(object):
    def __init__(self, Xprev):
        """
        Class which contains methods that output states which can be reached
        at any given moment. These states can be constrained/ unconstrained.
        Inputs:
        Xprev : Previous state value
        Outputs:
            Xgrid : states which can be reached from current state
            Xgridnorm : Normalized Xgrid
        if shape(Xprev) = (1,) ; shape(Xreachable) = 1 X R (R = num of reachable states)
        that means only 1D feature is used to define state
        if shape(Xprev) = (2,) ; shape(Xreachable) = 2 X R (R = num of reachable states)
        """
        self.Xprev = Xprev
        
    def rs1D(self, config_file):
        """
        For this method, the reachable state is not dependent on the previous state
        1D feature is used to define the state of the room
        """
        VD = ThermalPrefDataGen(config_file)
        Xr, Xrnorm = VD.feat1_nxt_point_grid(VD.numgrid1)
        print 'Previous state:'
        print self.Xprev
        
        return Xr, Xrnorm

    def reachable(self, config_file):
        """
        Reachable states
        """
        VD = ThermalPrefDataGen(config_file)
        # Read configuration file
        with open(config_file, 'r') as fd:
            config = json.loads(fd.read())
        temp_diff1 = config['Grid']['temp_diff1']
        temp_diff2 = config['Grid']['temp_diff2']
        grid_min = config['Grid']['min1']
        grid_max = config['Grid']['max1']
        reach_min = self.Xprev - temp_diff2
        reach_max = self.Xprev + temp_diff2
        if reach_min < grid_min:
            reach_min = grid_min
        if reach_max > grid_max:
            reach_max = grid_max
        Xr = np.arange(reach_min, reach_max, temp_diff1)[:,None]
        Xrnorm = VD.normalize1D(Xr)
        
        return Xr, Xrnorm
        
        
    def user_defined(self):
        """
        User defined reachable state function
        """
        print 'Define your reachable state function ...'
        
        return
    
    
if __name__ == '__main__':
    config_file = '../config_files/thermal_config_2.json'
    save_file_name1 = '../data/initial_duels/syn_2.npz'
    
    ThermalP = ThermalPrefDataGen(config_file)
    
    X1, y_pr1, y_der = ThermalP.gradientpairwise1D(1, save_file_name1, save_file = True)
    #X2, y_pr2 = ThermalP.pairwise2D(40, save_file_name2, save_file = True)
    
       
    