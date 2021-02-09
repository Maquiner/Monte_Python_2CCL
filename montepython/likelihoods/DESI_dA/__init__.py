import os
import pyccl as ccl
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts

class DESI_dA(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)
        
        # create redshift array 
        self.z_arr = np.arange(0.15, 1.85+0.1, 0.1)
        # read data file 
        self.data = np.loadtxt(os.path.join(self.data_directory, self.data_file))
        # read covariance matrix
        self.cov_data = np.loadtxt(os.path.join(self.data_directory, self.cov_file))
        
        # Initialize classy instance 
        self.classy = ccl.boltzmann.classy.Class()
        #self.classy.compute()
        
    def loglkl(self, cosmo, data):
        
        # pass params from ccl to classy
        sampled_params = {'h': data.mcmc_parameters['h']['current'], 
                'Omega_cdm': data.mcmc_parameters['Omega_c']['current'],
                'Omega_b': data.mcmc_parameters['Omega_b']['current'],
                #'sigma8': 2.43e-9 * (data.mcmc_parameters['sigma8']['current'] / 0.87659)**2,
                'n_s': data.mcmc_parameters['n_s']['current'],
                'Omega_Lambda': 0,
                'w0_fld': data.mcmc_parameters['w_0']['current'],
                'wa_fld': data.mcmc_parameters['w_a']['current']}
        self.classy.set(sampled_params) 
        self.classy.compute()

        # calculate theory
        dA_arr = np.array([])
        H_arr = np.array([])
        for z in self.z_arr: 
            dA = self.classy.angular_distance(z)/self.classy.rs_drag()
            H = self.classy.Hubble(z)*self.classy.rs_drag()
            dA_arr = np.append(dA_arr, dA)
            H_arr = np.append(H_arr, H)  
        # Order is important dA before H
        theory = np.append(dA_arr, H_arr)
        inv_cov_data = np.linalg.inv(self.cov_data)
        chi2 = (self.data-theory).T.dot( inv_cov_data).dot(self.data-theory) 
        # return ln(L)
        loglkl = - 0.5 * chi2

        return loglkl 
    
