import numpy as np
import pyccl as ccl
import data
#Emulators import
import sys
sys.path.insert(1, '/mnt/zfsusers/jaimerz/Monte_Python_2CCL/montepython/likelihoods/cl_cross_corr_2CCL')
import lin_pk_emul as emul


class CCL():
    """
    General class for the CCL object.

    """

    def __init__(self):

        self.state = 1

        # Set default parameters
        self.pars = {
            'h':       0.67,
            'Omega_c': 0.27,
            'Omega_b': 0.045,
            'sigma8': 0.840,
            'n_s':     0.96,
            'w0': -1.0,
            'wa':  0.0
        }

        # Start up emulator
        self.emulator = emul.LinPkEmulator(10, 200, 100, new=False)

    def get_cosmo_ccl(self):
        param_dict = dict({'transfer_function': 'boltzmann_class'},
                          **self.pars)
        try:
            param_dict.pop('output')
        except KeyError:
            pass
        cosmo_ccl = ccl.Cosmology(**param_dict)
        return cosmo_ccl

    def get_cosmo_gro(self):
        cosmo_gro = ccl.Cosmology(Omega_c=data.mcmc_parameters['Omega_c^{gro}']['current'],
                    Omega_b=data.mcmc_parameters['Omega_b^{gro}']['current'],
                    h=data.mcmc_parameters['h^{gro}']['current'], 
                    sigma8=data.mcmc_parameters['sigma8^{gro}']['current'], 
                    n_s=data.mcmc_parameters['n_s^{gro}']['current'])
        return cosmo_gro

    def get_sigma8(self):
        return ccl.sigma8(self.cosmo_ccl)

    def get_Omegam(self):
        Omm = self.pars['Omega_c'] + self.pars['Omega_b']
        return Omm

    def get_S8(self):
        S8 = self.get_sigma8()*(self.get_Omegam()/0.3)**(0.5)
        return S8
    
    def get_D_z(self, z):
        a = 1/(1+z)
        D_z = ccl.background.growth_rate(self.cosmo_ccl, a)
        return D_z 

    def get_growth_factor(self, a):
        return ccl.background.growth_factor(self.cosmo_ccl, a)

    def struct_cleanup(self):
        return

    def empty(self):
        return

    # Set up the dictionary
    def set(self, *pars_in, **kars):
        if ('A_s' in pars_in[0].keys()) and ('sigma8' in self.pars.keys()):
            self.pars.pop('sigma8')
        if len(pars_in) == 1:
            self.pars.update(dict(pars_in[0]))
        elif len(pars_in) != 0:
            raise RuntimeError("bad call")
        ### Check for parmeters of cl_cross_corr lkl
        if 'params_dir' in self.pars.keys():
            del[self.pars['params_dir']]
        if 'fiducial_cov' in self.pars.keys():
            del[self.pars['fiducial_cov']]
        #
        if 'tau_reio' in self.pars.keys():
            raise ValueError('CCL does not read tau_reio. Remove it.')
        # Translate w_0, w_a CLASS vars to CCL w0, wa
        if 'w_0' in self.pars.keys():
            self.pars['w0'] = self.pars.pop('w_0')
        if 'w_a' in self.pars.keys():
            self.pars['wa'] = self.pars.pop('w_a')

        self.pars.update(kars)
        return True

    def compute(self, level=[]):
        cosmo_gro = self.get_cosmo_gro()
        # Modified growth part
        if 'growth_param' in self.pars:
            k_arr, pk0 = self.emulator.get_emulated_Pk(cosmo_gro['Omega_c'], cosmo_gro['h'])
            pk0_k = scipy.interpolate.interp1d(k_arr, np.exp(pk0.real), kind='linear')
            pknew = ccl.Pk2D(pkfunc=self.pk2D(pk0_k), cosmo=cosmo_gro, is_logp=False)
            ccl.ccllib.cosmology_compute_linear_power(cosmo_gro.cosmo, pknew.psp, 0)
            # self.cosmo_ccl.compute_nonlin_power()
        ccl.sigma8(self.cosmo_ccl)  # David's suggestion
        return

    def get_current_derived_parameters(self, names):

        derived = {}
        for name in names:
            if name == 'sigma_8':
                value = self.get_sigma8()
            elif name == 'Omega_m':
                value = self.get_Omegam()
            elif name == 'S_8':
                value = self.get_S8()
            elif 'D_z' in name:
                z = float(name[3:])
                value = self.get_D_z(z)
            else:
                msg = "%s was not recognized as a derived parameter" % name
                raise RuntimeError(msg)
            derived[name] = value

        return derived

    def pk2D(self, pk):
        def pknew(k, a):
            return ccl.background.growth_factor(cosmo, a) ** 2 * pk(k)
        return pknew
