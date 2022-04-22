import numpy as np

import os

from scipy.interpolate import interp1d

from Utils.spherical_bessel_transform_fftw import SphericalBesselTransform
from Utils.qfuncfft import QFuncFFT
from Utils.loginterp import loginterp

class Zenbu:
    '''
    Class to calculate power spectra up to one loop.
    
    Based on velocileptors
    
    https://github.com/sfschen/velocileptors/blob/master/LPT/cleft_fftw.py
    
    The bias parameters are ordered in pktable as
    1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2.
    Note that these are the component spectra (b_i, b_j) and not the coefficient multiplying b_i b_j in the auto.
    
    Can combine into a full one-loop real-space power spectrum using the function combine_bias_terms_pk.
    
    '''

    def __init__(self, k, p, cutoff=10, jn=5, N = 2000, threads=None, extrap_min = -5, extrap_max = 3, import_wisdom=False, wisdom_file='wisdom.npy'):

        
        self.N = N
        self.extrap_max = extrap_max
        self.extrap_min = extrap_min
        
        self.cutoff = cutoff
        self.kint = np.logspace(extrap_min,extrap_max,self.N)
        self.qint = np.logspace(-extrap_max,-extrap_min,self.N)
        
        self.update_power_spectrum(k,p)        
        self.pktable = None
        self.num_power_components = 11

        
        self.jn = jn
        
        if threads is None:
            self.threads = int( os.getenv("OMP_NUM_THREADS","1") )
        else:
            self.threads = threads
    
        self.import_wisdom = import_wisdom
        self.wisdom_file = wisdom_file
        self.sph = SphericalBesselTransform(self.qint, L=self.jn, ncol=self.num_power_components, threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)
        

    def update_power_spectrum(self, k, p):
        # Updates the power spectrum and various q functions. Can continually compute for new cosmologies without reloading FFTW
        self.k = k
        self.p = p
        self.pint = loginterp(k,p)(self.kint) * np.exp(-(self.kint/self.cutoff)**2)
        self.setup_powerspectrum()

    def setup_powerspectrum(self):
                
        self.qf = QFuncFFT(self.kint, self.pint, qv=self.qint, oneloop=False, shear=True, third_order=False)
        
        # linear terms
        self.Xlin = self.qf.Xlin
        self.Ylin = self.qf.Ylin
        
        self.XYlin = self.Xlin + self.Ylin; self.sigma = self.XYlin[-1]
        self.yq = self.Ylin / self.qint
        
        self.Ulin = self.qf.Ulin
        self.corlin = self.qf.corlin
    
        self.Xs2 = self.qf.Xs2
        self.Ys2 = self.qf.Ys2; self.sigmas2 = (self.Xs2 + self.Ys2)[-1]
        self.V = self.qf.V
        self.zeta = self.qf.zeta
        self.chi = self.qf.chi


    def p_integrals(self, k):
        '''
        Compute P(k) for a single k as a vector of all bias contributions.
        
        '''
        ksq = k**2; kcu = k**3; k4 = k**4
        expon = np.exp(-0.5*ksq * (self.XYlin - self.sigma))
        exponm1 = np.expm1(-0.5*ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5 * ksq *self.sigma)
        
        ret = np.zeros(self.num_power_components)
        
        bias_integrands = np.zeros( (self.num_power_components,self.N)  )
        
        for l in range(self.jn):
            # l-dep functions
            mu1fac = (l>0)/(k * self.yq)
            mu2fac = 1. - 2.*l/ksq/self.Ylin
            mu3fac = (1. - 2.*(l-1)/ksq/self.Ylin) * mu1fac # mu3 terms start at j1 so l -> l-1
            mu4fac = 1 - 4*l/ksq/self.Ylin + 4*l*(l-1)/(ksq*self.Ylin)**2
            
            bias_integrands[0,:] = 1 # (1,1)
            bias_integrands[1,:] = - k * self.Ulin * mu1fac # (1, b1)
            bias_integrands[2,:] = self.corlin - ksq*mu2fac*self.Ulin**2 # (b1, b1)
            bias_integrands[3,:] = - ksq * mu2fac * self.Ulin**2 # (1,b2)
            bias_integrands[4,:] = -2 * k * self.Ulin * self.corlin * mu1fac + kcu * self.Ulin**3  * mu3fac # (b1,b2)
            bias_integrands[5,:] = 2 * self.corlin**2 - 4*ksq*self.Ulin**2*self.corlin*mu2fac \
                                       + ksq**2*self.Ulin**4*mu4fac # (b2,b2)
            
            bias_integrands[6,:] = -0.5 * ksq * (self.Xs2 + mu2fac*self.Ys2) # (1,bs)
            bias_integrands[7,:] = -k*self.V*mu1fac + 0.5*kcu*self.Ulin*(self.Xs2*mu1fac + self.Ys2*mu2fac) # (b1,bs)
            bias_integrands[8,:] = 2*self.chi - 2*ksq*self.Ulin*self.V*mu2fac \
                                      + 0.5*ksq**2*self.Ulin**2*(self.Xs2*mu2fac + self.Ys2*mu4fac) # (b2,bs)
            bias_integrands[9,:] = self.zeta + 0.25*ksq**4 * (self.Xs2**2 + 2*self.Xs2*self.Ys2*mu2fac + self.Ys2**2*mu4fac)# (bs,bs)

            bias_integrands[-1,:] = 1 # this is the counterterm, minus a factor of k2


            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
            
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)
    

        return 4*suppress*np.pi*ret

    def make_ptable(self, kmin = 1e-3, kmax = 3, nk = 100):
        '''
        Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        '''
        self.pktable = np.zeros([nk, self.num_power_components+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable[:, 0] = kv[:]
        for foo in range(nk):
            self.pktable[foo, 1:] = self.p_integrals(kv[foo])

    def combine_bias_terms_pk(self, b1, b2, bs, b3, alpha, sn):
        '''
        Combine all the bias terms into one power spectrum,
        where alpha is the counterterm and sn the shot noise/stochastic contribution.
        
        Three options, for
        
        (1) Full one-loop bias expansion (third order bias)
        (2) only quadratic bias, including shear
        (3) only density bias
        
        If (2) or (3), i.e. the class is set such that shear=False or third_order=False then the bs
        and b3 parameters are not used.
        
        '''
        arr = self.pktable
        
        
        bias_monomials = np.array([1, 2*b1, b1**2, 2*b2, 2*b1*b2, b2**2, 2*bs, 2*b1*bs, 2*b2*bs, bs**2])

        kv = arr[:,0]; za = arr[:,-1]
        pktemp = np.copy(arr)[:,1:-1]

        res = np.sum(pktemp * bias_monomials, axis =1) + alpha*kv**2 * za + sn

        return kv, res



    def combine_bias_terms_pk_crossmatter(self,b1,b2,bs,b3,alpha):
        """A helper function to return P_{gm}, which is a common use-case."""
        kv  = self.pktable[:,0]
        ret = self.pktable[:,1]+b1*self.pktable[:,2]+\
              b2*self.pktable[:,4]+bs*self.pktable[:,7]+\
              alpha*kv**2*self.pktable[:,13]
        return(kv,ret)
        #



    def export_wisdom(self, wisdom_file='./wisdom.npy'):
        self.sph.export_wisdom(wisdom_file=wisdom_file)
