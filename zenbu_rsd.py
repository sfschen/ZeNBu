import numpy as np

import os

from scipy.special import hyp2f1, gamma
from scipy.interpolate import interp1d

from Utils.spherical_bessel_transform_fftw import SphericalBesselTransform
from Utils.qfuncfft import QFuncFFT
from Utils.loginterp import loginterp

class Zenbu_RSD:
    '''
    Class to calculate Zeldovich power spectra in redshift space.
    
    Based on velocileptors.
    
    The bias parameters are ordered in pktable as
    1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2.
    Note that these are the component spectra (b_i, b_j) and not the coefficient multiplying b_i b_j in the auto.
    
    '''

    def __init__(self, k, p, third_order = True, shear=True, one_loop=True,\
                 kIR = None, cutoff=10, jn=5, N = 2000, threads=None, extrap_min = -5, extrap_max = 3):

        self.N = N
        self.extrap_max = extrap_max
        self.extrap_min = extrap_min
        
        self.kIR = kIR
        self.cutoff = cutoff
        self.kint = np.logspace(extrap_min,extrap_max,self.N)
        self.qint = np.logspace(-extrap_max,-extrap_min,self.N)
        
        self.third_order = third_order
        self.shear = shear or third_order
        self.one_loop = one_loop
        
        self.k = k
        self.p = p
        self.pint = loginterp(k,p)(self.kint) * np.exp(-(self.kint/self.cutoff)**2)
        self.setup_powerspectrum()
        
        self.pktables = {}
        
        if self.third_order:
            self.num_power_components = 13
        elif self.shear:
            self.num_power_components = 11
        else:
            self.num_power_components = 7
        
        self.jn = jn
        
        if threads is None:
            self.threads = int( os.getenv("OMP_NUM_THREADS","1") )
        else:
            self.threads = threads

        self.sph = SphericalBesselTransform(self.qint, L=self.jn, ncol=self.num_power_components, threads=self.threads)
    
    def setup_powerspectrum(self):
    
        # This sets up terms up to one loop in the combination (symmetry factors) they appear in pk
    
        self.qf = QFuncFFT(self.kint, self.pint, qv=self.qint, oneloop=False, shear=True, third_order=False)

        # linear terms
        self.Xlin = self.qf.Xlin
        self.Ylin = self.qf.Ylin
        self.XYlin = self.Xlin + self.Ylin; self.sigma = self.XYlin[-1]
        self.yq = self.Ylin / self.qint

        self.Ulin = self.qf.Ulin
        self.corlin = self.qf.corlin
        
        # load shear functions
        self.Xs2 = self.qf.Xs2
        self.Ys2 = self.qf.Ys2; self.sigmas2 = (self.Xs2 + self.Ys2)[-1]
        self.V = self.qf.V
        self.zeta = self.qf.zeta
        self.chi = self.qf.chi
        self.Xs4 = self.qf.Xs4
        self.Ys4 = self.qf.Ys4

            
    #### Define RSD Kernels #######
    
    def setup_rsd_facs(self,f,nu,D=1,nmax=10):
    
        self.f = f
        self.nu = nu
        self.D = D
        self.Kfac = np.sqrt(1+f*(2+f)*nu**2); self.Kfac2 = self.Kfac**2
        self.s = f*nu*np.sqrt(1-nu**2)/self.Kfac
        self.c = np.sqrt(1-self.s**2); self.c2 = self.c**2; self.ic2 = 1/self.c2; self.c3 = self.c**3
        self.Bfac = -0.5 * self.Kfac2 * self.Ylin * self.D**2 # this times k is "B"
        
        # Define Anu, Bnu such that \hn \cdot \hq = Anu * mu + Bnu * sqrt(1-mu^2) cos(phi)
        self.Anu, self.Bnu = self.nu * (1 + f) / self.Kfac, np.sqrt(1-nu**2) / self.Kfac
        
        # Compute derivatives
        # Each is a function of f, nu times (kq)^(-n) for the nth derivative
        
        # and the hypergeometric functions
        self.hyp1 = np.zeros( (self.jn+nmax, self.jn+nmax))
        self.hyp2 = np.zeros( (self.jn+nmax, self.jn+nmax))
        self.fnms = np.zeros( (self.jn+nmax, self.jn+nmax))
        
        for n in range(self.jn+nmax):
            for m in range(self.jn+nmax):
                self.hyp1[n,m] = hyp2f1(0.5-n,-n,0.5-m-n,self.ic2)
                self.hyp2[n,m] = hyp2f1(1.5-n,-n,0.5-m-n,self.ic2)
                self.fnms[n,m] = gamma(m+n+0.5)/gamma(m+1)/gamma(n+0.5)/gamma(1-m+n)
        
        self.G0_l_ns = np.zeros( (self.jn,nmax) )
        self.dG0dA_l_ns = np.zeros( (self.jn,nmax) )
        self.d2G0dA2_l_ns = np.zeros( (self.jn,nmax) )
        self.dG0dC_l_ns = np.zeros( (self.jn,nmax) )
        self.d2G0dCdA_l_ns = np.zeros( (self.jn,nmax) )
        self.d2G0dC2_l_ns = np.zeros( (self.jn,nmax) )
        self.d3G0dA3_l_ns = np.zeros( (self.jn,nmax) )
        self.d3G0dCdA2_l_ns = np.zeros( (self.jn,nmax) )
        self.d4G0dA4_l_ns = np.zeros( (self.jn,nmax) )
        
        for ll in range(self.jn):
            for nn in range(nmax):
                self.G0_l_ns[ll,nn] = self._G0_l_n(ll+nn,ll)
                self.dG0dA_l_ns[ll,nn] = self._dG0dA_l_n(ll+nn,ll)
                self.d2G0dA2_l_ns[ll,nn] = self._d2G0dA2_l_n(ll+nn,ll)
                
                # One loop terms
                self.dG0dC_l_ns[ll,nn] = self._dG0dC_l_n(ll+nn,ll)
                self.d2G0dCdA_l_ns[ll,nn] = self._d2G0dCdA_l_n(ll+nn,ll)
                self.d2G0dC2_l_ns[ll,nn] = self._d2G0dC2_l_n(ll+nn,ll)
                self.d3G0dA3_l_ns[ll,nn] = self._d3G0dA3_l_n(ll+nn,ll)
                self.d3G0dCdA2_l_ns[ll,nn] = self._d3G0dCdA2_l_n(ll+nn,ll)
                self.d4G0dA4_l_ns[ll,nn] = self._d4G0dA4_l_n(ll+nn,ll)
                
        # Also precompute the (BA^2/rho^2) factor
        self.powerfacs = np.array([ (self.Bfac /self.ic2)**n for n in range(self.jn + nmax) ]) # does not include factor of k^2n
        

        
    
    def _G0_l_n(self,n,m):
        x = self.ic2

        return  self.fnms[n,m] * self.hyp1[n,m]
    
    
    def _dG0dA_l_n(self,n,m):
        # Note that in the derivatives we omit factors of (kq)^n left in comments for speedier vector evaluation later
    
        x = self.ic2

        ret = self.s * (-self.hyp1[n,m] + (1-2*n)*self.hyp2[n,m])
        ret *= - self.s
        
        return self.fnms[n,m] * ret # / (k*self.qint)
    
    def _d2G0dA2_l_n(self,n,m):
        x = self.ic2
        
        ret = (1-1./x) * ( (2*m-1-4*n*(m+1))*self.hyp1[n,m] \
                                                +(1-4*n**2+m*(4*n-2))*self.hyp2[n,m] )
        return self.fnms[n,m] * ret #/(k*self.qint)**2
        
    def _dG0dC_l_n(self,n,m):
        x = self.ic2

        ret = self.s * (-self.hyp1[n,m] + (1-2*n)*self.hyp2[n,m])
        
        return self.fnms[n,m] * ret # / (k*self.qint)
        
    def _d2G0dCdA_l_n(self,n,m):
        x = self.ic2
        
        ret  = - ( 2*(m - 2*n*(1+m))*self.c**2 + self.s**2 ) * self.hyp1[n,m]
        ret += (1-2*n) * ( 2*(m-n)*self.c**2 + self.s**2 ) * self.hyp2[n,m]
        
        ret *= self.s / self.c
        
        return self.fnms[n,m] * ret # /(k*self.qint)**2
        
    def _d2G0dC2_l_n(self,n,m):
        x = self.ic2

        ret  = ( (1+2*m-4*n*(1+m))*self.c**2 + 2*self.s**2 ) * self.hyp1[n,m]
        ret += -(1-2*n) * ( (1+2*m-2*n)*self.c**2 + 2*self.s**2 ) * self.hyp2[n,m]
                
        return self.fnms[n,m] * ret # / (k*self.qint)**2
        
    def _d3G0dA3_l_n(self,n,m):
        x = self.ic2
        
        coeff1A = 2*(1-m)*(1-2*m) + 8*(2-m)*(1+m)*n + 8*n**2*(1+m)
        coeff1C = - (1-2*m+4*n*(1+m))
        ret = (coeff1A * self.c**2 + coeff1C * self.s**2) * self.hyp1[n,m]
        
        coeff2A = -(1-2*n)*( 2*(1-2*m+2*n)*(1-m+n) )
        coeff2C = (1-2*n)*(1-2*m+4*n*(1+m))
        ret += (coeff2A * self.c**2 + coeff2C * self.s**2) * self.hyp2[n,m]

        ret *= (self.s**2/self.c)
        
        return self.fnms[n,m] * ret # / (k*self.qint)**3
        
    
    def _d4G0dA4_l_n(self,n,m):
        x = self.ic2
        
        coeff1A = -6 + 22*m - 24*m**2 + 8*m**3 \
                 + n*(-76 - 28*m + 32*m**2 - 16*m**3) \
                 + n**2 * (-56 - 24*m + 32*m**2 ) + n**3 * ( -16 - 16*m )
        coeff1C = 9 - 24*m + 12*m**2 + n * (56 + 24*m - 32*m**2) +\
                   n**2 * (32 + 48*m + 16*m**2)
        ret = (coeff1A * self.c**2 + coeff1C * self.s**2) * self.hyp1[n,m]
        
        coeff2A = 2*(-3+2*m-2*n)*(1-2*m+2*n)*(1-m+n)
        coeff2C = 9 - 24*m + 12*m**2 + n*(44 + 8*m - 16*m**2) + n**2 * (20 + 16*m)
        ret += -(1-2*n)*(coeff2A * self.c**2 + coeff2C * self.s**2) * self.hyp2[n,m]

        ret *= self.fnms[n,m] * self.s**2 # / (k*self.qint)**4
        
        return ret

        
    # dG/dA^2dC
    def _d3G0dCdA2_l_n(self,n,m):
        x = self.ic2
        
        coeff1 =  2 * (m-2*m**2-4*n*(1-m**2)-4*n**2*(1+m)) * self.c**2
        coeff1 += 3 * (1-2*m+4*n*(1+m)) * self.s**2
        
        coeff2 = 2 * (1-2*m+2*n)*(m-n) * self.c**2
        coeff2 += (3-6*m+8*n+4*m*n) * self.s**2
        coeff2 *= - (1-2*n)
        
        ret  = coeff1 * self.hyp1[n,m]
        ret += coeff2 * self.hyp2[n,m]
        
        ret *= self.s
        
        return self.fnms[n,m] * ret # / (k*self.qint)**3
    

    def _G0_l(self,l,k, nmax=10):
        
        summand =  (k**(2* (l+np.arange(nmax))) * self.G0_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0)
        
        
    def _dG0dA_l(self,l,k,nmax=10):
        
        summand =  (k**(2* (l+np.arange(nmax))) * self.dG0dA_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)
    
    
    def _d2G0dA2_l(self,l,k,nmax=10):

        
        summand =  (k**(2* (l+np.arange(nmax))) * self.d2G0dA2_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)**2
        
        
    def _d3G0dA3_l(self,l,k,nmax=10):

        summand =  (k**(2* (l+np.arange(nmax))) * self.d3G0dA3_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)**3
    
        
    def _d4G0dA4_l(self,l,k,nmax=10):

        summand =  (k**(2* (l+np.arange(nmax))) * self.d4G0dA4_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)**4
        
    
    ### Now define the actual integrals!

    def p_integrals(self, k, nmax=8):
        
        ksq = k**2
        Kfac = self.Kfac
        f = self.f
        nu = self.nu
        Anu, Bnu = self.Anu, self.Bnu
        
        K = k*self.Kfac; Ksq = K**2; Kcu = K**3; K4 = K**4
        Knfac = nu*(1+f)
        
        D2 = self.D**2; D4 = D2**2

        expon = np.exp(-0.5*Ksq * D2* (self.XYlin - self.sigma))
        exponm1 = np.expm1(-0.5*Ksq * D2* (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*Ksq * D2* self.sigma)
            
            
        A = k*self.qint*self.c
        C = k*self.qint*self.s
        
        
        G0s =  [self._G0_l(ii,k,nmax=nmax)    for ii in range(self.jn)] + [0] + [0] + [0] + [0]
        dGdAs =  [self._dG0dA_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0] + [0] + [0]
        d2GdA2s = [self._d2G0dA2_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0] + [0]
        d3GdA3s = [self._d3G0dA3_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0]
        d4GdA4s = [self._d4G0dA4_l(ii,k,nmax=nmax) for ii in range(self.jn) ]
                
        G01s = [-(dGdAs[ii] + 0.5*A*G0s[ii-1])   for ii in range(self.jn)]
        G02s = [-(d2GdA2s[ii] + A * dGdAs[ii-1] + 0.5*G0s[ii-1] + 0.25 * A**2 *G0s[ii-2]) for ii in range(self.jn)]
        G03s = [d3GdA3s[ii] + 1.5*A*d2GdA2s[ii-1] + 1.5*dGdAs[ii-1] \
                 + 0.75*A**2*dGdAs[ii-2] + 0.75*A*G0s[ii-2] + A**3/8.*G0s[ii-3] for ii in range(self.jn)]
        G04s = [d4GdA4s[ii] + 2*A*d3GdA3s[ii-1] + 3*d2GdA2s[ii-1] \
                + 1.5*A**2*d2GdA2s[ii-2] + 3*A*dGdAs[ii-2] + 0.75*G0s[ii-2]\
                + 0.5*A**3*dGdAs[ii-3] + 0.75*A**2*G0s[ii-3]\
                + A**4/16. * G0s[ii-4] for ii in range(self.jn)]


        ret = np.zeros(self.num_power_components)
            
        bias_integrands = np.zeros( (self.num_power_components,self.N)  )
                            
        for l in range(self.jn):
            
            mu0 = G0s[l]
            #nq1 = self.Anu * G01s[l] + self.Bnu * G10s[l]
            #mu_nq1 = self.Anu * G02s[l] + self.Bnu * G11s[l]
            #nq2 = self.Anu**2 * G02s[l] + 2 * self.Anu * self.Bnu * G11s[l] + self.Bnu**2 * self.Bnu**2 * G20s[l]
            mu1 = G01s[l]
            mu2 = G02s[l]
            mu3 = G03s[l]
            #mu2_nq1 = self.Anu * G03s[l] + self.Bnu * G12s[l]
            mu4 = G04s[l]
            

            
            #bias_integrands[6,:] = -0.5 * ksq * (self.Xs2 + mu2fac*self.Ys2) # (1,bs)
            #bias_integrands[7,:] = -k*self.V*mu1fac + 0.5*kcu*self.Ulin*(self.Xs2*mu1fac + self.Ys2*mu3fac) # (b1,bs)
            #bias_integrands[8,:] = self.chi - 2*ksq*self.Ulin*self.V*mu2fac \
            #                          + 0.5*ksq**2*self.Ulin**2*(self.Xs2*mu2fac + self.Ys2*mu4fac) # (b2,bs)
            #bias_integrands[9,:] = self.zeta - 4*ksq*(self.Xs4 + mu2fac*self.Ys4) \
            #                        + 0.25*k4 * (self.Xs2**2 + 2*self.Xs2*self.Ys2*mu2fac + self.Ys2**2*mu4fac) # (bs,bs)
            
            bias_integrands[0,:] = 1 * G0s[l]
                                                
            bias_integrands[1,:] = - K * self.Ulin * mu1
                                   
            bias_integrands[2,:] = self.corlin * mu0 - Ksq*self.Ulin**2*mu2
                                   
                                   
            bias_integrands[3,:] = - Ksq * self.Ulin**2 * mu2 # b2
            bias_integrands[4,:] = -2 * K * self.Ulin * self.corlin * mu1 + Kcu * self.Ulin**3 * mu3 # b1b2
            bias_integrands[5,:] = 2 * self.corlin**2 * mu0 - 4*Ksq*self.Ulin**2*self.corlin*mu2 + K4*self.Ulin**4*mu4 # b2sq
            
            bias_integrands[6,:] = - 0.5 * Ksq * (self.Xs2 * mu0 + self.Ys2 * mu2) # bs should be both minus
            bias_integrands[7,:] = -K*self.V * mu1 + 0.5*Kcu*self.Ulin*(self.Xs2*mu1 + self.Ys2*mu3) # b1bs
            bias_integrands[8,:] = self.chi*mu0 - 2*Ksq*self.Ulin*self.V*mu2 \
                                      + 0.5*Ksq**2*self.Ulin**2*(self.Xs2*mu2 + self.Ys2*mu4) # (b2,bs)
            bias_integrands[9,:] = self.zeta*mu0 - 4*Ksq*(self.Xs4*mu0 + mu2*self.Ys4) \
                                    + 0.25*K4 * (self.Xs2**2*mu0 + 2*self.Xs2*self.Ys2*mu2 + self.Ys2**2*mu4) # (bs,bs)
                

            bias_integrands[-1,:] = 1 * G0s[l] # za
                                   
            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon * (-2./k/self.qint)**l
                bias_integrands -= bias_integrands[:,-1][:,None]
            else:
                bias_integrands = bias_integrands * expon * (-2./k/self.qint)**l
                                                                
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            ret += interp1d(ktemps, bias_ffts)(k)

        return 4*suppress*np.pi*ret
        

    def make_ptable(self, f, nu, kv = None, kmin = 1e-2, kmax = 0.25, nk = 50,nmax=5):
    
        self.setup_rsd_facs(f,nu,nmax=nmax)
        
        if kv is None:
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            nk = len(kv)
            
        self.pktable = np.zeros([nk, self.num_power_components+1]) # one column for ks
        
        self.pktable[:, 0] = kv[:]
        for foo in range(nk):
            self.pktable[foo, 1:] = self.p_integrals(kv[foo],nmax=nmax)
        
        # store a copy in pktables dictionary
        self.pktables[nu] = np.array(self.pktable)
        



    def make_pltable(self,f, apar = 1, aperp = 1, ngauss = 3, kv = None, kmin = 1e-2, kmax = 0.25, nk = 50, nmax=8):
        '''
        Make a table of the monopole and quadrupole in k space.
        Uses gauss legendre integration.
            
        '''
        
        # since we are always symmetric in nu, can ignore negative values
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        #self.pknutable = np.zeros((len(nus),nk,self.num_power_components+3)) # counterterms have distinct nu structure
        # counterterms + stoch terms have distinct nu structure and have to be added here
        # e.g. k^2 mu^2 is not the same as k_obs^2 mu_obs^2!
        if kv is None:
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            nk = len(kv)
        self.pknutable = np.zeros((len(nus),nk,self.num_power_components+6))
        
        
        # To implement AP:
        # Calculate P(k,nu) at the true coordinates, given by
        # k_true = k_apfac * kobs
        # nu_true = nu * a_perp/a_par/fac
        # Note that the integration grid on the other hand is never observed
        
        for ii, nu in enumerate(nus_calc):
        
            fac = np.sqrt(1 + nu**2 * ((aperp/apar)**2-1))
            k_apfac = fac / aperp
            nu_true = nu * aperp/apar/fac
            vol_fac = apar * aperp**2
        
            self.setup_rsd_facs(f,nu_true)
            
            for jj, k in enumerate(kv):
                ktrue = k_apfac * k
                pterms = self.p_integrals(ktrue,nmax=nmax)
                
                #self.pknutable[ii,jj,:-4] = pterms[:-1]
                self.pknutable[ii,jj,:-7] = pterms[:-1]
                
                # counterterms
                
                #self.pknutable[ii,jj,-4] = ktrue**2 * pterms[-1]
                #self.pknutable[ii,jj,-3] = ktrue**2 * nu_true**2 * pterms[-1]
                #self.pknutable[ii,jj,-2] = ktrue**2 * nu_true**4 * pterms[-1]
                #self.pknutable[ii,jj,-1] = ktrue**2 * nu_true**6 * pterms[-1]
                
                self.pknutable[ii,jj,-7] = ktrue**2 * pterms[-1]
                self.pknutable[ii,jj,-6] = ktrue**2 * nu_true**2 * pterms[-1]
                self.pknutable[ii,jj,-5] = ktrue**2 * nu_true**4 * pterms[-1]
                self.pknutable[ii,jj,-4] = ktrue**2 * nu_true**6 * pterms[-1]
                
                # stochastic terms
                self.pknutable[ii,jj,-3] = 1
                self.pknutable[ii,jj,-2] = ktrue**2 * nu_true**2
                self.pknutable[ii,jj,-1] = ktrue**4 * nu_true**4
        
        self.pknutable[ngauss:,:,:] = np.flip(self.pknutable[0:ngauss],axis=0)
        
        self.kv = kv
        self.p0ktable = 0.5 * np.sum((ws*L0)[:,None,None]*self.pknutable,axis=0) / vol_fac
        self.p2ktable = 2.5 * np.sum((ws*L2)[:,None,None]*self.pknutable,axis=0) / vol_fac
        self.p4ktable = 4.5 * np.sum((ws*L4)[:,None,None]*self.pknutable,axis=0) / vol_fac
        
        return 0

