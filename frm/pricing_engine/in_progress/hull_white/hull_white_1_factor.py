# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())
    
import numpy as np
from numba import jit, prange
import scipy
from scipy.stats import norm
import timeit
import time
import pandas as pd

from frm.frm.pricing_engine.monte_carlo_generic import generate_rand_nbs

import numpy.random as npr
from   scipy import integrate
import scipy.integrate as sciIntegr


import matplotlib.pyplot as plt



from frm.frm.pricing_engine.hull_white.Bond import *


class HullWhite(Bond):
    def __init__(self, ZCB, times, theta, kappa, sigma, r0=0.):
        Bond.__init__(self, theta, kappa, sigma, r0)
    
        self.dt     = 0.00001
        self._ZCB   = ZCB
        self._times = times
        
        self.ret = scipy.interpolate.splrep(self._times, -np.log(self._ZCB)/ self._times)                
                
    def ForwardRate(self, time):                
        r        = self.ZeroRate(time) # evaluate the zero rate 
        dr_dt    = scipy.interpolate.splev(time, self.ret, der=1) # der=1 gets the 1st derivative
         
        fwd_rate = r + time * dr_dt                    
        return fwd_rate
    
    def ZeroRate(self, time):            
        return scipy.interpolate.splev(time, self.ret)
    
    def PrepareThetas(self):
        Timeline        = np.linspace(Maturities[0], Maturities[-1],100)
        Thetarates      = [self.Thetat(t) for t in Timeline]
        self.ThetaratesForTimeline = np.array(Thetarates)
        self.Thetarates = scipy.interpolate.splrep(Timeline, Thetarates) 
        
    def A(self, t, T):                    
               
        def integrand1(time):
            theta = scipy.interpolate.splev(time, self.Thetarates)
            return theta * self.B(time, T)
            
        def integrand2(time):
            return (self.B(time, T)**2)
        
        val1 = integrate.quad(integrand1, t, T)[0]
        val2 = integrate.quad(integrand2, t, T)[0]
        return -val1 + 0.5*(self.sigma**2)*val2
        
    def Yield(self, t, T, rate):
        res    = -self.A(t, T)/(T-t) + (1/self.kappa)*(1-np.exp(-self.kappa*(T-t))) * 1/(T-t)*rate
        return res
        
    def Thetat(self, time):  
        
        # Calculate dF/dt by numerical integration
        up       = np.max(time + self.dt,0)
        down     = np.max(time - self.dt,0)
        
        df_t = self.ForwardRate(time)
        df_t_up = self.ForwardRate(up)  
        df_t_down = self.ForwardRate(down)

        df_dt    = (df_t_up - df_t_down)/(2*self.dt)
        
        return df_dt + self.kappa * df_t + (self.sigma**2)/(self.kappa*2)*(1-np.exp(-2*self.kappa*time))
        
    def ZCB_Forward_Integral(self,t, T):                                         
        val = integrate.quad(self.ForwardRate, t, T)[0]
        return np.exp(-val)    
    
    def B(self, t, T):
        return (1/self.kappa)*(1-np.exp(-self.kappa*(T- t)))
    
    def Exact_zcb(self, t, T):
        B = self.B(t, T)
        A = self.A(t, T)
        return np.exp(A-self.r0*B)
    
    def Euler(self, M, I, tau):
        # I is the number of simulation
        # M is the number of time steps until maturity
        # tau is the maturity        
        dt = tau / float(M)

        M  = int(M)
        I  = int(I)
        xh = np.zeros((M + 1, I))
        self.rates = np.zeros_like(xh)
        self.times = np.linspace(0, tau, num = M + 1)
        
        xh[0]     = self.r0
        for t in range(1, M + 1):
            xh[t] = xh[t - 1] + (self.Thetat(self.times[t-1]) - self.kappa*xh[t - 1]) * dt + \
                  self.sigma * np.sqrt(dt) * npr.standard_normal(I)
        self.rates = xh
        
    def ExpectedRate(self,t, T):
        # this expectation if always from time t = 0
        def integrand(time):
            theta = scipy.interpolate.splev(time, self.Thetarates)
            
            return theta * np.exp(-self.kappa*(T-time))
                    
        val = self.r0*np.exp(-(T-t)*self.kappa) + integrate.quad(integrand, t, T)[0]
        return val
    
    
#%%

dataframe = pd.read_csv('./frm/frm/pricing_engine/hull_white/Strips.csv', index_col=None)
print (dataframe.shape)
dataframe.head(10)


Maturities = np.asarray(dataframe.years)
Prices     = np.asarray(dataframe.discount_factor)
rate0      = -np.log(Prices[0])/Maturities[0]
print(rate0)

r0    = rate0   # current level of rates
kappa = 0.19    # speed of convergence - Not required in Ho Lee Model
theta = ""      # long term rate       - Not required in Ho Lee Model
sigma = 0.0196  # vol    

# create an instance of the object HullWhite and calculate theta(t)
hullwhite = HullWhite(Prices,Maturities, theta, kappa, sigma,rate0)
hullwhite.PrepareThetas()

theta = hullwhite.ThetaratesForTimeline

#%%

CCZR = -1 * np.log(Prices) / Maturities

spline_def = scipy.interpolate.splrep(Maturities, CCZR)   

Timeline        = np.linspace(Maturities[0], Maturities[-1],100)

CCZR_0 = scipy.interpolate.splev(Timeline, spline_def, der=0) # der=1 gets the 1st derivative
CCZR_1 = scipy.interpolate.splev(Timeline, spline_def, der=1) # der=1 gets the 1st derivative




#%%

print(-np.log(hullwhite.Exact_zcb(0,1)))
print(hullwhite.Yield(0,1,rate0))
print(-np.log(hullwhite.ZCB_Forward_Integral(0,1)))
print(dataframe.zero_rates[9])
print(dataframe.zero_rates[8])


#%%

# The forward rate for the period t = 1, t= 2 is 
print(-np.log(hullwhite.ZCB_Forward_Integral(1,2)))

# we can also use Hull and White ZCB formula to calculate the forward price for the period t = 1, t= 2 
print(hullwhite.Exact_zcb(0,2)/hullwhite.Exact_zcb(0,1))
print(hullwhite.ZCB_Forward_Integral(1,2))

Timeline  = np.linspace(Maturities[0], Maturities[-1],100)
Fwdrates  = [hullwhite.ForwardRate(t) for t in Timeline]
zerorates = [hullwhite.ZeroRate(t) for t in Timeline]
Thetarates= [hullwhite.Thetat(t) for t in Timeline]
hullwhitePrices=[hullwhite.Exact_zcb(0, t) for t in Timeline]

FwdratesNp = np.array(Fwdrates)
zeroratesNp = np.array(zerorates)
ThetaratesNp = np.array(Thetarates)
hullwhitePricesp = np.array(hullwhitePrices)


#%%

plt.figure(figsize=(16,4))
plt.subplot(121)
plt.plot(Timeline, Fwdrates, label ='calculated Forward rates')
plt.plot(Timeline, zerorates, label='calculated Spot rates')
plt.plot(Maturities, -np.log(Prices)/Maturities, marker='.', label='Original Spot rates')
plt.xlabel(r'Maturity $T$ ')
plt.title(r'continously compounded $r(t,T)$ and $f(t,T)$')
plt.grid(True)
plt.legend()
plt.ylabel('spot and forward rates')

plt.subplot(122)
plt.plot(Timeline, Thetarates, marker='.',label =r'$\theta(t)$')
plt.xlabel(r'Maturity $T$') 
plt.title(r'$function$ $\theta(t)$')
plt.grid(True)
plt.ylabel(r'$\theta(t)$')
plt.show()


#%%

plt.figure(figsize=(16,4))

plt.plot(Maturities, Prices, marker='*', label = "Original Prices")
plt.plot(Timeline, hullwhitePrices, label ='Hull White Prices')

plt.xlabel(r'Maturity $T$ ')
plt.title(r'Original Prices vs Hull White Model Recovered Prices')
plt.grid(True)
plt.legend()
plt.ylabel('Prices')

plt.show()


#%%

tau = 5
I   = 10000       # no. of simulations
M   = tau * 252   # trading day per annum

# So we always use the same random numbers
npr.seed(1500)

# run the sde
%time hullwhite.Euler(M, I, tau)

vals = hullwhite.StochasticPrice(hullwhite.rates, hullwhite.times)

print("the price is:      ", vals[0])
print("the price +2sd is :", vals[1])
print("the price -2sd is :", vals[2])
print("the Analytic price is :", hullwhite.Exact_zcb(0, 5))


#%%

# So we always use the same random numbers
npr.seed(1500)

size   = (Prices.shape[0])
Result = np.zeros((size, 6))
for i, j in enumerate(Maturities):
    tau = j
    M   = tau * 252
    hullwhite.Euler(M, I, tau)
    vals = hullwhite.StochasticPrice(hullwhite.rates, hullwhite.times)
    Result[i,0] = np.round(j,5)
    Result[i,1] = np.round(vals[0],4)
    Result[i,2] = np.round(vals[1],4)
    Result[i,3] = np.round(Prices[i],4)
    Result[i,4] = np.round(hullwhite.Exact_zcb(0,j),4)
    Result[i,5] = np.round(vals[2],4)
    
Result = pd.DataFrame(Result)
Result.columns = ['Maturity','MC Price','MC Price+2CD', 'Original Price', 'hullwhite Price', 'MC Price-2CD',]
Result.tail(10)


#%%

# So we always use the same random numbers
npr.seed(1500)

I   = 10000     # no. of simulations
T_O = 1 # 1 year
T_M = 5 # 5 year
M   = T_O * 252 # trading day per annum
    
BondValue = ZCBValue(0) # an instance of class ZCBValue, to model the payoff value at year 1    
vals = hullwhite.FutureZCB(M,I, T_O, T_M, BondValue)

#%%

print( "----------------------------------------------------------------")
print("the MC price of a ZCB maturing in",T_M, "years is:", np.round(vals[0],4))
print("The ZCB price from Formulae is               ", np.round(hullwhite.Exact_zcb(0, 5),4))

print( "----------------------------------------------------------------")
print("the MC price +2sd is :", vals[1])
print("the MC price -2sd is :", vals[2])
print( "----------------------------------------------------------------")

print("the MC price of a ZCB maturing in",T_O, "year is:", np.round(vals[4],4))
print("The ZCB price from Formulae is              ", np.round(hullwhite.Exact_zcb(0, 1),4))

print( "----------------------------------------------------------------")
print("The MC Expected short term rate in",T_O,"year is ", np.round(vals[5],4))
print("The Expected short term rate in",T_O,"year is ", np.round(hullwhite.ExpectedRate(0,1),4))

print( "----------------------------------------------------------------")
print("the Price in",T_O,"year of a ZCB price maturing in",T_M,"years is", np.round(vals[3],4))
print("The fwd price from Formulae is                           ", np.round(hullwhite.Exact_zcb(0, 5)/hullwhite.Exact_zcb(0, 1),4))


#%%

# So we always use the same random numbers
npr.seed(1500)

I   = 10000   # no. of simulations
OptionStrike = 0.8
BondValue    = ZCBOption(OptionStrike) # an instance of class ZCBOption, to model the payoff value at year 1   


#%%

%time vals = hullwhite.FutureZCB(M,I, T_O, T_M, BondValue)

#%%

print( "------------------------------------------------------")
print("the MC price of an one year Option on a ZCB maturing in:",T_M, "years is:", np.round(vals[0],4))

print( "------------------------------------------------------")
print("the MC price +1.96sd is :", vals[1])
print("the MC price -1.96sd is :", vals[2])

print("The Analytic hullwhite Price is ", 0.024040) # Hull White Analytic Option model is not included. Price was taken from
# Fixed Income Securities, Pietro Veronesi, chapter 19, example 19.4, page 662 


#%%