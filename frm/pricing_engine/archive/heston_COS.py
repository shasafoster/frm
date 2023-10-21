#%%
"""
Created on Thu Nov 30 2018
Merton Model and convergence obtained with the COS method
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum
import scipy.optimize as optimize

# Set i= imaginary number
i = 1j


def CallPutOptionPriceCOSMthd(cf,cp,S0,r,tau,K,N,L):
    # cf - Characteristic function as a functon, in the book denoted by phi
    # cp - C for call and P for put
    # S0 - Initial stock price
    # r - Interest rate (constant)
    # tau - Time to maturity
    # K - List of strikes
    # N - Number of expansion terms
    # L - Size of truncation domain (typ.:L=8 or L=10)

    # Reshape K to become a column vector
    if K is not np.array:
        if np.isscalar(K):
            K = np.array([K]).reshape(1, 1)
        else:
            K = np.array(K).reshape(len(K), 1)

    x0 = np.log(S0 / K)

    # Truncation domain
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    
    # Summation from k = 0 to k=N-1
    k = np.linspace(0,N-1,N).reshape([N,1])
    u = k * np.pi / (b - a);
    # Determine coefficients for put prices
    H_k = CallPutCoefficients(cp,a,b,k)
    mat = np.exp(1j * np.outer((x0 - a) , u))
    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]
    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))

    return value


# Determine coefficients for put prices
def CallPutCoefficients(cp,a,b,k):
    if cp == 1:
        c = 0.0
        d = b
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k),1])
        else:
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k)
    elif cp == -1:
        c = a
        d = 0.0
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k = 2.0 / (b - a) * (- Chi_k + Psi_k)

    return H_k


def Chi_Psi(a,b,c,d,k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/ (b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    
    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0))
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d) - np.cos(k * np.pi  * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi * (d - a) / (b - a)) - k * np.pi / (b - a) * np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)
    
    value = {"chi":chi,"psi":psi }
    return value

# Black-Scholes call option price
def BS_Call_Option_Price(cp,S_0,K,sigma,tau,r):
    K = np.array(K).reshape([len(K),1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0))
    * tau) / float(sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if cp == 1:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif cp == -1:
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

# Implied volatility method
def ImpliedVolatility(cp,marketPrice,K,T,S_0,r):
    func = lambda sigma: np.power(BS_Call_Option_Price(cp,S_0,K,sigma,T,r) - marketPrice,1.0)
    impliedVol = optimize.newton(func, 0.7, tol=1e-5)
    return impliedVol

def ChFHestonModel(r,tau,kappa,gamma,vbar,v0,rho):
    i = complex(0.0,1.0)
    D1 = lambda u: np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u*u+i*u)*gamma*gamma)
    g = lambda u: (kappa-gamma*rho*i*u-D1(u))/(kappa-gamma*rho*i*u+D1(u))
    C = lambda u: (1.0-np.exp(-D1(u)*tau))/(gamma*gamma*(1.0-g(u)*D1(u)*tau)) *(kappa-gamma*rho*i*u-D1(u))
    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method
    A = lambda u: r * i*u *tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*rho*i*u-D1(u)) - 2*kappa*vbar/gamma/gamma*np.log((1-g(u)*np.exp(-D1(u)*tau))/(1-g(u)))
    # Characteristic function for the Heston model
    cf = lambda u: np.exp(A(u) + C(u)*v0)
    return cf



cp = 1
S0 = 100
r = 0.00
tau = 1

K = np.linspace(50,150)
K = np.array(K).reshape([len(K),1])

# COS method settings
L = 3

kappa = 1.5768
gamma = 0.5751
vbar = 0.0398
rho =-0.5711
v0 = 0.0175

# Compute ChF for the Heston model
cf = ChFHestonModel(r,tau,kappa,gamma,vbar,v0,rho)

# The COS method -- Reference value
N = 5000
valCOSRef = CallPutOptionPriceCOSMthd(cf, cp, S0, r, tau, K, N, L)

plt.figure(1)
plt.plot(K,valCOSRef,'-k')
plt.xlabel("strike, K")
plt.ylabel("Option Price")
plt.grid()

# For different numbers of integration terms

N = [32,64,96,128,160,400]
maxError = []
for n in N:
    valCOS = CallPutOptionPriceCOSMthd(cf, cp, S0, r, tau, K, n, L)
    errorMax = np.max(np.abs(valCOS - valCOSRef))
    maxError.append(errorMax)
    print('maximum error for n ={0} is {1}'.format(n,errorMax))
        
    
#%%


cp = 1
S0 = 1.2779
r = 0.0070075
tau = 2

kappa = 1.5
gamma = 0.017132031
vbar = 0.293475456
rho = 0.232453505
v0 = 0.014820628
K = 1.466967

cf = ChFHestonModel(r,tau,kappa,gamma,vbar,v0,rho)
P_COS = CallPutOptionPriceCOSMthd(cf, cp, S0, r, tau, K, n, L)    
print('PCOS',P_COS)
