# -*- coding: utf-8 -*-


import os
import pathlib

if __name__ == "__main__":
    os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve()) # path to ./frm/
    print(__file__.split('\\')[-1], os.getcwd())
    
from pricing_engine.garman_kohlhagen import garman_kohlhagen
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize, root
import time

def simHeston(n, s0, v0, mu, kappa, theta, sigma, rho, delta, no=None, method=0):
    
    
    # https://www.degruyter.com/document/doi/10.1515/math-2017-0058/html?lang=en
    # E + M scheme or QE scheme appear to be the quickest
    
    if no is None:
        no = np.random.normal(0, 1, (int(n/delta), 2))

    if len(no[:, 0]) != int(n/delta):
        raise ValueError('Size of no is inappropriate. Length of no[:, 0] should be equal to n/delta.')

    t = np.arange(0, np.ceil(n/delta)+1)
    sizet = len(t)
    x = np.zeros((sizet, 2))
    C = np.array([1, rho, rho, 1]).reshape(2, 2)
    u = normalcorr(C, no) * np.sqrt(delta)

    if method == 0:
        x[0, :] = [np.log(s0), v0]
        phiC = 1.5

        for i in range(1, sizet):
            m = theta + (x[i-1, 1] - theta) * np.exp(-kappa * delta)
            s2 = (x[i-1, 1] * sigma ** 2 * np.exp(-kappa * delta) / kappa * (1 - np.exp(-kappa * delta))
                  + theta * sigma ** 2 / (2 * kappa) * (1 - np.exp(-kappa * delta)) ** 2)
            phi = s2 / m ** 2
            gamma1, gamma2 = 0.5, 0.5
            K0 = -rho * kappa * theta / sigma * delta
            K1 = gamma1 * delta * (kappa * rho / sigma - 0.5) - rho / sigma
            K2 = gamma2 * delta * (kappa * rho / sigma - 0.5) + rho / sigma
            K3 = gamma1 * delta * (1 - rho ** 2)
            K4 = gamma2 * delta * (1 - rho ** 2)

            if phi <= phiC:
                b2 = 2 / phi - 1 + np.sqrt(2 / phi) * np.sqrt(2 / phi - 1)
                a = m / (1 + b2)
                x[i, 1] = a * (np.sqrt(b2) + no[i-1, 1]) ** 2
            else:
                p = (phi - 1) / (phi + 1)
                beta = (1 - p) / m

                if 0 <= norm.cdf(no[i-1, 1], 0, 1) <= p:
                    x[i, 1] = 0
                elif p < norm.cdf(no[i-1, 1], 0, 1) <= 1:
                    x[i, 1] = 1 / beta * np.log((1 - p) / (1 - norm.cdf(no[i-1, 1], 0, 1)))

            x[i, 0] = (x[i-1, 0] + mu * delta + K0 + K1 * x[i-1, 1] + K2 * x[i, 1]
                       + np.sqrt(K3 * x[i-1, 1] + K4 * x[i, 1]) * no[i-1, 0])

        x[:, 0] = np.exp(x[:, 0])
    else:
        x[0, :] = [s0, v0]

        for i in range(1, sizet):
            if method == 1:
                x[i-1, 1] = max(x[i-1, 1], 0)
            else:
                x[i-1, 1] = max(x[i-1, 1], -x[i-1, 1])

            x[i, 1] = (x[i-1, 1] + kappa * (theta - x[i-1, 1]) * delta
                       + sigma * np.sqrt(x[i-1, 1]) * u[i-1, 1])
            x[i, 0] = x[i-1, 0] + x[i-1, 0] * (mu * delta + np.sqrt(x[i-1, 1]) * u[i-1, 0])

    return x


def normalcorr(C, no):
    M = np.linalg.cholesky(C).T
    return (M @ no.T).T


def simGBM(n, x0, mu, sigma, delta, no=None, method=0):
    if no is None:
        no = np.random.normal(0, 1, int(np.ceil(n/delta)))
    else:
        no = no.flatten()
        if len(no) != int(np.ceil(n/delta)):
            raise ValueError('Error: length(no) <> n/delta')

    if method == 1: # Euler scheme
        x = x0 * np.cumprod(1 + mu * delta + sigma * delta**0.5 * no)
    elif method == 2: # Milstein scheme
        x = x0 * np.cumprod(1 + mu * delta + sigma * delta**0.5 * no +
                           0.5 * sigma**2 * delta * (no**2 - 1))
    elif method == 3: # 2nd order Milstein scheme
        x = x0 * np.cumprod(1 + mu * delta + sigma * delta**0.5 * no +
                           0.5 * sigma**2 * delta * (no**2 - 1) +
                           mu * sigma * no * (delta**1.5) + 0.5 * (mu**2) * (delta**2))
    else: # Direct integration
        x = x0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * delta + sigma * delta**0.5 * no))

    # Add starting value
    x = np.insert(x, 0, x0)
    return x


def pdfHestonInt(x, theta, kappa, sigma, rho, t, px):
    Gamma = kappa + complex(0,1) * rho * sigma * px
    Gamma2 = Gamma * Gamma
    Omega = np.sqrt(Gamma2 + sigma ** 2 * (px ** 2 - complex(0,1) * px))
    Omega2 = Omega * Omega
    Ft1 = kappa * theta / (sigma ** 2) * Gamma * t

    ncoeff = Omega2 - Gamma2 + 2 * kappa * Gamma
    coeff = ncoeff / (2 * kappa * Omega)
    Ot = Omega * t / 2
    Err = np.cosh(Ot)

    Ft2 = np.log(Err + coeff * np.sinh(Ot))
    Ft = Ft1 - 2 * kappa * theta / (sigma ** 2) * Ft2
    Eksp = np.exp(Ft + complex(0,1) * x * px)
    return np.real(Eksp)

def pdfHeston(x, theta, kappa, sigma, rho, t, par=0):
    n = len(x)
    y = np.zeros(n)
    pxm = 100

    for i in range(n):
        y[i] = 1 / (2 * np.pi) * quad(lambda s: pdfHestonInt(x[i], theta, kappa, sigma, rho, t, s), -100, 100, epsrel=1e-8)[0]

    if par == 1:
        import matplotlib.pyplot as plt
        plt.plot(x, y)
        plt.show()

    return y


def HestonVanilla(cp, s, k, v0, vv, rd, rf, tau, kappa, theta, lambda_, rho):
    def HestonVanillaInt(phi, m):
        a = kappa * theta
        u = [0.5, -0.5]
        b = [kappa + lambda_ - rho * vv, kappa + lambda_]
        x = np.log(s)
        y = np.log(k)

        d = np.sqrt((complex(0,1) * rho * vv * phi - b[m-1]) ** 2 - vv ** 2 * (2 * u[m-1] * phi * complex(0,1) - phi ** 2))
        g = (b[m-1] - rho * vv * phi * complex(0,1) - d) / (b[m-1] - rho * vv * phi * complex(0,1) + d)
        D = (b[m-1] - rho * vv * phi * complex(0,1) - d) / (vv ** 2) * ((1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau)))
        C = (rd - rf) * phi * complex(0,1) * tau + a / (vv ** 2) * ((b[m-1] - rho * vv * phi * complex(0,1) - d) * tau - 2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))
        f = np.exp(C + D * v0 + complex(0,1) * phi * x)
        F = np.real(np.exp(-complex(0,1) * phi * y) * f / (complex(0,1) * phi))

        return F

    P1 = 0.5 + 1/np.pi * quad(lambda phi: HestonVanillaInt(phi, 1), 0, np.inf, epsrel=1e-8)[0]
    P2 = 0.5 + 1/np.pi * quad(lambda phi: HestonVanillaInt(phi, 2), 0, np.inf, epsrel=1e-8)[0]

    Pplus = (1 - cp) / 2 + cp * P1
    Pminus = (1 - cp) / 2 + cp * P2

    P = cp * (s * np.exp(-rf * tau) * Pplus - k * np.exp(-rd * tau) * Pminus)
    return P



def HestonVanillaFitSmile(delta, marketvols, spot, rd, rf, tau, cp):

    strikes = garman_kohlhagen(S=spot,σ=marketvols,r_f=rf,r_d=rd,tau=tau,cp=cp,Δ=delta,task='strike',Δ_convention='regular_spot_Δ')
    nostrikes = len(strikes)
    v0 = (marketvols[nostrikes // 2])**2
    kappa = 1.5
    initparam = [2 * np.sqrt(v0), 2 * v0, 0]

    def HestonVanillaSSE(param, cp, spot, strikes, v0, marketvols, rd, rf, tau, kappa):
        number_calls = 1
        print(number_calls)
        
        vv, theta, rho = param
        if vv < 0 or theta < 0 or abs(rho) > 1:
            return np.inf
        P = np.zeros(nostrikes)
        IV = np.zeros(nostrikes)
        for i in range(nostrikes):
            t1 = time.time()
            P[i] = HestonVanilla(cp, spot, strikes[i], v0, vv, rd, rf, tau, kappa, theta, 0, rho)
            t2 = time.time()
            IV[i] = ImplVolFX(P[i], spot, strikes[i], rd, rf, tau, cp)
            t3 = time.time()
            print('HestonVanilla', t2-t1)
            print('ImplVol', t3-t2)
        return np.sum((marketvols - IV)**2)

    def ImplVolFX(X, S, K, rd, rf, tau, cp):
        
        return root(lambda vol: (garman_kohlhagen(S=spot,σ=vol,r_f=rf,r_d=rd,tau=tau,cp=cp,K=K,Δ=delta,task='px') - X), 0.001).x[0]
        #return root(lambda vol: (GarmanKohlhagen(S, K, vol, rd, rf, tau, cp, 0) - X), 0.001).x[0]

    res = minimize(lambda param: HestonVanillaSSE(param, cp, spot, strikes, v0, marketvols, rd, rf, tau, kappa), initparam)
    vv, theta, rho = res.x
    P = np.zeros(nostrikes)
    IV = np.zeros(nostrikes)
    
    t4 = time.time()
    print(t4-t3, 'res = minimize(lambda param')    
    
    for i in range(nostrikes):
        #t1 = time.time()
        P[i] = HestonVanilla(cp, spot, strikes[i], v0, vv, rd, rf, tau, kappa, theta, 0, rho)
        #t2 = time.time()
        #print(t2-t1, 'HestonVanilla(cp, spot, strikes[i], v0, vv, rd, rf, tau, kappa, theta, 0, rho)')                 
        IV[i] = ImplVolFX(P[i], spot, strikes[i], rd, rf, tau, cp)
        #t3 = time.time()
        #print(t3-t2,'IV[i] = ImplVolFX(P[i], spot, strikes[i], rd, rf, tau, cp)') 
        
    t5 = time.time()
    print(t5-t4,'for loop')  
    
    SSE = HestonVanillaSSE(res.x, cp, spot, strikes, v0, marketvols, rd, rf, tau, kappa)
    return v0, vv, kappa, theta, rho, IV, SSE


