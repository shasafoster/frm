# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm



def simHeston(n, s0, v0, mu, kappa, theta, sigma, rho, delta, no=None, method=0):
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
