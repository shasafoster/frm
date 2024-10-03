# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))






    # def price_cap_floor(self, d1, d2, K, σ):
    #
    #     T = self.daycounter(d1,d2)
    #     F = self.fwd_crv.forward_rate(d1,d2)
    #
    #     d1 = (np.log(F/K) + (0.5 * σ**2 * T)) / (σ*np.sqrt(T))
    #     d2 = d1 - σ*np.sqrt(T)
    #
    #     bought_cap = F * norm.cdf(d1) - K * norm.cdf(d2)
    #     bought_put = K * norm.cdf(d2) - F * norm.cdf(d1)