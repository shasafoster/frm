# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import itertools
import os.path

# Import both Lognormal and Normal SABR model classes
from pysabr import Hagan2002LognormalSABR
from pysabr import Hagan2002NormalSABR
from pysabr.helpers import year_frac_from_maturity_label


df = pd.read_csv('./tests/pricing_engine/sabr/vols.csv')
df.set_index(['Type', 'Option_expiry'], inplace=True)
df.sort_index(inplace=True)
idx = pd.IndexSlice
df.loc[idx[:, '1Y'], '10Y']


option_expiries = ['1M', '1Y', '10Y']
swap_tenors = ['2Y', '10Y', '30Y']
m = len(option_expiries)
n = len(swap_tenors)
swaption_grid = list(itertools.product(*[option_expiries, swap_tenors]))
n_strikes = 100
strikes = np.linspace(-1.00, 6.00, n_strikes)


fig, axes = plt.subplots(m, n)
fig.set_dpi(200)
fig.set_size_inches((12, 12))
fig.tight_layout(w_pad=0.5, h_pad=2.0)

for ((option_expiry, swap_tenor), ax) in zip(swaption_grid, fig.get_axes()):
    beta, f, v_atm_n, rho, shift, volvol = list(df.loc[idx[:, option_expiry], swap_tenor].reset_index(level=1, drop=True))
    t = year_frac_from_maturity_label(option_expiry)
    sabr_ln = Hagan2002LognormalSABR(f/100, shift/100, t, v_atm_n/1e4, beta, rho, volvol)
    sabr_n = Hagan2002NormalSABR(f/100, shift/100, t, v_atm_n/1e4, beta, rho, volvol)
    sabr_ln_vols = [sabr_ln.normal_vol(k/100) * 1e4 for k in strikes]
    sabr_n_vols = [sabr_n.normal_vol(k/100) * 1e4 for k in strikes]
    ax.plot(strikes, sabr_ln_vols, linewidth=1.0, linestyle='-')
    ax.plot(strikes, sabr_n_vols, linewidth=1.0, linestyle='--')
    ax.set_xlim((-1.0, 6.0))
    ax.set_ylim((30., 170.))
    ax.set_title("{} into {}".format(option_expiry, swap_tenor))

line_sabr_ln = axes[0][0].get_lines()[0]
line_sabr_n = axes[0][0].get_lines()[1]
fig.legend(handles=(line_sabr_ln, line_sabr_n), labels=('Hagan 2002 Lognormal SABR', 'Hagan 2002 Normal SABR'), loc='upper right')
fig.suptitle("Hagan 2002 SABR: Lognormal vs Normal expansion", fontsize=16)
fig.subplots_adjust(top=0.92)
#fig.savefig("Lognormal SABR vs Normal SABR.pdf", format='pdf')
fig.show()