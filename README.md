# frm

[![PyPI](https://img.shields.io/pypi/v/frm?label=PyPI%20Package)](https://pypi.org/project/frm/)

frm is an in-development python package for quantitative financial pricing and modelling.
frm uses common 3rd party python packages for scientific computing (numpy, scipy, pandas, numba, matplotlib) and the holidays package.

This package will have a similar function set to Quantlib however we want to make it more accessible, documented, productive though:
1. The python (core + 3rd party libaries) implementation
2. Academic and industry references (at specific lines of code) to support users own validation and testing
3. Supporting [excel/VBA models](https://frmcalcs.com) that validate/support the code 
4. Significant code examples  

# Installation
```bash
pip install frm
```

## In progress

Interest rate swaps
- pricing
- schedule construction (including detailed stub logic) 
- iterative single currency bootstrapping
- fixed rate / spread par solvers

Vanilla European FX options
- pricing + greeks (under Garman-Kohlhagen)
- volatility surface construction (smile construction via Heston or splines)  

## Pipeline
- SABR volatility model
- European interest rate swaption pricing
- CDS Bootstrapper

## Hosted examples
At [frmcalcs.com](https://frmcalcs.com), the following tools are are hosted:
- FX forward valuations and exposure modelling for CVA/DVA 
- Vanilla FX option valuations




