# frm

[![PyPI](https://img.shields.io/pypi/v/frm?label=PyPI%20Package)](https://pypi.org/project/frm/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/84233a0d4c944e7e92abdb4011db33b4)](https://app.codacy.com/gh/frmcalcs/frm/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

frm is an in-development python package for quantitative financial pricing and modelling.
frm uses common 3rd party python packages for scientific computing (numpy, scipy, pandas, numba, matplotlib) and the holidays package.

This package will have a similar function set to Quantlib however we want to make it more *accessible*, *documented*, and *productive* though:
1. The python (core + 3rd party libaries) implementation
2. Academic and industry references (at specific lines of code) to support users own validation and testing
3. Supporting [excel/VBA models](https://github.com/frmcalcs/frm/tree/master/excel_models) that validate/support the code 
4. Significant code examples  

## Installation
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
At <https://frmcalcs.com>, the following tools are are hosted:
- FX forward valuations and exposure modelling for CVA/DVA 
- Vanilla FX option valuations




