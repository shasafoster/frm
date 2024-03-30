# frm

A python package for quantitative finance and derivative pricing.
Emphasis on documentation, references and detailed examples.

This package will have a similar scope to Quantlib. The rational for this package is:
- QuantLib-Python is fiddly and due to SWIG it's hard to drill down into errors. 
- QuantLib C++ is in C++ which is unproductive for many use cases and is harder to read than native python (which is nearly pseudocode) 

## Complete

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
- Heston-Local Volatility model (for pricing path dependent FX options)



At <a href="https://www.frmcalcs.com/app_frm/" target="_blank">frmcalcs.com/app_frm/</a>, some use cases are are hosted.




