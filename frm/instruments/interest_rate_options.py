# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

#from frm.term_structures.swap_curve import TermSwapCurve, OISCurve




# Cap/Floor = List of Caplets/Floorlets
# Can simplify to just have methods loop over rows in the schedule. Each row is a caplet/floorlet.
# Valuation + greeks = sum of caplet/floorlet values
# Cap/Floor schedule is the same as the underlying leg schedule
# Inputs - same as inputs to "float leg" in swap, plus Strike
# Caplet/floorlet valuation + greeks is simply black76 formula
# Valuation requires zero_curve object (forward and discount curves) and ir_vol_curve object.

# IR vol curve
# Pandas dataframe:
# (i) SABR smile for each expiry
# (ii) interpolation of SABR smile for every day




        
        
               