/***************************************************************************
This file is distributed as part of the -strs- package.

This file is available at:
http://www.pauldickman.com/rsmodel/stata_colon/age-std.do

This code illustrates three approaches to obtaining age-standardised  
estimates of net survival (Pohar Perme) using ICSS weights. 

1. Traditional age-standardisation. First estimate net survival
   within age stratum and then take the weighted average of the
   age-specific estimates (using ICSS weights).
   
2. Brenner approach. Each individual is given an individual-specific
   weight and a weighted life table produced. Obviates the need for
   age specific estimates in strata that may be small or not have
   sufficient follow-up.
   
3. Same as Brenner approach, but implemented using a more general
   approach with individual weights. Weights must be calculated before
   calling -strs-.

Approaches 2 and 3 are identical. Approach 3 illustrates a general 
approach to weighting that ca be used in other applications.

More details are available in the -strs- help file.   
   
Authors: Paul Dickman, Enzo Coviello, Paul Lambert, Mark Rutherford
Updated: 19 February 2020

References

Brenner, H., V. Arndt, O. Gefeller, and T. Hakulinen.  2004.
An alternative approach to age adjustment of cancer survival rates.
European Journal of Cancer 40:2317-2322.

Corazziari, I., M. Quinn, and R. Capocaccia.  2004.
Standard cancer patient population for age standardising survival ratios.
European Journal of Cancer 40: 2307-2316.

***************************************************************************/
set more off
use http://pauldickman.com/data/colon.dta if stage == 1, clear

// Reclassify age groups according to International Cancer Survival Standard (ICSS) 
drop agegrp
label drop agegrp
egen agegrp=cut(age), at(0 15 45 55 65 75 200) icodes
label variable agegrp "Age group"
label define agegrp 1 "15-44" 2 "45-54" 3 "55-64" 4 "65-74" 5 "75+" 
label values agegrp agegrp

// Specify ICSS weights for each agegroup
recode agegrp (1=0.07) (2=0.12) (3=0.23) (4=0.29) (5=0.29), gen(ICSSwt)

stset exit, origin(dx) fail(status==1 2) id(id) scale(365.24)

// Traditional age-standardisation
strs using http://pauldickman.com/data/popmort [iw=ICSSwt], ///
    breaks(0(0.5)5) mergeby(_year sex _age) ///
    standstrata(agegrp) pohar f(%7.5f) 

// Pohar Perme using Brenner (2004) approach	
strs using http://pauldickman.com/data/popmort [iw=ICSSwt], /// 
    breaks(0(0.5)5) mergeby(_year sex _age) ///
    standstrata(agegrp) pohar f(%7.5f) brenner

// Pohar Perme using Brenner approach, but implemented with individual weights
// First create the weights, which are the same as the Brenner weights
local total= _N
bysort agegrp:gen a_age = _N/`total'
gen wt = ICSSwt/a_age

strs using http://pauldickman.com/data/popmort, /// 
    breaks(0(0.5)5) mergeby(_year sex _age) ///
    pohar f(%7.5f) indweight(wt)
