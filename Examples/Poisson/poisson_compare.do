clear all
global sampsizes 1000 10000 /*50000 100000 250000 500000 1000000  2500000 5000000 10000000*/
global repeats 1
global nx 10
global NP 2
cd "$DRIVE/GitSoftware/mlad/Examples/Poisson"

adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"
//!cp -r   ${DRIVE}/GitSoftware/mlad/mlad/py/mladutil.py ${DRIVE}/ado/personal/py/
if "$HPC" != "" {
  python set exec "/cm/shared/apps/python/gcc/3.6.4/bin/python3"
  python set userpath  ${DRIVE}/ado/personal/py/
}

program define simweib
  syntax [,NOBS(integer 1000) lambda(real 0.2) gamma(real 0.8) NX(integer 10)] 
  set obs `nobs'
  
  
  forvalues i = 1/`nx' {
    gen x`i' = rnormal()
    local cov `cov' x`i' 0.1
  }  
  
  
  survsim t d, dist(weib) lambda(`lambda') gamma(`gamma') maxt(5) cov(`cov')
end

clear 

forvalues i = 1/$repeats {
  local tvars `tvars' time`i'
}

set seed 234

foreach ss in $sampsizes {
  postfile poispost sampsize str20 method `tvars' using Results/poisson_results${HPC}_`ss'_g_and_H, replace
  clear 
  simweib, nobs(`ss') nx(${nx})  
  gen id = _n

  stset t, f(d==1) id(id)
  local covlist
  forvalues i = 1/$nx {
    local covlist `covlist' x`i'
  }  
  stsplit fu, every(`=1/12')
  gen risktime = _t-_t0
  gen lnrisktime = ln(risktime)
  gen lnt  = ln(0.5*(_t+_t0))
  rcsgen lnt, df(5) gen(rcs) orthog if2(_d==1)
	  
   glm d if _n<1000, family(poisson) link(log) iter(1)
   mlad (xb:  = ,offset(lnrisktime) )   ///
       if _n<1000 , othervars(_d)                    ///
	 othervarnames(y)                 ///
	 llfile(poisson)                    ///
	 search(off)  

  // glm 
  forvalues p = 1/$NP {
    set processors `p'
    local method "glm-p`p'"
    local tresults
    forvalues i = 1/$repeats {
      timer clear
      timer on 1
      glm _d rcs* x1-x10, family(poisson) link(log) lnoffset(risktime) 
      timer off 1
      timer list
      local tresults `tresults' (`r(t1)')
    } 
    
      
    post poispost (`ss') ("`method'") `tresults'
  }
  
  // mlad AD
  forvalues p = 1/$NP {
    set processors `p'
    local method "mladAD-p`p'"
    local tresults
    forvalues i = 1/$repeats {
      timer clear
      timer on 1
      mlad (xb:  = rcs* `covlist', offset(lnrisktime))   ///
	 , othervars(_d)                    ///
	   othervarnames(y)                 ///
	   llfile(poisson)                    ///
	   search(off) 
      ml display    
      timer off 1
      timer list
      local tresults `tresults' (`r(t1)')
    }

    post poispost (`ss') ("`method'") `tresults'
  }
  
  // mlad G & H
  forvalues p = 1/$NP {
    set processors `p'
    local method "mladGH-p`p'"
    local tresults
    forvalues i = 1/$repeats {
      timer clear
      timer on 1
      mlad (xb:  = rcs* `covlist', offset(lnrisktime))   ///
	 , othervars(_d)                    ///
	   othervarnames(y)                 ///
	   llfile(poisson)                    ///
     pygradient pyhess ///
	   search(off) 
      ml display    
      timer off 1
      timer list
      local tresults `tresults' (`r(t1)')
    }
    post poispost (`ss') ("`method'") `tresults'
  }  
  postclose poispost
}  
