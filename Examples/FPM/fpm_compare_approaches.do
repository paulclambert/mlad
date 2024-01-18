clear all
global sampsizes /*1000 10000  50000*/ 100000 /*250000 500000 1000000 2500000 5000000  10000000*/
global repeats 1
global nx 10
global Ngammacovs 2
adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"
cd "${DRIVE}/GitSoftware/mlad/Examples/FPM"


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

set seed 129837
foreach ss in $sampsizes {
  postfile fpmpost sampsize str20 method `tvars' using Results/fpm_results_`ss', replace
  clear 
  simweib, nobs(`ss') nx(${nx})  

  stset t, f(d==1)
  local covlist
  forvalues i = 1/$nx {
    local covlist `covlist' x`i'
  }  

  gen rate = 0
  tempfile tempdata
  stpm2  `covlist', scale(hazard) df(4) keepcons bhazard(rate)
  save `tempdata', replace

  // constant only
  mlad (xb:  = `covlist' _rcs1 , )                         ///
       (dxb: = _d_rcs1, nocons)                            ///
       , othervars(_d _t rate)                             ///
         othervarnames(d t rate)                             ///
         llfile(fpm_hazard_ll)                             ///
         constraints(1999) search(norescale) collinear     
  matrix b = e(b) 

    
  // full model  
  mlad (xb:  = `covlist' _rcs1 _rcs2 _rcs3 _rcs4, )        ///
       (dxb: = _d_rcs1 _d_rcs2 _d_rcs3 _d_rcs4, nocons),   ///
       othervars(_d _t rate)                               ///
       othervarnames(d t rate)                             ///
       llfile(fpm_hazard_ll)                               ///
       constraints(1999 1998 1997 1996)                    ///
       init(b) search(off) collinear


// JAXJIT 1 PROCESSOR
  use `tempdata', clear
  local method "jaxjit_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    // constant only
    mlad (xb:  = `covlist' _rcs1 , )                         ///
         (dxb: = _d_rcs1, nocons)                            ///
         , othervars(_d _t rate)                             ///
         othervarnames(d t rate)                             ///
           llfile(fpm_hazard_ll)                             ///
           constraints(1999) search(norescale) collinear     
    matrix b = e(b) 
    
    // full model  
    mlad (xb:  = `covlist' _rcs1 _rcs2 _rcs3 _rcs4, )        ///
         (dxb: = _d_rcs1 _d_rcs2 _d_rcs3 _d_rcs4, nocons),   ///
         othervars(_d _t rate)                             ///
         othervarnames(d t rate)                             ///                            
         llfile(fpm_hazard_ll)                               ///
         constraints(1999 1998 1997 1996)                    ///
         init(b) search(off) collinear    
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post fpmpost (`ss') ("`method'") `tresults'

// JAXJIT 2 PROCESSOR
  use `tempdata', clear
  local method "jaxjit_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    // constant only
    mlad (xb:  = `covlist' _rcs1 , )                         ///
         (dxb: = _d_rcs1, nocons)                            ///
         , othervars(_d _t rate)                             ///
         othervarnames(d t rate)                             ///
           llfile(fpm_hazard_ll)                             ///
           constraints(1999) search(norescale) collinear     
    matrix b = e(b) 
    
    // full model  
    mlad (xb:  = `covlist' _rcs1 _rcs2 _rcs3 _rcs4, )        ///
         (dxb: = _d_rcs1 _d_rcs2 _d_rcs3 _d_rcs4, nocons),   ///
         othervars(_d _t rate)                             ///
         othervarnames(d t rate)                             ///
         llfile(fpm_hazard_ll)                               ///
         constraints(1999 1998 1997 1996)                    ///
         init(b) search(off) collinear      
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post fpmpost (`ss') ("`method'") `tresults'

// JAX_noJIT 1 PROCESSOR
  use `tempdata', clear
  local method "jax_nojit_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    // constant only
    mlad (xb:  = `covlist' _rcs1 , )                         ///
         (dxb: = _d_rcs1, nocons)                            ///
         , othervars(_d _t rate)                             ///
         othervarnames(d t rate)                             ///
           llfile(fpm_hazard_ll)                             ///
           nojit                                             ///
           constraints(1999) search(norescale) collinear     
    matrix b = e(b) 
    
    // full model  
    mlad (xb:  = `covlist' _rcs1 _rcs2 _rcs3 _rcs4, )        ///
         (dxb: = _d_rcs1 _d_rcs2 _d_rcs3 _d_rcs4, nocons),   ///
        othervars(_d _t rate)                             ///
         othervarnames(d t rate)                             ///
         llfile(fpm_hazard_ll)                               ///
         nojit                                               ///
         constraints(1999 1998 1997 1996)                    ///
         init(b) search(off) collinear    
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post fpmpost (`ss') ("`method'") `tresults'

// JAXnoJIT 2 PROCESSOR
  use `tempdata', clear
  local method "jax_nojit_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    // constant only
    mlad (xb:  = `covlist' _rcs1 , )                         ///
         (dxb: = _d_rcs1, nocons)                            ///
         , othervars(_d _t rate)                             ///
         othervarnames(d t rate)                             ///
           llfile(fpm_hazard_ll)                             ///
           nojit                                            ///
           constraints(1999) search(norescale) collinear     
    matrix b = e(b) 
    
    // full model  
    mlad (xb:  = `covlist' _rcs1 _rcs2 _rcs3 _rcs4, )        ///
         (dxb: = _d_rcs1 _d_rcs2 _d_rcs3 _d_rcs4, nocons),   ///
         othervars(_d _t rate)                             ///
         othervarnames(d t rate)                             ///
         llfile(fpm_hazard_ll)                               ///
         nojit                                               ///
         constraints(1999 1998 1997 1996)                    ///
         init(b) search(off) collinear      
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post fpmpost (`ss') ("`method'") `tresults'
  
  

// stpm2 1 PROCESSOR
  use `tempdata', clear
  local method "stpm2_pro2"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    stpm2 `covlist', scale(hazard) df(4) lininit  bhazard(rate)
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post fpmpost (`ss') ("`method'") `tresults'

// stpm2 2 PROCESSOR
  use `tempdata', clear
  local method "stpm2_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    stpm2 `covlist', scale(hazard) df(4) lininit  bhazard(rate)
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post fpmpost (`ss') ("`method'") `tresults'  

// stpm2 1 PROCESSOR - cox init
  use `tempdata', clear
  local method "stpm2_cox_pro2"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    stpm2 `covlist', scale(hazard) df(4)   bhazard(rate)
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post fpmpost (`ss') ("`method'") `tresults'

// stpm2 2 PROCESSOR - cox init
  use `tempdata', clear
  local method "stpm2_cox_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    stpm2 `covlist', scale(hazard) df(4)  bhazard(rate)
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post fpmpost (`ss') ("`method'") `tresults'   
 
// stpm2 1 PROCESSOR - lf
  use `tempdata', clear
  local method "stpm2_lf_pro2"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    stpm2 `covlist', scale(hazard) df(4) oldest mlmethod(lf)  bhazard(rate)
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post fpmpost (`ss') ("`method'") `tresults'

// stpm2 2 PROCESSOR - lf
  use `tempdata', clear
  local method "stpm2_lf_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    stpm2 `covlist', scale(hazard) df(4) oldest mlmethod(lf)  bhazard(rate)
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post fpmpost (`ss') ("`method'") `tresults'  
 
  
  postclose fpmpost
  
}   
 
