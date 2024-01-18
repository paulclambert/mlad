clear all
global sampsizes 1000 /*10000 50000 100000 250000 500000 1000000 2500000 5000000 10000000*/
global repeats 1
global nx 10
adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"
cd "${DRIVE}/GitSoftware/mlad/Examples/mixture_cure"
!cp -r   "${DRIVE}/GitSoftware/mlad/mlad/py/mladutil.py" "${DRIVE}/ado/personal/py/mladutil.py"

program define simweib
  syntax [,NOBS(integer 1000) lambda1(real 0.2) gamma1(real 0.6) NX(integer 10)] 
  set obs `nobs'
  forvalues i = 1/`nx' {
    gen x`i' = rnormal()
    local cov `cov' x`i' 0.1
  }  
  
  forvalues i = 1/`nx' {
    local covlist `covlist' + 0.1*x`i' 
  }  
  
  survsim t d, dist(weibull) lambda(`lambda1') gamma(`gamma1') maxt(10) cov(`cov')

  gen cure = runiform()<(invlogit(0 `covlist'))
  replace t = 10 if !cure
  replace d=0 if !cure
  
end


forvalues i = 1/$repeats {
  local tvars `tvars' time`i'
}

set trace off
set seed 129837
foreach ss in $sampsizes {
  postfile curepost sampsize str20 method `tvars' using Results/mixture_cure_results_`ss', replace
  clear 
  simweib, nobs(`ss') nx(${nx})  
  stset t, f(d==1)
  gen zeros = 0
  
  local covlist
  forvalues i = 1/$nx {
    local covlist `covlist' x`i'
  }  
  
  // JAXJIT 1 PROCESSOR
  local method "jaxjit_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    mlad (pi:= `covlist')           ///
         (lnlam:=`covlist')         ///
         (lngam:=)                  ///
         , othervars(_t _d zeros)   ///
           othervarnames(t d rate) /// 
           llfile(strsmix_ll)       ///
           search(on) diff  
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post curepost (`ss') ("`method'") `tresults'

/*  
// JAXJIT 2 PROCESSOR
  local method "jaxjit_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    mlad (pi:= `covlist')          ///
         (lnlam:=`covlist')        ///
         (lngam:=)                 ///
         , othervars(_t _d zeros)  ///
           llfile(strsmix_ll)      ///
           search(on) diff  
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post curepost (`ss') ("`method'") `tresults'
*/
/*  
// JAXnoJIT 1 PROCESSOR
  local method "jax_nojit_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    mlad (pi:= `covlist')          ///
         (lnlam:=`covlist')        ///
         (lngam:=)                 ///
         , othervars(_t _d zeros)  ///
           llfile(strsmix_ll)      ///
           search(on) diff nojit 
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post curepost (`ss') ("`method'") `tresults'

// JAXnoJIT 2 PROCESSOR
  local method "jax_nojit_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    mlad (pi:= `covlist')          ///
         (lnlam:=`covlist')        ///
         (lngam:=)                 ///
         , othervars(_t _d zeros)  ///
           llfile(strsmix_ll)      ///
           search(on) diff nojit 
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post curepost (`ss') ("`method'") `tresults'  
*/  
// strsmix 1 PROCESSOR
  local method "strsmix_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    strsmix `covlist', dist(weibull) link(logit) k1(`covlist') bhazard(zeros) diff
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post curepost (`ss') ("`method'") `tresults'

// strsmix 2 PROCESSOR
/*
  local method "strsmix_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    strsmix `covlist', dist(weibull) link(logit) k1(`covlist') bhazard(zeros) diff
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post curepost (`ss') ("`method'") `tresults'  
  
  */
  postclose curepost
  
}

  

  

