clear all
global sampsizes 10000 /*50000 100000 250000 500000 1000000 2500000 5000000 10000000*/
global repeats 1
global nx 10
global Ngammacovs 2
adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"
cd "${DRIVE}/GitSoftware/mlad/Examples/Interval_Censoring"


program define simweib
  syntax [,NOBS(integer 1000) lambda(real 0.2) gamma(real 0.8) NX(integer 10)] 
  set obs `nobs'
  forvalues i = 1/`nx' {
    gen x`i' = rnormal()
    local cov `cov' x`i' 0.1
  }  


  survsim rtime d, dist(weib) lambda(`lambda') gamma(`gamma') maxt(5) cov(`cov')
  gen double ltime = cond(runiform()<1 & d==1,runiform(0.2,0.8)*rtime,rtime) 
  replace ltime = rtime if d==0
  replace rtime = . if d==0

  gen byte ctype = .
  qui replace ctype = 1 if ltime==rtime                         // UC
  qui replace ctype = 2 if rtime >= .                           // RC
  qui replace ctype = 3 if (ltime >= . | ltime==0)              // LC
  qui replace ctype = 4 if (rtime-ltime)>0 & !inlist(ctype,2,3) // IC 
  
  gen double ltime2 = cond(ltime==0,1e-8,ltime)
  gen double rtime2 = cond(rtime==.,99,rtime)  
  
end


forvalues i = 1/$repeats {
  local tvars `tvars' time`i'
}


set seed 129837
foreach ss in $sampsizes {
  postfile weibpost sampsize str20 method `tvars' using Results/IC_results_`ss', replace
  clear 
  simweib, nobs(`ss') nx(${nx})  
  compress

  local covlist
  local gcovlist
  forvalues i = 1/$nx {
    local covlist `covlist' x`i'
  }  
  forvalues i = 1/$Ngammacovs {
    local gcovlist `gcovlist' x`i'
  }
  tempfile tempdata
  save `tempdata', replace
  // fit 1 model to load ml routines
    mlad (ln_lambda: = `covlist', ) (lngamma: = `gcovlist') , ///
    othervars(ctype ltime2 rtime2) ///
    othervarnames(ctype ltime rtime) ///
    llfile(weib_ic_ll)  
  

// JAXJIT 1 PROCESSOR
  use `tempdata', clear
  local method "jaxjit_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    mlad (ln_lambda: = `covlist', ) (lngamma: = `gcovlist') , ///
      othervars(ctype ltime2 rtime2) ///
      othervarnames(ctype ltime rtime) ///
      llfile(weib_ic_ll)  
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'

// JAXJIT 2 PROCESSOR
  use `tempdata', clear
  local method "jaxjit_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    mlad (ln_lambda: = `covlist', ) (lngamma: = `gcovlist') , /// 
         othervars(ctype ltime2 rtime2) ///
         othervarnames(ctype ltime rtime) ///
         llfile(weib_ic_ll)  
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'

/*  
// JAX NOJIT 1 PROCESSOR
  use `tempdata', clear
  local method "jax_nojit_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    mlad (ln_lambda: = `covlist', ) (lngamma: = `gcovlist') , othervars(ctype ltime rtime) llfile(weib_ic_ll) jax nojit
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'

// JAX NOJIT 2 PROCESSOR
  use `tempdata', clear
  local method "jax_nojit_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    mlad (ln_lambda: = `covlist', ) (lngamma: = `gcovlist') , othervars(ctype ltime rtime) llfile(weib_ic_ll) jax nojit
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'
*/
// stintreg 1 PROCESSOR
  use `tempdata', clear
  local method "stintreg_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    stintreg `covlist', dist(weibull) anc(`gcovlist') interval(ltime rtime)
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'

// stintreg 2 PROCESSOR
  use `tempdata', clear
  local method "stintreg_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    stintreg `covlist', dist(weibull) anc(`gcovlist') interval(ltime rtime)
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'
  
  postclose weibpost
}


