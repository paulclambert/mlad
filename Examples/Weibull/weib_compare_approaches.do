  clear all
  global sampsizes 1000 /*10000 50000 100000 250000 500000 1000000 2500000 5000000*/
  global repeats 3
  global nx 10
  global Ngammacovs 2
  adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"
  cd "${DRIVE}/GitSoftware/mlad/Examples/Weibull"


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
    //postfile weibpost sampsize str20 method `tvars' using Results/weib_results_`ss', replace
    postfile weibpost sampsize str20 method `tvars' using Results/temp_`ss', replace

    clear 
    simweib, nobs(`ss') nx(${nx})  
    compress

    stset t, f(d==1)
    local covlist
    local gcovlist
    forvalues i = 1/$nx {
      local covlist `covlist' x`i'
    }  
    forvalues i = 1/$Ngammacovs {
      local gcovlist `gcovlist' x`i'
    }
    tempname tempdata
    save `tempdata', replace
    // fit 1 model to load ml routines
    mlad (ln_lambda: = `covlist', ) (lngamma: = `gcovlist'), ///
         llfile(weib_like_jax)                               ///
         othervars(_t _d)                                    ///
         othervarnames(t d)

// JAXJIT 1 PROCESSOR
  use `tempdata', clear
  local method "jaxjit_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    mlad (ln_lambda: = `covlist', ) (lngamma: = `gcovlist'), ///
         llfile(weib_like_jax)                               ///
         othervars(_t _d)                                    ///
         othervarnames(t d)    
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
    mlad (ln_lambda: = `covlist', ) (lngamma: = `gcovlist'), ///
         llfile(weib_like_jax)                               ///
         othervars(_t _d)                                    ///
         othervarnames(t d)    
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'

// JAX NOJIT 1 PROCESSOR
  use `tempdata', clear
  local method "jax_nojit_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    mlad (ln_lambda: = `covlist', ) (lngamma: = `gcovlist'), ///
         llfile(weib_like_jax)                               ///
         othervars(_t _d)                                    ///
         othervarnames(t d)    
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
    mlad (ln_lambda: = `covlist', ) (lngamma: = `gcovlist'), ///
         llfile(weib_like_jax)                               ///
         othervars(_t _d)                                    ///
         othervarnames(t d)    
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'

// ML d0 1 PROCESSOR
  use `tempdata', clear
  local method "ml_d0_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    ml model d0 myweib ( = `covlist') (=`gcovlist'), maximize search(off) 
    ml display
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'

// ML d0 2 PROCESSOR
  use `tempdata', clear
  local method "ml_d0_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    ml model d0 myweib ( = `covlist') (=`gcovlist'), maximize search(off) 
    ml display
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'

// ML d2 1 PROCESSOR
  use `tempdata', clear
  local method "ml_d2_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    ml model d2 myweib ( = `covlist') (=`gcovlist'), maximize search(off) 
    ml display
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'

// ML d2 1 PROCESSOR
  use `tempdata', clear
  local method "ml_d2_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    ml model d2 myweib ( = `covlist') (=`gcovlist'), maximize search(off) 
    ml display
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'

// streg 1 PROCESSOR
  use `tempdata', clear
  local method "streg_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    streg `covlist', dist(weibull) anc(`gcovlist')
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'

// streg 2 PROCESSOR
  use `tempdata', clear
  local method "streg_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    streg `covlist', dist(weibull) anc(`gcovlist')
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'

  postclose weibpost
}


