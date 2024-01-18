clear all
global sampsizes /*1000 10000 50000  100000 250000  500000 */1000000 /*2500000 5000000*/
global repeats 1
global nx 10
global Ngammacovs 2

adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"
cd "${DRIVE}/GitSoftware/mlad/Examples/rcs_hazard"
!cp -r   ${DRIVE}/GitSoftware/mlad/mlad/py/mladutil.py ${DRIVE}/ado/personal/py/



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
  replace t = ceil(t*365.25)/365.25
end

clear 
forvalues i = 1/$repeats {
  local tvars `tvars' time`i'
}

capture postutil clear
set seed 2345
foreach ss in $sampsizes {
  postfile weibpost sampsize str20 method `tvars' using Results/rcshaz_results${HPC}_`ss', replace
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

  gen lnt = ln(_t)
  rcsgen lnt, gen(_rcs) df(4) if2(_d==1)
  mata: st_matrix("knots",strtoreal(tokens(st_global("r(knots)"))))  
  
  tempfile tempdata
  save `tempdata', replace
  // fit 1 model to load ml routines
  matrix list knots
  timer clear
  timer on 1
  scalar Nnodes = 30
  mlad (xb: = x1 x2 x3 x4 x5 x6 x7 x8 x9 x10, nocons ) ///
       (rcs: = _rcs1 _rcs2 _rcs3 _rcs4)                ///
       , llfile(rcs_numint_jax)                        ///
         othervars(_t0 _t _d)                          ///
         othervarnames(t0 t d)                         ///
         matrices(knots)                               ///
         staticscalars(Nnodes) 
  timer off 1       


// JAXJIT 1 PROCESSOR
  use `tempdata', clear
  local method "jaxjit_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    di "running mlad"
    mlad (xb: = x1 x2 x3 x4 x5 x6 x7 x8 x9 x10, nocons ) ///
         (rcs: = _rcs1 _rcs2 _rcs3 _rcs4)                ///
         , llfile(rcs_numint_jax)                        ///
           othervars(_t0 _t _d)                          ///
         othervarnames(t0 t d)                         ///
         matrices(knots)                               ///
         staticscalars(Nnodes) 
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'

/*  
// JAXJIT 2 PROCESSOR
  use `tempdata', clear
  local method "jaxjit_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    mlad (xb: = x1 x2 x3 x4 x5 x6 x7 x8 x9 x10, nocons ) ///
         (rcs: = _rcs1 _rcs2 _rcs3 _rcs4)                ///
         , llfile(rcs_numint_jax)                        ///
           othervars(_t0 _t _d)                          ///
         othervarnames(t0 t d)                         ///
         matrices(knots)                               ///
         staticscalars(Nnodes) 
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'
*/
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
    mlad (xb: = x1 x2 x3 x4 x5 x6 x7 x8 x9 x10, nocons ) ///
         (rcs: = _rcs1 _rcs2 _rcs3 _rcs4)                ///
         , llfile(rcs_numint_jax)                        ///
           othervars(_t0 _t _d)                              ///
           matrices(knots)                               ///
           scalars(Nnodes) 
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
    mlad (xb: = x1 x2 x3 x4 x5 x6 x7 x8 x9 x10, nocons ) ///
         (rcs: = _rcs1 _rcs2 _rcs3 _rcs4)                ///
         , llfile(rcs_numint_jax)                        ///
           othervars(_t0 _t _d)                              ///
           matrices(knots)                               ///
           scalars(Nnodes) 
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'
*/


// stgenreg 1 PROCESSOR


  use `tempdata', clear
  local method "stgenreg_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    di "running stgenreg"
    stgenreg, loghazard([xb]) xb(x1 x2 x3 x4 x5 x6 x7 x8 x9 x10| #rcs(df(4) noorthog)) nodes(30)
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'
*/

/*  
// stgenreg 2 PROCESSOR
  use `tempdata', clear
  local method "stgenreg_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    stgenreg, loghazard([xb]) xb(x1 x2 x3 x4 x5 x6 x7 x8 x9 x10| #rcs(df(4) noorthog)) nodes(30)
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'
*/
 
// stpm3 1 PROCESSOR
  use `tempdata', clear
  local method "stpm3_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    di "running stpm3"
    stpm3 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 , df(4) scale(lnhazard)
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'

/*  
// strcs 2 PROCESSOR
  use `tempdata', clear
  local method "stpm3_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    strcs x1 x2 x3 x4 x5 x6 x7 x8 x9 x10, df(4) nohr noorthog
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'
*/  
// merlin 1 PROCESSOR

  use `tempdata', clear
  local method "merlin_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    di "running merlin"
    merlin (_t x1 x2 x3 x4 x5 x6 x7 x8 x9 x10          ///
        rcs(_t, df(5)  noorthog log event) /// 
        , family(loghazard, failure(_d)) ) ,    ///
        chintpoints(50)    
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'
  
/*  
// merlin 2 PROCESSOR
  use `tempdata', clear
  local method "merlin_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    merlin (_t x1 x2 x3 x4 x5 x6 x7 x8 x9 x10         ///
        rcs(_t, df(5)  noorthog log event) /// 
        , family(loghazard, failure(_d)) ) ,    ///
        chintpoints(50)    
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post weibpost (`ss') ("`method'") `tresults'  
*/  

  

  postclose weibpost  
}


