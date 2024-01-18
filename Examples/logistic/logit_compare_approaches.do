clear all
global sampsizes /*1000*/ 10000 //50000 100000 250000 500000 1000000 /*2500000 5000000*/
global repeats 2
clear all
adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"

cd "/media/paul/Storage/GitSoftware/mlad/Examples/logistic"
!cp -r   "${DRIVE}/GitSoftware/mlad/mlad/py/mladutil.py" "${DRIVE}/ado/personal/py/mladutil.py"

set tracedepth 3
set trace off
program define logitsim
  syntax, [NOBS(integer 1000)]
  set obs `nobs'
  local Ngroups = ceil(`nobs'/100)
  egen id = seq(), from(1) to(`Ngroups')
  
  bysort id: gen iobs = _n
  bysort id (iobs): gen first = _n==1
  gen uj = rnormal(0,2) if first
  bysort id (iobs): replace uj = uj[1]
  
  forvalues i = 1/10 {
    gen x`i' = rnormal()
    local cov `cov' x`i' 0.1
  }  
  
  gen xb = 0.1*x1 + 0.1*x2 + 0.1*x3 + 0.1*x4 + 0.1*x5 +  0.1*x6 + 0.1*x7 + 0.1*x8 + 0.1*x9 + 0.1*x10 + uj
  gen prob = invlogit(xb)
  gen y = runiform()<prob
end

clear 


forvalues i = 1/$repeats {
  local tvars `tvars' time`i'
}

set seed 129837
foreach ss in $sampsizes {
  postfile logitpost sampsize str20 method `tvars' using Results/logit_results_`ss', replace
  clear 
  logitsim, nobs(`ss') 

  local covlist
  forvalues i = 1/10 {
    local covlist `covlist' x`i'
  }  
  tempfile tempdata
  save `tempdata', replace
  // fit 1 model to load ml routines
  qui distinct id
  scalar Nid = `r(ndistinct)'  
  scalar Nnodes = 20
  mlad (xb: = x1 x2 x3 x4 x5 x6 x7 x8 x9 x10, )  (ln_sd:=), ///
       othervars(y) scalar(Nid Nnodes) llfile(logit_ll) id(id)
  di exp([ln_sd][_cons]/2)^2       
  melogit y x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 || id:, intmethod(ghermite) intpoints(7) 
}  

// JAXJIT 1 PROCESSOR
  use `tempdata', clear
  local method "jaxjit_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    qui distinct id
    scalar Nid = `r(ndistinct)'
    mlad (xb: = x1 x2 x3 x4 x5 x6 x7 x8 x9 x10, )  (ln_sd:=), ///
       othervars(y) scalar(Nid) llfile(logit_ll) id(id) nodesgh(7) jax  
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post logitpost (`ss') ("`method'") `tresults'

// JAXJIT 2 PROCESSOR
  use `tempdata', clear
  local method "jaxjit_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    qui distinct id
    scalar Nid = `r(ndistinct)'
    mlad (xb: = x1 x2 x3 x4 x5 x6 x7 x8 x9 x10, )  (ln_sd:=), ///
         othervars(y) scalar(Nid) llfile(logit_ll) id(id) nodesgh(7) jax  
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post logitpost (`ss') ("`method'") `tresults'

  
// MELOGIT 1 PROCESSOR
  use `tempdata', clear
  local method "melogit_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    melogit y x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 || id:, intmethod(ghermite) intpoints(7)
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post logitpost (`ss') ("`method'") `tresults'  
  
// MELOGIT 2 PROCESSOR
  use `tempdata', clear
  local method "melogit_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    melogit y x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 || id:, intmethod(ghermite) intpoints(7)
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post logitpost (`ss') ("`method'") `tresults'  
  
  
// MELOGIT 1 PROCESSOR
  use `tempdata', clear
  local method "melogit_dnum_pro1"
  set processors 1
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    melogit y x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 || id:, intmethod(ghermite) intpoints(7) dnumerical
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post logitpost (`ss') ("`method'") `tresults'  
  
// MELOGIT 2 PROCESSOR
  use `tempdata', clear
  local method "melogit_dnum_pro2"
  set processors 2
  timer clear 
  local tresults
  forvalues i = 1/$repeats {
    timer clear 
    timer on 1
    melogit y x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 || id:, intmethod(ghermite) intpoints(7) dnumerical
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post logitpost (`ss') ("`method'") `tresults'  
  
  postclose logitpost  
  
}


