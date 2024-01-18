clear all
global sampsizes 1000 10000 /*50000 100000 250000 500000 1000000 2500000 5000000*/
global repeats 2
global nx 10
global ngroups 100
adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"
cd "$DRIVE/GitSoftware/mlad/Examples/logistic"

program define simlogit
  syntax [,NOBS(integer 1000) NX(integer 10) NGROUPS(integer 10) SIGMA(real 1)] 
  set obs `nobs'
  forvalues i = 1/`nx' {
    gen x`i' = rnormal()
    local cov `cov' x`i'* 0.1
    if `i'<`nx'{
      local cov `cov' +
    } 
  }  
  gen grp = runiformint(1,`ngroups')
  bysort grp:gen first = _n==1
  gen uj = rnormal(0,`sigma') if first
  bysort grp: replace uj = uj[1] if _n>1
  
 
   
  gen logodds = 0 + `cov' + uj
  gen p = invlogit(logodds)
  gen y = runiform()<p
end



forvalues i = 1/$repeats {
  local tvars `tvars' time`i'
}

set seed 129837
foreach ss in $sampsizes {
  postfile logitpost sampsize str20 method `tvars' using Results/logit_results_`ss', replace
  clear 
  simlogit, nobs(`ss') nx(${nx})  ngroups(${ngroups})
  compress

  local covlist
  forvalues i = 1/$nx {
    local covlist `covlist' x`i'
  }  

  tempname tempdata
  save `tempdata', replace
  // fit 1 model to load ml routines
  tr 2:mlad (xb: = `covlist' ) (lnsig2u: = ) , othervars(y) llfile(logit_ll) id(grp) nodesgh(12) nojit

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
    mlad (xb: = `covlist', ) (lnsig2u: = ) , othervars(y) llfile(logit_ll) id(grp) nodesgh(12)
    timer off 1
    timer list
    local tresults `tresults' (`r(t1)')
  }
  post logitpost (`ss') ("`method'") `tresults'  
  postclose logitpost
 
}
