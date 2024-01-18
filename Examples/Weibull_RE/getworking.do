clear all
global nx 10
adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"
cd "$DRIVE/GitSoftware/mlad/Examples/Weibull_RE"
!cp -r   "${DRIVE}/GitSoftware/mlad/mlad/py/mladutil.py" "${DRIVE}/ado/personal/py/mladutil.py"

program define simweib
  syntax [,NOBS(integer 1000) lambda(real 0.2) gamma(real 0.8) NX(integer 10)] 
  set obs `nobs'
  local Ngroups = ceil(`nobs'/100)
  egen id = seq(), from(1) to(`Ngroups')
  
  bysort id: gen iobs = _n
  bysort id (iobs): gen first = _n==1
  gen uj = rnormal(0,0.5) if first
  bysort id (iobs): replace uj = uj[1]  
  
  forvalues i = 1/`nx' {
    gen x`i' = rnormal()
    local cov `cov' x`i' 0.1
  }  


  survsim t d, dist(weib) lambda(`lambda') gamma(`gamma') maxt(5) cov(`cov' uj 1)
end

clear 


set seed 129837
local ss 100000
simweib, nobs(`ss') nx(${nx})  

stset t, f(d==1)
local covlist
forvalues i = 1/$nx {
  local covlist `covlist' x`i'
}  


// constant only
qui distinct id
scalar Nid = `r(ndistinct)'   
scalar Nnodes = 7
gen cons = 1


gen bob = .
mlad (ln_lambda:  = `covlist'  , )     ///
     (ln_gamma:   = )                  ///
     (v0:=cons, nocons)                ///
     , othervars(_d _t)                ///
       othervarnames(d t)              ///
       llfile(weib_like_re)            ///
       staticscalars(Nid Nnodes)      ///
       id(id)          search(on) 

       
timer clear
timer on 1       
mlad (ln_lambda:  = `covlist'  , )     ///
     (ln_gamma:   = )                  ///
     (v0:=cons, nocons)                ///
     , othervars(_d _t)                ///
       othervarnames(d t)              ///
       llfile(weib_like_re)            ///
       staticscalars(Nid Nnodes)      ///
       id(id)          search(on) 
ml display       
timer off 1
timer list       

       
timer on 2       
mestreg `covlist' || id: , dist(weib) nohr  intmethod(ghermite) dnumerical intpoints(7)
timer off 2
timer list

timer on 3       
mestreg `covlist' || id: , dist(weib) nohr  
timer off 3
timer list








