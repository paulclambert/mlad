clear all
global nx 1
adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"
cd "$DRIVE/GitSoftware/mlad/Examples/Weibull_RE2"
!cp -r   "${DRIVE}/GitSoftware/mlad/mlad/py/mladutil.py" "${DRIVE}/ado/personal/py/mladutil.py"

program define simweib
  syntax [,NOBS(integer 1000) lambda(real 0.2) gamma(real 0.8) NX(integer 10)] 
  set obs `nobs'
  local Ngroups = ceil(`nobs'/100)
  egen id = seq(), from(1) to(`Ngroups')
  
  bysort id: gen iobs = _n
  bysort id (iobs): gen first = _n==1
  gen u0j = rnormal(0,0.4) if first
  gen u1j = rnormal(0,0.2) if first
  
  bysort id (iobs): replace u0j = u0j[1]  
  bysort id (iobs): replace u1j = u1j[1]  
  
  forvalues i = 1/`nx' {
    gen x`i' = rnormal()
    local cov `cov' x`i' 0.1
  }   

  gen x1tmp = x1*u1j
  survsim t d, dist(weib) lambda(`lambda') gamma(`gamma') maxt(5) cov(`cov' u0j 1 x1tmp 1)
end

clear 


set seed 0987
local ss 10000
**# Bookmark #1
simweib, nobs(`ss') nx(${nx})  

stset t, f(d==1)


timer clear
timer on 1
// Fixed effects model
mlad (ln_lambda:  = x1  , )     ///
     (ln_gamma:   = )                  ///
     , othervars(_d _t)                ///
       othervarnames(d t)              ///
       llfile(weib_ll) search(off)            
matrix initb = e(b)

matrix initb = initb, 0.1,0.004,0.001
//matrix initb = initb, 0.1,0.01
qui distinct id
scalar Nid = `r(ndistinct)'   
scalar Nnodes = 7

mlad (ln_lambda:  = x1  , )            ///
     (ln_gamma:   = )                  ///
     (v0:=, )                ///
     (v1:=, )                ///
     (v01:=, )               ///
     , othervars(_d _t x1)                ///
       othervarnames(d t z1)              ///
       scalars(Nnodes)                ///
       staticscalars(Nid)      ///
       id(id)                         ///
       pysetup(weib_setup) ///
       init(initb,copy)               ///
       llfile(weib_ll_re_adaptive) search(off)      
ml display   
timer off 1       

timer on 2
//mestreg x1 || id: x1 , dist(weib) nohr  cov(unstr) intmethod(gh) //dnumerical
  timer off 2
  
timer on 3  
stmixed x1 || id:x1, dist(weib)  cov(unstr) intmethod(gh) diff
timer off 3

timer on 4 
stmixed x1 || id:x1, dist(weib)  cov(unstr) 
timer off 4

timer list




local covlist
forvalues i = 1/$nx {
  local covlist `covlist' x`i'
}  


// constant only
qui distinct id
scalar Nid = `r(ndistinct)'   
scalar Nnodes = 7





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








