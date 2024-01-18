clear all
global sampsizes 2500000
global nx 10
global Ngammacovs 2
adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"
cd "${DRIVE}/GitSoftware/mlad/Examples/rcs_hazard"
copy "${DRIVE}/GitSoftware/mlad/mlad/py/mladutil.py" "${DRIVE}/ado/personal/py/mladutil.py", replace



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
set seed 4894

clear 
simweib, nobs(${sampsizes}) nx(${nx})  

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

// fit 1 model to load ml routines
matrix list knots
scalar Nnodes = 30
mlad (xb: = x1 x2 x3 x4 x5 x6 x7 x8 x9 x10, nocons ) ///
     (rcs: = _rcs1 _rcs2 _rcs3 _rcs4)                ///
     , llfile(rcs_numint_jax)                        ///
       othervars(_t0 _t _d)                          ///
       scalars(Nnodes)                               ///
       matrices(knots)                               
