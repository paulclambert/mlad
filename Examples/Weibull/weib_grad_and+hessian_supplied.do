clear all
global sampsize 1000 
global nx 10
global Ngammacovs 2
adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"
cd "/media/paul/Storage/GitSoftware/mlad/Examples/Weibull"
//!cp -r   "${DRIVE}/GitSoftware/mlad/mlad/py/mladutil.py" "${DRIVE}/ado/personal/py/mladutil.py"


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



set seed 129837
clear 
simweib, nobs(${sampsize}) nx(${nx})  

stset t, f(d==1)
local covlist
local gcovlist
forvalues i = 1/$nx {
  local covlist `covlist' x`i'
}  
forvalues i = 1/$Ngammacovs {
  local gcovlist `gcovlist' x`i'
}
mlad (ln_lambda: = `covlist', ) (lngamma: = `gcovlist'), ///
       llfile(weib_like_jax)                               ///
       othervars(_t _d)                                    ///
       othervarnames(t d) pygradient pyhess mlmethod(d2debug)
ml display
       
      
clear       
simweib, nobs(10000000) nx(${nx})  

stset t, f(d==1)
local covlist
local gcovlist
forvalues i = 1/$nx {
  local covlist `covlist' x`i'
}  
forvalues i = 1/$Ngammacovs {
  local gcovlist `gcovlist' x`i'
}
timer clear
timer on 1
mlad (ln_lambda: = `covlist', ) (lngamma: = `gcovlist'), ///
       llfile(weib_like_jax)                               ///
       othervars(_t _d)                                    ///
       othervarnames(t d) pygradient pyhessian
ml display
timer off 1       

timer on 2
mlad (ln_lambda: = `covlist', ) (lngamma: = `gcovlist'), ///
       llfile(weib_like_jax)                               ///
       othervars(_t _d)                                    ///
       othervarnames(t d)  
ml display
timer off 2

timer list       
     
