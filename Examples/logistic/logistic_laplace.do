clear all
adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"

cd "$DRIVE/GitSoftware/mlad/Examples/logistic"
!cp -r   "${DRIVE}/GitSoftware/mlad/mlad/py/mladutil.py" "${DRIVE}/ado/personal/py/mladutil.py"


//use https://www.stata-press.com/data/r16/union
//replace idcode = runiformint(1,50000)
timer clear 
set processors 1


set seed 142
set obs 10000
local Ngroups = ceil(`=_N/100')
egen id = seq(), from(1) to(`Ngroups')

bysort id: gen iobs = _n
bysort id (iobs): gen first = _n==1
gen uj = rnormal(0,2) if first
bysort id (iobs): replace uj = uj[1]

forvalues i = 1/10 {
  gen x`i' = rnormal()
  local cov `cov' x`i' 0.05
}  

gen xb = 0.1*x1 + 0.1*x2 + 0.1*x3 + 0.1*x4 + 0.1*x5 +  ///
         0.1*x6 + 0.1*x7 + 0.1*x8 + 0.1*x9 + 0.1*x10 + uj
gen prob = invlogit(xb)
gen y = runiform()<prob

gen cons = 1

timer on 1
qui distinct id
scalar Nid = `r(ndistinct)'
scalar Nnodes = 12

// fixed effect model
logit y x1 x2 x3 x4 x5 x6 x7 x8 x9 x10
matrix b = e(b)
matrix b = b,0.5

mlad (xb: = x1 x2 x3 x4 x5 x6 x7 x8 x9 x10, )  ///
     (sigma2: = ),  ///
     othervars(y) /// 
     staticscalars(Nid Nnodes) ///
     llfile(logit_ll) ///
     init(b, copy) ///
     id(id) search(on)
ml display     
timer off 1


timer on 2
melogit y x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 || id: x2, intmethod(mvaghermite) covariance(unstructured) //dnumerical //intpoints(12) dnumerical
timer off 2


timer list
set processors 2
