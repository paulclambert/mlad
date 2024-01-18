  clear all
  adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"

  cd "$DRIVE/GitSoftware/mlad/Examples/logistic"
  !cp -r   "${DRIVE}/GitSoftware/mlad/mlad/py/mladutil.py" "${DRIVE}/ado/personal/py/mladutil.py"


  //use https://www.stata-press.com/data/r16/union
  //replace idcode = runiformint(1,50000)
  timer clear 
  set processors 1


  set obs 1000000
  egen id = seq(), from(1) to(100000)

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


  timer on 1
  qui distinct id
  scalar Nid = `r(ndistinct)'
  mlad (xb: = x1 x2 x3 x4 x5 x6 x7 x8 x9 x10, )  (ln_sd:=), othervars(y) scalar(Nid) llfile(logit_ll) id(id) nodesgh(7)   //search(on)
  timer off 1


timer on 2
melogit y x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 || id:, intmethod(ghermite) intpoints(7)
timer off 2


timer list
set processors 2
