  clear all
  adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"

  cd "$DRIVE/GitSoftware/mlad/Examples/logit_re"
  !cp -r   "${DRIVE}/GitSoftware/mlad/mlad/py/mladutil.py" "${DRIVE}/ado/personal/py/mladutil.py"


  //use https://www.stata-press.com/data/r16/union
  //replace idcode = runiformint(1,50000)
  timer clear 
  set processors 2

  set seed 8798
  set obs 100000
  egen id = seq(), from(1) to(1000)

  bysort id: gen iobs = _n
  bysort id (iobs): gen first = _n==1
  gen u0j = rnormal(0,1) if first
  bysort id (iobs): replace u0j = u0j[1]
  gen u1j = rnormal(0,0.2) if first
  bysort id (iobs): replace u1j = u1j[1]

  forvalues i = 1/1 {
    gen x`i' = rnormal()
    local cov `cov' x`i' 0.1
  }  

  gen xb = 0.1*x1 + u0j + u1j*x1
  gen prob = invlogit(xb)
  gen y = runiform()<prob


  qui distinct id
  scalar Nid = `r(ndistinct)'
  scalar Nodes = 7
  matrix b = 0.14, -0.03,0.91,0.056,-0.104
  mlad (xb: = x1, ) ///
       (v0:)       ///
       (v1:)       ///
       (v01:)      ///
       , othervars(y x1) othervarnames(y z) ///
       init(b,copy) ///
       staticscalars(Nid Nodes) llfile(logit_re) id(id)  trace  
  timer off 1
ml display

melogit y x1 || id: x1, intmethod(ghermite) intpoints(7) cov(unstr) //dnumerical 

stmixed x1 || id: x1, dist(weib)

