clear all
adopath ++ "${DRIVE}/GitSoftware/mlad/mlad"

cd "$DRIVE/GitSoftware/mlad/Examples/logit_re"
!cp -r   "${DRIVE}/GitSoftware/mlad/mlad/py/mladutil.py" "${DRIVE}/ado/personal/py/mladutil.py"


  //use https://www.stata-press.com/data/r16/union
  //replace idcode = runiformint(1,50000)

set seed 8798
set obs 10000
egen id = seq(), from(1) to(100)

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

gen xb = 0 + 0.1*x1 + u0j + u1j*x1
gen prob = invlogit(xb)
gen y = runiform()<prob


set matastrict off
  qui distinct id
  scalar Nid = `r(ndistinct)'
mata:
function logit_lf() {

  nw = _gauss_hermite_nodes(7)
  nodes = nw[1,]
  weights = nw[2,]
  Nobs = st_nobs()
  Nid = st_numscalar("Nid")
  xb = st_data(.,st_local("xb"),.)
  v0 = st_numscalar(st_local("v0"))
  v1 = st_numscalar(st_local("v1"))
  v01 = st_numscalar(st_local("v01"))
  x1 = st_data(.,"x1",.)
  y = st_data(.,"y",.)
  id = st_data(.,"id",.)
  
  V = v0,v01\v01,v1
  C = cholesky(V)
  
  Z = (J(Nobs,1,1),x1)
  
  newnodes = C*(nodes\nodes)

  nodesmesh = J(0,2,.)
  combweights = J(1,49,.)
  k = 1
  for(i=1;i<=7;i++) {
    for(j=1;j<=7;j++) {
      nodesmesh = nodesmesh\(newnodes[1,i],newnodes[2,j])
      combweights[k] = (weights[i]:/sqrt(pi())) :* (weights[j]:/sqrt(pi()))
      k = k + 1
    } 
  }
  lnpdf = J(Nobs,49,.)
  for(i=1;i<=49;i++) {
   //nodesmesh[i,]'
   xbv = xb :+ sqrt(2):*(Z*nodesmesh[i,]')
   lnpdf[,i] = -ln(1:+exp(xbv)) :+ y:*xbv  
  }
  //lnpdf[1..10,1..10]
  P = panelsetup(id,1)
  pdf = exp(panelsum(lnpdf,P))
  llj = log(quadrowsum(J(Nid,1,combweights):*pdf))
  //llj[1..10]
  lnf= (quadsum(llj))
  st_numscalar(st_local("lnf"),lnf)
  //P+1
}
end


//    bob = jnp.sqrt(2)*(Z@(C@v.T))
//    lp = xb + bob
//    return((-jnp.log(1+jnp.exp(lp)) + M["y"]*lp)[:,0])    



program define logit_ml, eclass
  args todo b lnf g H
  tempvar xb v0 v1 v01
  mleval `xb' = `b', eq(1)
  mleval `v0'  = `b', eq(2)  scalar 
  mleval `v1'  = `b', eq(3)   scalar
  mleval `v01'  = `b', eq(4)   scalar
  mata logit_lf()
  
  
  
end
  

matrix b = 0,0,0.1,0.1,0
ml model d0 logit_ml (xb: = x1, ) ///
      (v0:, freeparm)       ///
      (v1:, freeparm)       ///
      (v01:, freeparm)      ///
      , maximize ///
      init(b,copy) trace search(on) 
      
      
qui distinct id
scalar Nid = `r(ndistinct)'
scalar Nodes = 7
mlad (xb: = x1, ) ///
     (v0:)       ///
       (v1:)       ///
       (v01:)      ///
       , othervars(y x1) othervarnames(y z) ///
       init(b,copy) ///
       staticscalars(Nid Nodes) llfile(logit_re) id(id)  trace nojit search(off)

      
      
      
melogit y x1 || id: x1, intmethod(ghermite) intpoints(7) cov(unstr) //dnumerical 
