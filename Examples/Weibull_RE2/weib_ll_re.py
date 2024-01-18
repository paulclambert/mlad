import jax.numpy as jnp   
#import mladutil as mu
from jax import vmap
from jax.numpy.linalg import cholesky

import importlib
mu = importlib.import_module("mladutil")
mu = importlib.reload(mu)

def python_ll(beta,X,wt,M,Nid):
  lnlam   =  mu.linpred(beta,X,1)
  lngam   =  mu.linpred(beta,X,2)
  sigma0  =  mu.linpred(beta,X,3)
  sigma1  =  mu.linpred(beta,X,4)
  sigma01 =  mu.linpred(beta,X,5)
  gam     =  jnp.exp(lngam)
 
  V = jnp.vstack((jnp.hstack((sigma0,sigma01)),jnp.hstack((sigma01,sigma1))))
  C = cholesky(V)
  
  
  
  def calc_lnft_weib(v):
    lp = lnlam + M["Z"]@C@v
    return((M["d"]*(lp + lngam + (gam-1)*jnp.log(M["t"])) - jnp.exp(lp)*M["t"]**(gam))[:,0])  
  
  vect_calc_lnft_weib = vmap(calc_lnft_weib,(0),1)
  
  def getllj(v):
    logF = vect_calc_lnft_weib(v)
    return(jnp.exp(mu.sumoverid(M["id"],logF,Nid)))

  llj = jnp.log(jnp.sum(M["weights"]*getllj(M["nodes"]),axis=1,keepdims=True)) 
  return(jnp.sum(llj))

  
  
  
