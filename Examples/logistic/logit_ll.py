import jax.numpy as jnp   
from   jax.ops import index
from   jax.scipy.stats import bernoulli
from   jax import vmap

import importlib
mu = importlib.import_module("mladutil")
mu = importlib.reload(mu)

def python_ll(beta,X,wt,M,Nid,Nnodes,bob)
  xb        = mu.linpred(beta,X,1)
  sigma = jnp.sqrt(mu.linpred(beta,X,2))
  
  def calc_lnpmf(v):
    xbv = xb + v*jnp.sqrt(2)*sigma
    return((-jnp.log(1+jnp.exp(xbv)) + M["y"]*xbv)[:,0])    

  vect_calc_lnpmf = vmap(calc_lnpmf,0,1)
 
  def getllj(v):
    logF = vect_calc_lnpmf(v)
    return(jnp.exp(mu.sumoverid(M["id"],logF,Nid)))
  
  llj = jnp.log(1/(jnp.sqrt(jnp.pi))*mu.vecquad_gh(getllj,int(Nnodes),()))   
  return(jnp.sum(llj))












