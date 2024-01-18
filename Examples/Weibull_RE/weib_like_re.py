import jax.numpy as jnp   
#import mladutil as mu
from jax import vmap
import importlib
mu = importlib.import_module("mladutil")
mu = importlib.reload(mu)



def python_ll(beta,X,wt,M,Nid,Nnodes):
  lnlam =  mu.linpred(beta,X,1)
  lngam =  mu.linpred(beta,X,2)
  sigma =  jnp.sqrt(mu.linpred(beta,X,3))
  gam   =  jnp.exp(lngam)

  def calc_lnft_weib(v):
    lp = lnlam + v*jnp.sqrt(2)*sigma 
    return((M["d"]*(lp + lngam + (gam-1)*jnp.log(M["t"])) - jnp.exp(lp)*M["t"]**(gam))[:,0])  
  
  vect_calc_lnft_weib = vmap(calc_lnft_weib,0,1)
  
  def getllj(v):
    logF = vect_calc_lnft_weib(v)
    return(jnp.exp(mu.sumoverid(M["id"],logF,Nid)))
  
  llj = jnp.log(1/(jnp.sqrt(jnp.pi))*mu.vecquad_gh(getllj,Nnodes,()))   
  return(jnp.sum(llj))
