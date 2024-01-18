import jax.numpy as jnp   
import mladutil as mu
from jax import vmap
from sfi import Data, Macro, Scalar, Matrix
import importlib
mu = importlib.import_module("mladutil")
mu = importlib.reload(mu)


def python_ll(beta1,beta2,v0,X,wt,d,t,id,Nid,Nnodes):
  lnlam =  mu.linpred(beta1,X,1)
  lngam =  mu.linpred(beta2,X,2)
  sigma =  jnp.sqrt(mu.linpred(v0,X,3))
  gam   =  jnp.exp(lngam)

  def calc_lnft_weib(v):
    lp = lnlam + v*jnp.sqrt(2)*sigma 
    return((d*(lp + lngam + (gam-1)*jnp.log(t)) - jnp.exp(lp)*t**(gam))[:,0])  
  
  vect_calc_lnft_weib = vmap(calc_lnft_weib,0,1)
  
  def getllj(v):
    logF = vect_calc_lnft_weib(v)
    return(jnp.exp(mu.sumoverid(id,logF,Nid)))
  
  llj = jnp.log(1/(jnp.sqrt(jnp.pi))*mu.vecquad_gh(getllj,Nnodes,()))   
  Data.store("bob",None,sigma)
  return(jnp.sum(llj))
