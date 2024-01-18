import jax.numpy as jnp   
from   jax import vmap

#import importlib
#mu = importlib.reload(mu)
import mladutil as mu

def python_ll(beta,beta,beta,X,wt,M,Nid,Nnodes):
  xb        =  mu.linpred(beta,X,1)
  dxb       = mu.linpred(beta,X,2)
  sigma     = jnp.sqrt(mu.linpred(beta,X,3))
  
  def calc_lnfpm(v):
    xbv = xb + v*jnp.sqrt(2)*sigma
    return((M["d"]*(-jnp.log(M["t"]) + jnp.log(dxb) + xbv) - jnp.exp(xbv))[:,0])

  vect_calc_lnfpm = vmap(calc_lnfpm,0,1)
 
  def getllj(v):
    logF = vect_calc_lnfpm(v)
    return(jnp.exp(mu.sumoverid(M["id"],logF,Nid)))
  
  llj = jnp.log((1/jnp.sqrt(jnp.pi))* mu.vecquad_gh(getllj,int(Nnodes),()))   
  
  return(jnp.sum(llj))




