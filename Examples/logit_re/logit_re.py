import jax.numpy as jnp   
#import mladutil as mu
from jax import vmap
from jax import make_jaxpr
from jax.scipy.linalg import cholesky
from scipy.special import roots_legendre, roots_hermitenorm, roots_hermite
from jax.ops import index_update, index

import importlib
mu = importlib.import_module("mladutil")
mu = importlib.reload(mu)

def python_ll(beta,X,wt,M,Nid,Nnodes):
  xb   =  mu.linpred(beta,X,1)
  sigma0  =  (mu.linpred(beta,X,2))
  sigma1  =  (mu.linpred(beta,X,3))
  sigma01 =  mu.linpred(beta,X,4)
 
  V = jnp.vstack((jnp.hstack((sigma0,sigma01)),jnp.hstack((sigma01,sigma1))))
  C = cholesky(V,lower=True)
  
  Z = jnp.hstack((jnp.ones((10000,1)),M["z"]))
  
  def calc_lnft_logit(v):
    bob = jnp.sqrt(2)*(Z@(C@v.T))
    lp = xb + bob
    return((-jnp.log(1+jnp.exp(lp)) + M["y"]*lp)[:,0])    
  
  vect_calc_lnft_logit = vmap(calc_lnft_logit,(1),1)
  
  def getllj(v):
    logF = vect_calc_lnft_logit(v)
    return(jnp.exp(mu.sumoverid(M["id"],logF,Nid)))
  
  nodes, weights = roots_hermite(Nnodes)
  allnodes = jnp.repeat(jnp.asarray(nodes[:,None]),2,axis=1).T
  allweights = jnp.repeat(jnp.asarray(weights[:,None])/jnp.sqrt(jnp.pi),2,axis=1).T

  
  
  nodescomb = jnp.asarray(jnp.meshgrid(allnodes[0,:],allnodes[1,:])).T.reshape(-1, 2)
  weightscomb = (jnp.asarray(jnp.meshgrid(allweights[0,:],allweights[1,:])).T.reshape(-1, 2)) 
  weights = jnp.prod(weightscomb,axis=1)

  nodescomb = nodescomb[None,:,:]
  llj = jnp.log(jnp.sum(weights*getllj(nodescomb),axis=1,keepdims=True)) 
  return(jnp.sum(llj))



  
