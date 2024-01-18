import jax.numpy as jnp   

import importlib
mu = importlib.import_module("mladutil")
mu = importlib.reload(mu)

def python_ll(beta1,beta2,X,d):
  xb =  mu.linpred(beta1,X,1)
  dxb = mu.linpred(beta2,X,2)
  return(jnp.sum(d*(jnp.log(dxb) + xb) - jnp.exp(xb)))



