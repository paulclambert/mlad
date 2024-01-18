from scipy.special import roots_legendre
import jax.numpy as jnp
from jax import vmap, jit
import importlib
mu = importlib.import_module("mladutil")
mu = importlib.reload(mu)

def mlad_setup(M):
  vrcsgen = jit(vmap(mu.rcsgen,(0,None,None),0))
  nodes, weights = roots_legendre(M["Nnodes"])
  
  nodes2 = 0.5*(M["t"] - M["t0"])*nodes + 0.5*(M["t"] + M["t0"])
  M["allnodes"] = vrcsgen(jnp.log(nodes2),M["knots"][0],jnp.ones((1,1)))
  M["weights"] =  weights 
  return(M)
