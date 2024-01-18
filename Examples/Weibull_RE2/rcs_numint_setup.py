from scipy.special import roots_legendre
import jax.numpy as jnp

def mlad_setup(M):

  ## hazard function
  def lnrcshaz(t):
    vrcsgen = vmap(mu.rcsgen,(0,None,None))
    return(jnp.exp(vrcsgen(jnp.log(t),M["knots"][0],beta[1]) + xb))


  M['Z'] = jnp.hstack((jnp.ones((M["z1"].shape[0],1)),M["z1"]))
  nodes, weights = roots_hermite(M["Nnodes"])
  allnodes = jnp.repeat(jnp.asarray(jnp.sqrt(2)*nodes[:,None]),2,axis=1).T
  M['nodes'] = (jnp.asarray(jnp.meshgrid(allnodes[0,:],allnodes[1,:])).T.reshape(-1, 2))
  M['nodes'] = jnp.asarray(M['nodes'])[:,:,None]
  
  allweights = jnp.repeat(jnp.asarray(weights[:,None])/jnp.sqrt(jnp.pi),2,axis=1).T
  weightscomb = (jnp.asarray(jnp.meshgrid(allweights[0,:],allweights[1,:])).T.reshape(-1, 2)) 
  M['weights'] = jnp.prod(weightscomb,axis=1)
  return(M)
