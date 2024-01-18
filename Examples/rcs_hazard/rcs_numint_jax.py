## likelihood function for splines on the log hazard scale

import jax.numpy as jnp   
from   jax import vmap
import mladutil as mu

def python_ll(beta,X,wt,M,Nnodes):
  ## Parameters
  xb    = mu.linpred(beta,X,1)
  xbrcs = mu.linpred(beta,X,2)

  ## hazard function
  def rcshaz(t):
    vrcsgen = vmap(mu.rcsgen_beta,(0,None,None))
    return(jnp.exp(vrcsgen(jnp.log(t),M["knots"][0],beta[1]) + xb))

  ## cumulative hazard
  cumhaz = mu.vecquad_gl(rcshaz,M["t0"],M["t"],Nnodes,())   

  ## return likelhood
  return(jnp.sum(wt*(M["d"]*(xb + xbrcs) - cumhaz)))

