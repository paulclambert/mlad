import jax.numpy as jnp   
import mladutil as mu

def python_ll(beta,X,wt,M):
  xb   =  mu.linpred(beta,X,1)
  time =  mu.linpred(beta,X,2)
  eta = xb + time
  dtime = jnp.dot(M["dns"], beta[1])[:,None] 
  return(jnp.sum(wt*(M["_d"]*(jnp.log(dtime) + eta) - jnp.exp(eta))))
