import jax.numpy as jnp   
import mladutil as mu

def python_ll(beta,X,wt,M):
  xb =  mu.linpred(beta,X,1)
  dxb = mu.linpred(beta,X,2)
  expxb = jnp.exp(xb)
  return(jnp.sum(wt*M["d"]*jnp.log(M["rate"] + (dxb)*expxb/M["t"]) - expxb))



	
