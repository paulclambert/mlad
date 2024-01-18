import jax.numpy as jnp   
import mladutil as mu
import importlib
#mu = importlib.import_module("mladutil")
mu = importlib.reload(mu)


def python_ll(beta,X,wt,M):
  lnlam =  mu.linpred(beta,X,1)
  lngam  = mu.linpred(beta,X,2)
  gam = jnp.exp(lngam)

  return(jnp.sum(M["d"]*(lnlam + lngam + (gam - 1)*jnp.log(M["t"])) - jnp.exp(lnlam)*M["t"]**(gam)))



