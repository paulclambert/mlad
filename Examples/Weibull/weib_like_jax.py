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

def python_grad(beta,X,wt,M):
  lam = jnp.exp(mu.linpred(beta,X,1))
  gam = jnp.exp(mu.linpred(beta,X,2))
  g1 = mu.mlvecsum(M["d"] - lam*M["t"]**gam,X,1)
  g2 = mu.mlvecsum(M["d"]*(1+gam*jnp.log(M["t"])) - lam*M["t"]**(gam)*jnp.log(M["t"])*gam,X,2)
  return(jnp.hstack((g1,g2)))

def python_hessian(beta,X,wt,M):
  lam = jnp.exp(mu.linpred(beta,X,1))
  gam = jnp.exp(mu.linpred(beta,X,2))
  
  d11 = mu.mlmatsum(-lam*M["t"]**gam,X,1,1)
  d12 = mu.mlmatsum((-lam*M["t"]**gam)*jnp.log(M["t"])*gam,X,1,2)
  d22 = mu.mlmatsum(-jnp.log(M["t"])*gam*(lam*M["t"]**gam * jnp.log(M["t"])*gam + 
                                           lam*M["t"]**gam - 
                                           M["d"]),X,2,2)
  
  return(jnp.vstack((jnp.hstack((d11  , d12)),
                     jnp.hstack((d12.T, d22)))))


