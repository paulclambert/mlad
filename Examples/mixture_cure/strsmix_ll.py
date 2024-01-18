import jax.numpy as jnp
import mladutil as mu

def python_ll(beta, X, wt, M):
  pi    = mu.invlogit(mu.linpred(beta,X,1))
  lam   = jnp.exp(mu.linpred(beta,X,2))
  gam   = jnp.exp(mu.linpred(beta,X,3))

  ftc = mu.weibdens(M["t"],lam,gam)
  Stc = mu.weibsurv(M["t"],lam,gam)
  
  ht = (1-pi)*(ftc)/(pi + (1-pi)*(Stc))
  St = (pi + (1-pi)*(Stc))
  
  return(jnp.sum(wt*(M["d"]*jnp.log(M["rate"] + ht) + jnp.log(St))))
    
