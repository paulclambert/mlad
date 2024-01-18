import jax.numpy as jnp
import mladutil as mu

def python_ll(beta, X, wt, M):
  xb = mu.linpred(beta,X,1)
  return(jnp.sum(wt*(M["y"]*xb - jnp.exp(xb))))      
        
def python_grad(beta,X,wt,M):
  xb = mu.linpred(beta,X,1)
  g1 = mu.mlvecsum(wt*(M["y"] - jnp.exp(xb)),X,1)
  return(g1)      
 
def python_hessian(beta,X,wt,M):
  xb = mu.linpred(beta,X,1)
  d11 = mu.mlmatsum(-wt*jnp.exp(xb),X,1,1)
  return(d11)      
 

