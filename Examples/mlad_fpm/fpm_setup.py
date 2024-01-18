import jax.numpy as jnp
from sfi import Macro
def mlad_setup(M):
  dnsvars = Macro.getGlobal("dnsvars").split()
  dns = []
  for v in range(len(dnsvars)):
    dns.append(M[dnsvars[v]])
    
  dns.append(jnp.zeros((len(M[dnsvars[1]]),1)))
  dns = (jnp.array(dns).squeeze(axis=2)).T
  M["dns"] = dns
  return(M)
  