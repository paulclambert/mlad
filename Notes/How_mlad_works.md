
# How `mlad` works

This is a brief explanation of how `mlad` works describing how it integrates with Stata’s optimizer `ml`. 

## Stata’s `ml` command
* Stata’s built in optimizer is `ml`. It is used bmy most official and user written estimation commands.
* Most applied users of Stata are not aware of it as it is called from a more user-friendly wrapper command.
* The user needs to write a Stata command (an ado file) that returns as a minimum the log likelihood (or the individual contributions to the log-likelihood). 
* If just the log-likelihood is returned, numerical differentiation is used to obtain the gradient and Hessian matrices.
* Alternatively, the gradient and Hessian matrices can be coded. This will make the estimation command run much faster. 
* I have written `mlad` so that the syntax is similar to `ml`.
* See this [example](https://pclambert.net/software/mlad/weibull_model/) on my webpage to see `ml` being used to fit a Weibull survival model, and how the same modell can be fitted using `mlad`.

## Some background to `mlad`

* `mlad` works by first calling `ml` and then `ml` calling the ado file, `mlad_ll.ado`. I wil explain this file later.
* An example of using mlad is shown below

```Stata
mlad (ln_lambda: = x1 x2)   ///
     (ln_gamma: ),          /// 
      othervars(_t _d)      ///
      othervarnames(t d)    ///
      llfile(weibull_ll)  
```
* This has two equations (linear predictors). 
* `ln_lambda` will be $\beta_0 + \beta_1 x_1 + \beta_2 x_2$
* `ln_gamma` will just be a constant, $\alpha_0$
* The likelihood also needs the survival time and event indicator. These are passed (and renamed) to Python by using the `othervars()` and `othervarnames()` options.
* The `llfile()` option gives the name of the Python file containing the log-likelhood function.
* If we wanted to fit a weibull survival model we write the following Python function in a file called `weibull_ll.py`
  
```python
. type weibull_ll.py
import jax.numpy as jnp   
import mladutil as mu

def python_ll(beta,X,wt,M):
  lnlam =  mu.linpred(beta,X,1)
  lngam  = mu.linpred(beta,X,2)
  gam = jnp.exp(lngam)

  lli = M["d"]*(lnlam + lngam + (gam - 1)*jnp.log(M["t"])) - jnp.exp(lnlam)*M["t"]**(gam)
  return(jnp.sum(lli))
```
* This returns a scalar (the log-likelhood).
* Note I import `jax.numpy` rather than standard `numpy` as I make use of Jax for automatic differentiation and fast compilation. 
* The function name must always be `python_ll`, but of course this function can call other functions.

## The `mlad.ado` file

* This defines the Stata command, `mlad`.
* It defines the command syntax and various options.
* It then calls `ml`, which will then call the `mlad_ll.ado` file

## The `mlad_ll.ado` file

* Stata's `ml` command expects to receive the name of an ado file. This is always `mlad_ll.ado` when using `mlad`.
* When implementing an estimation command with analytical gradients and Hessian, `mlad_ll.ado` will return the log-likelhood, gradient or Hessian depending on the value of `todo`
  * `todo=0`  - log-likelhood
  * `todo=1`  - log-likelhood & gradient
  * `todo=2`  - log-likelhood & gradient & Hessian
* The first time `mlad_ll.ado` is called by `ml` Python is initialized by  by calling the `GetInfo()` function in Python.
  * This function is found in the `mlad_ll.ado` file.
  * It imports the various Python libraries.
  * This includes the `sfi` library, which enables Python to send and receive data from Stata.
  * It reads the required data into Python.
  * It imports the user written file that returns the log-liklihood.
  * It obtains functions for the gradient vector and Hessian matrix using automatic differentiation.
  * It uses fast compilation (jit) for the log-likelihood, gradient and Hessian functions.
  * It will run an aditional Python setup file if requested.
* On all subsequant calls to `mlad_ll.ado`  the Python `calcAll()` function is called.
  * This function is found in the `mlad_ll.ado` file.
  * It will first read the current parameter (beta) matrix from Stata.
  * It will call the compiled log-likelihood, gradient and Hessian functions.
  * It will then return the log-likelhood, gradient and Hessian functions to Stata (sometimes `ml` will only need the log-liklihood and so `todo` saves the function from performing unnecessary calculations).
* When `ml` has received the log-likelhood, gradient and Hessian it will update the beta parameters using  Newton Raphson (other approaches are available if needed). If the model has converged to the specified level of tolerance it will stop, otherwise it will call `mlad_ll.ado` again, which in turn will call the Python `calcAll()` function.
* When the model has converged  `mlad_ll.ado` is called a final time with `todo == "tidy"`, which then deletes data etc from Python.