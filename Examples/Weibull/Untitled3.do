clear
cd "/media/paul/Storage/GitSoftware/mlad/Examples/Weibull"

global sampsizes 1000 10000 50000 100000 250000 500000 1000000 2500000 5000000

local i = 1
foreach ss in $sampsizes {
  if `i'==1 {
    use Results/weib_results_`ss'
  } 
  else {
    append using Results/weib_results_`ss'
  }
  local i = `i' + 1
}

egen mtime = rowmean(time*)
format mtime %6.2f

list sampsize method  mtime if method == "jaxjit_pro2", noobs 


local SS
levelsof method
local i = 1
summ mtime if sampsize == ${SS}, meanonly
local xtext `r(max)'*1.02
foreach m in `r(levels)' {
  local text `text' text(`i' `xtext' "bbb")   
  

  local i = `i' +1
} 


graph hbar (asis) mtime if sampsize == 50000, over(method)
