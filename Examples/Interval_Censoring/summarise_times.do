clear
cd "/media/paul/Storage/GitSoftware/mlad/Examples/Interval_Censoring"

global sampsizes 1000 10000 50000 100000 250000 500000 1000000 2500000 5000000 10000000

local i = 1
foreach ss in $sampsizes {
  if `i'==1 {
    use Results/IC_results_`ss'
  } 
  else {
    append using Results/IC_results_`ss'
  }
  local i = `i' + 1
}

egen mtime = rowmean(time*)
format mtime %6.2f


twoway (line mtime sampsize if method=="jaxjit_pro1") ///
       (connected mtime sampsize if method=="stintreg_pro1") ///
       , 

preserve
  keep if inlist(method,"jaxjit_pro1","stintreg_pro1")
  bysort sampsize (method): gen methnum = _n
  drop time1 time2 method
  reshape wide mtime, i(sampsize) j(methnum)
  gen percentfast = 100*(mtime2-mtime1)/mtime2
  format mtime1 mtime2 %5.1f
  format sampsize %9.0f
  format percentfast %4.1f
  list, noobs sep(2) abbrev(16)
restore       
       
       
       
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
