************************************************************************
* SURVIVAL_PERIOD.DO
*
* Produce life table estimats of relative survival for each combination
* of sex, period of diagnosis, and age. 
*
* A period approach is used. SURVIVAL.DO uses a cohort approach.
*
* See the strs help file for further information.
*
* Paul Dickman (paul.dickman@ki.se)
* Apr 2004 v1.0
* Oct 2006 v1.1
*
*************************************************************************
use colon, clear
keep if stage==1

/* stset the data with time since diagnosis as the timescale */ 
/* restrict person-time at risk to that within the period window (01jan1994-31dec1995) */
stset exit, enter(time mdy(1,1,1994)) exit(time mdy(12,31,1995)) ///
  origin(dx) f(status==1 2) id(id) scale(365.24)

strs using popmort, br(0(1)10) mergeby(_year sex _age) ///
  by(sex) list(n d p r cr_e2 se_cp)
