************************************************************************
* SURVIVAL.DO
*
* Produce life table estimats of relative survival for each combination
* of sex, period of diagnosis, and age. The estimates are saved to Stata
* data files (GROUPED.DTA and INDIVID.DTA) for modelling (see MODELS.DO). 
*
* A cohort approach is used. SURVIVAL_PERIOD.DO uses a period approach.
*
* See the strs help file for further information.
*
* Paul Dickman (paul.dickman@ki.se)
* Apr 2004 v 1.0
*
*************************************************************************

use colon, clear
keep if stage==1 /* restrict to localised */

stset surv_mm, fail(status==1 2) id(id) scale(12)

strs using popmort, br(0(1)10) mergeby(_year sex _age) by(sex year8594 agegrp) save(replace)
