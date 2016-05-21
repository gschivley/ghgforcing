# ghgforcing
Python package to calculating forcing from continuous GHG emissions. All calculations are
based on equations and parameters in the 2013 IPCC AR5 report. As such, the model does not
account for changing background concentrations. But all results are consistant with the 
GWP values published in AR5.

The main functions are `CO2` and `CH4`. It should be easy to add in full functions for
N2O and SF6.

## Methane options
The `CH4` function can account for a couple of different indirect effects:
- Decomposition of fossil CH4 to CO2
- Climate-carbon feedbacks from increased temperatures caused by CH4 emissions

## Uncertainty
A large effort has been put into including uncertainty for every parameter possible. This
includes:
- Uncertainty in the [Joos et al](http://www.atmos-chem-phys.net/13/2793/2013/) CO2 IRF,
calculated by [Olivie and Peters](http://www.earth-syst-dynam.net/4/267/2013/)
- The radiative efficiencies of CO2 and CH4
- The adjusted lifetime of CH4
- The indirect forcing adders to CH4 from tropospheric ozone and stratospheric water vapor
- The fraction of CH4 that oxidizes to CO2
- The size of climate-carbon feedback effects