import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.integrate import cumtrapz
from scipy.stats import multivariate_normal, norm
import pandas as pd
import random


# Radiative efficiencies of each gas, calculated from AR5 & AR5 SM
co2_re, ch4_re, n2o_re, sf6_re = 1.756E-15, 1.277E-13 * 1.65, 3.845E-13, 2.010E-11

# AR5 2013 IRF values
a0, a1, a2, a3 = 0.2173, 0.224, 0.2824, 0.2763
tau1, tau2, tau3 = 394.4, 36.54, 4.304

def f0(t):
    return a0
def f1(t):
    return a1*np.exp(-t/tau1)
def f2(t):
    return a2*np.exp(-t/tau2)
def f3(t):
    return a3*np.exp(-t/tau3)
    

def CO2_AR5(t, **kwargs):
    """ Returns the IRF for CO2 using parameter values from IPCC AR5/Joos et al (2013)
    
    Keyword arguments are parameter values.
    """
    
    a0 = kwargs.get('a0', 0.2173)
    a1 = kwargs.get('a1', 0.224)
    a2 = kwargs.get('a2', 0.2824)
    a3 = kwargs.get('a3', 0.2763)
    tau1 = kwargs.get('tau1', 394.4)
    tau2 = kwargs.get('tau2', 36.54)
    tau3 = kwargs.get('tau3', 4.304)            
    
    IRF = a0 + a1*np.exp(-t/tau1) + a2*np.exp(-t/tau2) + a3*np.exp(-t/tau3)
    return IRF 
    
#Methane response fuction

def CH4_AR5(t, CH4tau = 12.4):
    """IRF for Methane with choice of lifetime.
    CH4tau: adjusted lifetime for methane. uncertainty is +/- 18.57% for 90% CI
    """
    return np.exp(-t/CH4tau)
    
#N2O response fuction
N2Otau = 121
def N2O_AR5(t):
    return np.exp(-t/CH4tau)

#SF6 response fuction   
SF6tau = 3200
def SF6_AR5(t):
    return np.exp(-t/CH4tau)

#Temperature response function to radiative forcing

def AR5_GTP(t):
    c1, c2, d1, d2 = 0.631, 0.429, 8.4, 409.5
    """ The default response function for radiative forcing from AR5. Source is \
    Boucher (2008). ECR is 3.9K, which is on the high side.
    Convolve with radiative forcing to get temperature.
    """
    return c1/d1*np.exp(-t/d1) + c2/d2*np.exp(-t/d2)

def Alt_GTP(t):
    c1, c2, d1, d2 = 0.43, 0.32, 2.57, 82.24
    """ The response function for radiative forcing. Taken from Olivie and Peters (2013),
    Table 4, using the CMIP5 data. This has a slightly lower climate response value than
    Boucher (2008), which is used in AR5.
    Convolve with radiative forcing to get temperature.
    """
    return c1/d1*np.exp(-t/d1) + c2/d2*np.exp(-t/d2)

def Alt_low_GTP(t):
    c1, c2, d1, d2 = 0.43 / (1 + 0.29), 0.32 / (1 + 0.59), 2.57 * 1.46, 82.24 * 2.92
    #c1, c2, d1, d2 = 0.48 * (1 - 0.3), 0.20 * (1 - 0.52), 7.15 * 1.35, 105.55 * 1.38
    #c1, c2, d1, d2 = 0.48 * (1 - 0.3), 0.20 * (1 - 0.52), 7.15, 105.55
    #c1, c2, d1, d2 = 0.631 * 0.7, 0.429 * 0.7, 8.4, 409.5
    """ The response function for radiative forcing. Taken from Olivie and Peters (2013),
    Table 4, using the CMIP5 data. This has a lower climate response value than AR5.
    The uncertainty in Table 4 assumes lognormal distributions, which is why values less
    than the median are determined by dividing by (1 + uncertainty).
    Convolve with radiative forcing to get temperature.
    """
    return c1/d1*np.exp(-t/d1) + c2/d2*np.exp(-t/d2)

def Alt_high_GTP(t):
    c1, c2, d1, d2 = 0.43 * 1.29, 0.32 * 1.59, 2.57 / (1 + 0.46), 82.24 / (1 + 1.92)
    #c1, c2, d1, d2 = 0.48 * 1.3, 0.20 * 1.52, 7.15 * (1 - 0.35), 105.55 * (1 - 0.38)
    #c1, c2, d1, d2 = 0.48 * 1.2, 0.20 * 1.52, 7.15, 105.55
    #c1, c2, d1, d2 = 0.631, 0.429 * 1.3, 8.4, 409.5    
    """ The response function for radiative forcing. Taken from Olivie and Peters (2013),
    Table 4, using the CMIP5 data. This has a higher climate response value than AR5.
    The uncertainty in Table 4 assumes lognormal distributions, which is why values less
    than the median are determined by dividing by (1 + uncertainty).
    Convolve with radiative forcing to get temperature.
    """
    return c1/d1*np.exp(-t/d1) + c2/d2*np.exp(-t/d2)

def CO2_mass(emission, time, tstep=0.01, **kwargs):
    """
    Just the convolution of emission with IRF to calculate mass in atmosphere
    
    """
    a0 = kwargs.get('a0', 0.2173)
    a1 = kwargs.get('a1', 0.224)
    a2 = kwargs.get('a2', 0.2824)
    a3 = kwargs.get('a3', 0.2763)
    tau1 = kwargs.get('tau1', 394.4)
    tau2 = kwargs.get('tau2', 36.54)
    tau3 = kwargs.get('tau3', 4.304)
    
    
    
    atmos = np.resize(fftconvolve(CO2_AR5(time, **co2_kwargs), 
                  emission), time.size) * tstep
                  
    return atmos
    

def CO2(emission, years, tstep=0.01, kind='RF', interpolation='linear', source='AR5',
         runs=1, RS=1, full_output=False, **kwargs):
    """
    a0=0.2173, a1=0.224, a2=0.2824, a3=0.2763, tau1=394.4, tau2=36.54, tau3=4.304,
        RE=1.756E-15,
    Transforms an array of CO2 emissions into radiative forcing, CRF, or temperature
    with user defined time-step.
    
    Parameters:
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: RF, CRF, or temp
    interpolation: the type of interpolation to use; can be linear or cubic
    source: the source of parameters for the temperature IRF. default is AR5,
    'Alt', 'Alt_low', and 'Alt_high' are also options.
    runs: Number of runs for monte carlo. A single run will return a numpy array, multiple
    runs will return a pandas dataframe with columns "mean", "+sigma", and "-sigma".
    RS: Random state initiator for continuity between calls
    full_output: When True, outputs the results from all runs as an array in addition to
    the mean and +/- sigma as a DataFrame
    
    Keyword arguments are used to pass random IRF parameter values for a single run as
    part of a larger monte carlo calculation (currently limited to CH4 decomposing to
    CO2 in the *CH4* function).
    
    Returns:
    output: When runs=1, deterministic RF, CRF, or temp. When runs > 1 and full_output is
            False, returns a dataframe with 'mean', '+sigma', and '-sigma' columns.
    output, full_output: Only returned when full_output=True. Both the dataframe with
            'mean', '+sigma', and '-sigma' columns, and results from all MC runs.
    """
    if min(years) > 0:
        years = years - min(years)
    
    end = max(years) 
    f = interp1d(years, emission, kind=interpolation)
    time = np.linspace(years[0], end, end/tstep + 1)    
    inter_emissions = f(time)
    results = np.zeros((len(time), runs))
    count = 0
    slice_step = int(1/tstep)
    
    if runs > 1:
        
        # sigma and x are from Olivie and Peters (2013) Table 5 (J13 values)
        # They are the covariance and mean arrays for CO2 IRF uncertainty
        sigma = np.array([[0.129, -0.058, 0.017,	-0.042,	-0.004,	-0.009],
                        [-0.058, 0.167,	-0.109,	0.072,	-0.015,	0.003],
                        [0.017,	-0.109,	0.148,	-0.043,	0.013,	-0.013],
                        [-0.042, 0.072,	-0.043,	0.090,	0.009,	0.006],
                        [-0.004, -0.015, 0.013,	0.009,	0.082,	0.013],
                        [-0.009, 0.003,	-0.013,	0.006,	0.013,	0.046]])
    
        x = np.array([5.479, 2.913,	0.496, 0.181, 0.401, -0.472])
    
        data = multivariate_normal.rvs(x,sigma, runs, random_state=RS)
    
        data_df = pd.DataFrame(data, columns=['t1', 't2', 't3', 'b1','b2','b3'])
    
        df_exp = np.exp(data_df)
    
        a0 = 1 / (1 + df_exp['b1'] + df_exp['b2'] + df_exp['b3'])
        a1 = df_exp['b1'] / (1 + df_exp['b1'] + df_exp['b2'] + df_exp['b3'])
        a2 = df_exp['b2'] / (1 + df_exp['b1'] + df_exp['b2'] + df_exp['b3'])
        a3 = df_exp['b3'] / (1 + df_exp['b1'] + df_exp['b2'] + df_exp['b3'])
    
        #a = pd.concat([A0, A1, A2, A3], axis=1)
        
        # 90% CI is +/- 10% of mean. Divide by 1.64 to find sigma
        RE = norm.rvs(1.756e-15, 1.756e-15 * .1 / 1.64, size=runs, random_state=RS+1)
    
        tau1=df_exp['t1'].values
        tau2=df_exp['t2'].values
        tau3=df_exp['t3'].values
    
    
        for count in np.arange(runs): #Is there a way to do this in parallel?
            co2_kwargs = {'a1' : a1[count], 
                        'a2' : a2[count], 
                        'a3' : a3[count],
                        'tau1' : tau1[count],
                        'tau2' : tau2[count],
                        'tau3' : tau3[count]}
            CO2_re = RE[count]
            
            atmos = np.resize(fftconvolve(CO2_AR5(time, **co2_kwargs), 
                              inter_emissions), time.size) * tstep
            rf = atmos * CO2_re
            
            
            if kind == 'temp':
                temp = np.resize(fftconvolve(AR5_GTP(time), rf), time.size) * tstep
                results[:,count] = temp
                #continue
            else:
                results[:,count] = rf
            
            count += 1
        
        if full_output == False:
            output = pd.DataFrame(columns = ['mean', '-sigma', '+sigma'])    
            if kind == 'CRF':
                crf = cumtrapz(results, dx = tstep, initial = 0, axis=0)
                output['mean'] = np.mean(crf[0::slice_step], axis=1)
                output['-sigma'] = output['mean'] - np.std(crf[0::slice_step], axis=1)
                output['+sigma'] = output['mean'] + np.std(crf[0::slice_step], axis=1)
            
            elif kind == 'RF' or 'temp':
                output['mean'] = np.mean(results[0::slice_step], axis=1)
                output['-sigma'] = output['mean'] - np.std(results[0::slice_step], axis=1)
                output['+sigma'] = output['mean'] + np.std(results[0::slice_step], axis=1)

            #elif kind == 'CRF':
            #    crf = cumtrapz(results, dx = tstep, initial = 0, axis=0)
            #    output['mean'] = np.mean(crf[0::slice_step], axis=1)
            #    output['-sigma'] = output['mean'] - np.std(crf[0::slice_step], axis=1)
            #    output['+sigma'] = output['mean'] + np.std(crf[0::slice_step], axis=1)
                    
            return output
        
        else:
            output = pd.DataFrame(columns = ['mean', '-sigma', '+sigma'])    
            if kind == 'CRF':
                crf = cumtrapz(results, dx = tstep, initial = 0, axis=0)
                output['mean'] = np.mean(crf[0::slice_step], axis=1)
                output['-sigma'] = output['mean'] - np.std(crf[0::slice_step], axis=1)
                output['+sigma'] = output['mean'] + np.std(crf[0::slice_step], axis=1)
                
                full_output = crf[0::slice_step]
            
            elif kind == 'RF' or 'temp':
                output['mean'] = np.mean(results[0::slice_step], axis=1)
                output['-sigma'] = output['mean'] - np.std(results[0::slice_step], axis=1)
                output['+sigma'] = output['mean'] + np.std(results[0::slice_step], axis=1)
                
                full_output = results[0::slice_step]

            #elif kind == 'CRF':
            #    crf = cumtrapz(results, dx = tstep, initial = 0, axis=0)
            #    output['mean'] = np.mean(crf[0::slice_step], axis=1)
            #    output['-sigma'] = output['mean'] - np.std(crf[0::slice_step], axis=1)
            #    output['+sigma'] = output['mean'] + np.std(crf[0::slice_step], axis=1)
                
            #    full_output = crf[0::slice_step]
                
            return output, full_output
            
                 
    else:
        CO2_re=1.756E-15            
        atmos = np.resize(fftconvolve(CO2_AR5(time, **kwargs), 
                            inter_emissions), time.size) * tstep
        rf = atmos * CO2_re
        
        if kind == 'RF':
            return rf[0::slice_step]
        elif kind == 'CRF':
            crf = cumtrapz(rf, dx = tstep, initial = 0)
            return crf[0::slice_step]
        elif kind == 'temp':
            if source == 'AR5':
                temp = np.resize(fftconvolve(AR5_GTP(time), rf), time.size) * tstep
            elif source == 'Alt':
                temp = np.resize(fftconvolve(Alt_GTP(time), rf), time.size) * tstep
            elif source == 'Alt_low':
                temp = np.resize(fftconvolve(Alt_low_GTP(time), rf), 
                                time.size) * tstep
            elif source == 'Alt_high':
                temp = np.resize(fftconvolve(Alt_high_GTP(time), rf), 
                                time.size) * tstep
            return temp[0::slice_step]
        

def ch42co2(t, CH4tau=12.4, alpha=0.51):
    """As methane decays some fraction is converted to CO2. This function is 
    from Boucher (2009). By default it converts 51%. The convolution of this 
    function with the methane emission profile gives the CO2 emission profile.
    
    t: time
    alpha: fraction of methane converted to CO2
    """
    #ch4tau = 12.4
    return 1/CH4tau * alpha * np.exp(-t/CH4tau)


def CH4(emission, years, tstep=0.01, kind='RF', interpolation='linear', source='AR5',
        cc_fb = True, decay=True, CH4tau = 12.4, RE=1.277E-13 * 1.65, runs=1, RS=1,
        full_output=False):
    """Transforms an array of CO2 emissions into radiative forcing, CRF, or temperature
    with user defined time-step. Still need to set up cc_fb for monte carlo. For MC,
    all variable inputs should come in with the same number of values as "runs", and 
    already be randomly distributed.
    
    
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: RF, CRF, or temp
    interpolation: the type of interpolation to use; can be linear or cubic
    source: the source of parameters for the temperature IRF. default is AR5,
    'Alt', 'Alt_low', and 'Alt_high' are also options.
    cc_fb: True if climate-carbon cycle feedbacks are included.
    decay: True if methane is fossil-based, will include decay to CO2.
    CH4tau: adjusted lifetime for methane. uncertainty is +/- 18.57% for 90% CI. Use tuple
    for multiple values.
    RE: Radiative efficiency of methane, including the 15% and 50% adders for indirect
    effects on water vapor and ozone.
    runs: number of runs for monte carlo
    RS: Random state initiator for continuity between calls
    full_output: When True, outputs the results from all runs as an array in addition to
    the mean and +/- sigma as a DataFrame
    
    Returns:
    output: When runs=1, deterministic RF, CRF, or temp. When runs > 1 and full_output is
            False, returns a dataframe with 'mean', '+sigma', and '-sigma' columns.
    output, full_output: Only returned when full_output=True. Both the dataframe with
            'mean', '+sigma', and '-sigma' columns, and results from all MC runs.
    """
    
	# Gamma is the carbon released per K temperature increase - Collins et al (2013)
    gamma = (44.0/12.0) * 10**12

	# This is a holdover from some models with emissions in non-consecutive years
    if min(years) > 0:
        years = years - min(years)
    end = max(years) 
    time = np.linspace(years[0], end, end/tstep + 1)
    
    #if not isinstance(emission, pd.DataFrame):
    #    emission = pd.DataFrame(emission)
    
    results = np.zeros((len(time), runs))

    slice_step = int(1/tstep)
            
    #Attempting to account for uncertainty in cc-fb, which is +/- 100% through a triang
    #distribution. This is based on the footnote of AR5 Table 8.7 that the uncertainties
    #in magnitude of cc-fb are comparable in size to the effect.
    ccfb_dist = sp.stats.triang.rvs(1, scale=2, size=runs, random_state=RS)
    
 
    
      
    if runs == 1:
        co2_re = 1.756E-15
        ch4_re = 1.277E-13 * 1.65
    
                
    count = 0
    if decay == True: # CH4 to CO2 decay
        if runs > 1: # More than one run, so use MC
        
            tau = norm.rvs(12.4, 1.4, size=runs, random_state=RS)
           
            # 90% CI is +/- 60% of mean. Divide by 1.64 to find sigma
            f1 = norm.rvs(0.5, 0.5 * 0.6 / 1.64, size=runs, random_state=RS+1) 
            
            # 90% CI is +/- 71.43% of mean. Divide by 1.64 to find sigma
            f2 = norm.rvs(0.15, 0.15 * 0.7143 / 1.64, size=runs, random_state=RS+2) 
            
            # 90% CI is +/- 10% of mean. Divide by 1.64 to find sigma
            RE = norm.rvs(1.277E-13, 1.277E-13 * 0.1 / 1.64, size=runs, random_state=RS+3)
            RE_total = RE * (1 + f1 + f2)
            
            # 90% CI is +/- 10% of mean. Divide by 1.64 to find sigma
            CO2RE = norm.rvs(1.756e-15, 1.756e-15 * 0.1 / 1.64, size=runs, 
                            random_state=RS+4)
                            
            #Uncertainty in CH4 decomposition to CO2. Boucher et al (2009) use a lower 
            #bound of 51% and an upper bound of 100%. GWP calculations in AR5 assume 51% 
            #(personal communication - find email to cite). Using a uniform distribution 
            #here for now.
            alpha_dist = sp.stats.uniform.rvs(loc=0.51, scale=0.49, 
                                              size=runs, random_state=RS)
            
            # sigma and x are from Olivie and Peters (2013) Table 5 (J13 values)
            # They are the covariance and mean arrays for CO2 IRF uncertainty
            sigma = np.array([[0.129, -0.058, 0.017, -0.042, -0.004, -0.009],
                            [-0.058, 0.167,	-0.109,	0.072, -0.015,	0.003],
                            [0.017,	-0.109,	0.148,	-0.043,	0.013,	-0.013],
                            [-0.042, 0.072,	-0.043,	0.090,	0.009,	0.006],
                            [-0.004, -0.015, 0.013,	0.009,	0.082,	0.013],
                            [-0.009, 0.003,	-0.013,	0.006,	0.013,	0.046]])
                            
            x = np.array([5.479, 2.913,	0.496, 0.181, 0.401, -0.472])

            data = multivariate_normal.rvs(x,sigma, runs, random_state=RS)
            data_df = pd.DataFrame(data, columns=['t1', 't2', 't3', 'b1','b2','b3'])
            df_exp = np.exp(data_df)
            
            a0 = 1 / (1 + df_exp['b1'] + df_exp['b2'] + df_exp['b3'])
            a1 = df_exp['b1'] / (1 + df_exp['b1'] + df_exp['b2'] + df_exp['b3'])
            a2 = df_exp['b2'] / (1 + df_exp['b1'] + df_exp['b2'] + df_exp['b3'])
            a3 = df_exp['b3'] / (1 + df_exp['b1'] + df_exp['b2'] + df_exp['b3'])

            tau1=df_exp['t1'].values
            tau2=df_exp['t2'].values
            tau3=df_exp['t3'].values
            
            while count < runs:
                
                #Random choice of emission scenario where more than one is available
                emiss = emission#[random.choice(emission.columns)]
            
				# Use sequential values of tau and RE, so they are the same between runs
                ch4_tau = tau[count]
                ch4_re = RE_total[count]
                co2_re = CO2RE[count]
                
                #CO2 IRF parameter values
                co2_kwargs = {'a0' : a0[count], 
                              'a1' : a1[count], 
                              'a2' : a2[count], 
                              'a3' : a3[count],
                              'tau1' : tau1[count],
                              'tau2' : tau2[count],
                              'tau3' : tau3[count]}
                
                #Percent of CH4 that decays to CO2
                alpha = alpha_dist[count]
                
				# Calculation of CH4 and CO2 in atmosphere over time.
                ch4_atmos = np.resize(fftconvolve(CH4_AR5(time, ch4_tau), emiss),
                                  time.size) * tstep
                co2 = np.resize(fftconvolve(ch42co2(time, ch4_tau, alpha), emiss),
                            time.size) * tstep
                
                #I've now included uncertainty here, but the code is pretty sloppy. Need
                #to clean it up soon, maybe use the *CO2* function to include uncertainty
                #rather than copying the multivariate normal stuff in here.                              
                co2_atmos = np.resize(fftconvolve(CO2_AR5(time, **co2_kwargs), co2),
                                  time.size) * tstep
            
                # Forcing from CH4 and CO2
                rf = ch4_atmos * ch4_re + co2_atmos * co2_re
            
                # Additional forcing from cc-fb
                if cc_fb == True: #I need to set up cc_fb for MC still
				    #Accounting for uncertainty through normal distribution
                    cc_co2 = CH4_cc_tempforrf(emiss, time) * gamma * ccfb_dist[count]
                    cc_co2_atmos = np.resize(fftconvolve(CO2_AR5(time), cc_co2),
                                      time.size) * tstep
                    rf += cc_co2_atmos * co2_re

                #Store single run of results, or calculate temp
                if kind == 'temp':
                    temp = np.resize(fftconvolve(AR5_GTP(time), rf), time.size) * tstep
                    results[:,count] = temp
                    #continue
                else:
					results[:,count] = rf
                

                count += 1
            
            #Calculate output, which is mean and +/- 1 sigma
            if full_output == False:
                output = pd.DataFrame(columns = ['mean', '-sigma', '+sigma'])    
                if kind == 'CRF':
                    crf = cumtrapz(results, dx = tstep, initial = 0, axis=0)
                    output['mean'] = np.mean(crf[0::slice_step], axis=1)
                    output['-sigma'] = output['mean'] - np.std(crf[0::slice_step], axis=1)
                    output['+sigma'] = output['mean'] + np.std(crf[0::slice_step], axis=1)
                
                elif kind == 'RF' or 'temp':
                    output['mean'] = np.mean(results[0::slice_step], axis=1)
                    output['-sigma'] = output['mean'] - np.std(results[0::slice_step],
                                                                axis=1)
                    output['+sigma'] = output['mean'] + np.std(results[0::slice_step], 
                                                                axis=1)

                #elif kind == 'CRF':
                #    crf = cumtrapz(results, dx = tstep, initial = 0, axis=0)
                #   output['mean'] = np.mean(crf[0::slice_step], axis=1)
                #    output['-sigma'] = output['mean'] - np.std(crf[0::slice_step], axis=1)
                #    output['+sigma'] = output['mean'] + np.std(crf[0::slice_step], axis=1)
                
                return output
    
            elif full_output == True:
                output = pd.DataFrame(columns = ['mean', '-sigma', '+sigma'])    
                if kind == 'CRF':
                    crf = cumtrapz(results, dx = tstep, initial = 0, axis=0)
                    output['mean'] = np.mean(crf[0::slice_step], axis=1)
                    output['-sigma'] = output['mean'] - np.std(crf[0::slice_step], axis=1)
                    output['+sigma'] = output['mean'] + np.std(crf[0::slice_step], axis=1)
                    
                    full_output = crf[0::slice_step]
                
                elif kind == 'RF' or 'temp':
                    output['mean'] = np.mean(results[0::slice_step], axis=1)
                    output['-sigma'] = output['mean'] - np.std(results[0::slice_step], 
                                                                axis=1)
                    output['+sigma'] = output['mean'] + np.std(results[0::slice_step], 
                                                                axis=1)
            
                    full_output = results[0::slice_step]

                #elif kind == 'CRF':
                #    crf = cumtrapz(results, dx = tstep, initial = 0, axis=0)
                #    output['mean'] = np.mean(crf[0::slice_step], axis=1)
                #    output['-sigma'] = output['mean'] - np.std(crf[0::slice_step], axis=1)
                #    output['+sigma'] = output['mean'] + np.std(crf[0::slice_step], axis=1)
            
                #    full_output = crf[0::slice_step]
            
                return output, full_output
        
        # Single run, no MC
        else:
            ch4_re = RE
            
            ch4_atmos = np.resize(fftconvolve(CH4_AR5(time), emission),
                                  time.size) * tstep
            co2 = np.resize(fftconvolve(ch42co2(time, CH4tau), emission),
                            time.size) * tstep
            co2_atmos = np.resize(fftconvolve(CO2_AR5(time), co2),
                                  time.size) * tstep
            
            rf = ch4_atmos * ch4_re + co2_atmos * co2_re
            
            if cc_fb == True: #I need to set up cc_fb for MC still
                cc_co2 = CH4_cc_tempforrf(emission, time) * gamma
                cc_co2_atmos = np.resize(fftconvolve(CO2_AR5(time), cc_co2),
                                  time.size) * tstep
                rf += cc_co2_atmos * co2_re
            
            
            if kind == 'RF':
                output = rf[0::slice_step]
            elif kind == 'temp':
				temp = np.resize(fftconvolve(AR5_GTP(time), rf), time.size) * tstep
				output = temp[0::slice_step]
            elif kind == 'CRF':
                crf = cumtrapz(rf, dx = tstep, initial = 0)
                output = crf[0::slice_step]
            
            return output
    
	#No CH4 decay to CO2 (biogenic CH4 source)          
    else:
        if runs > 1: #Multiple runs, so use MC
            
            tau = norm.rvs(12.4, 1.4, size=runs, random_state=RS)
           
            # 90% CI is +/- 60% of mean. Divide by 1.64 to find sigma
            f1 = norm.rvs(0.5, 0.5 * 0.6 / 1.64, size=runs, random_state=RS+1) 
            
            # 90% CI is +/- 71.43% of mean. Divide by 1.64 to find sigma
            f2 = norm.rvs(0.15, 0.15 * 0.7143 / 1.64, size=runs, random_state=RS+2) 
            
            # 90% CI is +/- 10% of mean. Divide by 1.64 to find sigma
            RE = norm.rvs(1.277E-13, 1.277E-13 * 0.1 / 1.64, size=runs, random_state=RS+3)
            RE_total = RE * (1 + f1 + f2)
            
            # 90% CI is +/- 10% of mean. Divide by 1.64 to find sigma
            CO2RE = norm.rvs(1.756e-15, 1.756e-15 * 0.1 / 1.64, size=runs, 
                            random_state=RS+4)
            
            
            while count < runs:
            
                #Random choice of emission scenario where more than one is available
                emiss = emission#[random.choice(emission.columns)]
            
				
                # Use sequential values of tau and RE, so they are the same between runs
                ch4_tau = tau[count]
                ch4_re = RE_total[count]
                co2_re = CO2RE[count]
				
				# CH4 in atmosphere, and calculation of forcing
                ch4_atmos = np.resize(fftconvolve(CH4_AR5(time, ch4_tau), emiss),
                                  time.size) * tstep
            
                rf = ch4_atmos * ch4_re
            
                # Additional CO2 emissions from cc-fb
                if cc_fb == True: #I need to set up cc_fb for MC still
				    #Accounting for uncertainty through normal distribution
                    cc_co2 = CH4_cc_tempforrf(emiss, time) * gamma * ccfb_dist[count]
                    cc_co2_atmos = np.resize(fftconvolve(CO2_AR5(time), cc_co2),
                                      time.size) * tstep
                    rf += cc_co2_atmos * co2_re
                
                #Store single run of results, or calculate temp
                if kind == 'temp':
                    temp = np.resize(fftconvolve(AR5_GTP(time), rf), time.size) * tstep
                    results[:,count] = temp
                    #continue
                else:
					results[:,count] = rf
                

                count += 1
            
			#Calculate output, which is mean and +- 1 sigma
            if full_output == False:
                output = pd.DataFrame(columns = ['mean', '-sigma', '+sigma'])    
                if kind == 'CRF':
                    crf = cumtrapz(results, dx = tstep, initial = 0, axis=0)
                    output['mean'] = np.mean(crf[0::slice_step], axis=1)
                    output['-sigma'] = output['mean'] - np.std(crf[0::slice_step], axis=1)
                    output['+sigma'] = output['mean'] + np.std(crf[0::slice_step], axis=1)
                
                elif kind == 'RF' or 'temp':
                    output['mean'] = np.mean(results[0::slice_step], axis=1)
                    output['-sigma'] = output['mean'] - np.std(results[0::slice_step],
                                                                axis=1)
                    output['+sigma'] = output['mean'] + np.std(results[0::slice_step], 
                                                                axis=1)

                #elif kind == 'CRF':
                #    crf = cumtrapz(results, dx = tstep, initial = 0, axis=0)
                #    output['mean'] = np.mean(crf[0::slice_step], axis=1)
                #    output['-sigma'] = output['mean'] - np.std(crf[0::slice_step], axis=1)
                #    output['+sigma'] = output['mean'] + np.std(crf[0::slice_step], axis=1)
                
                return output
    
            elif full_output == True:
                output = pd.DataFrame(columns = ['mean', '-sigma', '+sigma'])    
                if kind == 'CRF':
                    crf = cumtrapz(results, dx = tstep, initial = 0, axis=0)
                    output['mean'] = np.mean(crf[0::slice_step], axis=1)
                    output['-sigma'] = output['mean'] - np.std(crf[0::slice_step], axis=1)
                    output['+sigma'] = output['mean'] + np.std(crf[0::slice_step], axis=1)
            
                    full_output = crf[0::slice_step]
                
                elif kind == 'RF' or 'temp':
                    output['mean'] = np.mean(results[0::slice_step], axis=1)
                    output['-sigma'] = output['mean'] - np.std(results[0::slice_step], 
                                                                axis=1)
                    output['+sigma'] = output['mean'] + np.std(results[0::slice_step], 
                                                                axis=1)
            
                    full_output = results[0::slice_step]

                #elif kind == 'CRF':
                #    crf = cumtrapz(results, dx = tstep, initial = 0, axis=0)
                #    output['mean'] = np.mean(crf[0::slice_step], axis=1)
                #    output['-sigma'] = output['mean'] - np.std(crf[0::slice_step], axis=1)
                #    output['+sigma'] = output['mean'] + np.std(crf[0::slice_step], axis=1)
            
                #    full_output = crf[0::slice_step]
            
                return output, full_output
                
        # No CH4 decay, no MC
        else:
            ch4_re = RE
            
            ch4_atmos = np.resize(fftconvolve(CH4_AR5(time, CH4tau), emission),
                                  time.size) * tstep
            
            rf = ch4_atmos * ch4_re
            
            if cc_fb == True: #I need to set up cc_fb for MC still
                cc_co2 = CH4_cc_tempforrf(emission, time) * gamma
                cc_co2_atmos = np.resize(fftconvolve(CO2_AR5(time), cc_co2),
                                  time.size) * tstep
                rf += cc_co2_atmos * co2_re
            
            # Set output
            if kind == 'RF':
                output = rf[0::slice_step]
            elif kind == 'temp':
				temp = np.resize(fftconvolve(AR5_GTP(time), rf), time.size) * tstep
				output = temp[0::slice_step]
            elif kind == 'CRF':
                crf = cumtrapz(rf, dx = tstep, initial = 0)
                output = crf[0::slice_step]
    
            return output


def CH4_cc_tempforrf(emission, years, tstep=0.01, kind='linear', source='AR5',
             decay=True): 
    """Transforms an array of methane emissions into temperature with user-defined
    time-step. Default temperature IRF is from AR5, use 'Alt_low' or 'Alt_high'
    for a sensitivity test.
    
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: the type of interpolation to use; can be linear or cubic
    source: the source of parameters for the temperature IRF. default is AR5,
    'Alt', 'Alt_low', and 'Alt_high' are also options.
    decay: a boolean variable for if methane decay to CO2 should be included
    """
    if min(years) > 0:
        years = years - min(years)
    
    end = max(years) 
    f = interp1d(years, emission, kind=kind)
    time = np.linspace(years[0], end, end/tstep + 1)    
    ch4_inter_emissions = f(time)
    ch4_atmos = np.resize(fftconvolve(CH4_AR5(time), ch4_inter_emissions),
                          time.size) * tstep
    co2 = np.resize(fftconvolve(ch42co2(time), ch4_inter_emissions),
                    time.size) * tstep
    co2_atmos = np.resize(fftconvolve(CO2_AR5(time), co2),
                          time.size) * tstep
    if decay == True:
         rf = ch4_atmos * ch4_re + co2_atmos * co2_re
    else:
        rf = ch4_atmos * ch4_re
    if source == 'AR5':
        temp = np.resize(fftconvolve(AR5_GTP(time), rf), time.size) * tstep
    elif source == 'Alt':
        temp = np.resize(fftconvolve(Alt_GTP(time), rf), time.size) * tstep
    elif source == 'Alt_low':
        temp = np.resize(fftconvolve(Alt_low_GTP(time), rf), time.size) * tstep
    elif source == 'Alt_high':
        temp = np.resize(fftconvolve(Alt_high_GTP(time), rf), time.size) * tstep
    
    return temp