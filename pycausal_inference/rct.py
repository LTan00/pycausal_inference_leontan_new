import numpy as np
import pandas as pd
import scipy.stats as stats
from math import sqrt

def calculate_ate_ci(data, alpha=0.05):
    
    "Calculate ATE_estimate E[Y|T = t] − E[Y|T = 0]"
    
    treated_mean = np.mean(data[data['T'] == 1]["Y"])
    non_treated_mean = np.mean(data[data['T'] == 0]["Y"])
    
    ATE_estimate = treated_mean - non_treated_mean
    
    "Calculate Variance and n1, n0"
    
    treated_var = data[data['T'] == 1]["Y"].var()
    n1 = data[data['T'] == 1]["Y"].count()

    non_treated_var = data[data['T'] == 0]["Y"].var()
    n0 = data[data['T'] == 0]["Y"].count()
    
    "Calculate Standard Error"
    
    se = sqrt((treated_var / n1) + (non_treated_var / n0))
    
    "Calculate z"
    
    z = np.abs(stats.norm.ppf(alpha / 2))
    
    "Calculate Confidence Interval"
    
    CI_lower = ATE_estimate - z * se
    CI_upper = ATE_estimate + z * se
    
    return(ATE_estimate, CI_lower, CI_upper)

def calculate_ate_pvalue(data):
    
    "Calculate ATE_estimate E[Y|T = t] − E[Y|T = 0]"
    treated_mean = np.mean(data[data['T'] == 1]["Y"])
    non_treated_mean = np.mean(data[data['T'] == 0]["Y"])
    
    ATE_estimate = treated_mean - non_treated_mean
    
    "Calculate t_stat"
    diff_mu = treated_mean - non_treated_mean
    diff_se = np.sqrt(data[data['T'] == 1]["Y"].sem()**2 + data[data['T'] == 0]["Y"].sem()**2)
    t_stat = (diff_mu - 0) / diff_se
    
    "Calculate p_value"
    p_value = (1 - stats.norm.cdf(t_stat))*2
    
    return(ATE_estimate, t_stat, p_value)

