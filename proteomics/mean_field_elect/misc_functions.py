import numpy as np
import random

def get_nu_alpha(is_vals, k_targets):
    is_vals     = np.array(is_vals)
    kprime_vals = np.array(k_targets)
    log_is      = np.log(is_vals)
    log_kprime  = np.log(kprime_vals)

    nu = -1.0*(log_kprime[0] - log_kprime[1])/(log_is[0] - log_is[1])
    phi_alpha = kprime_vals[0]/((is_vals[0]*1e3)**(-1.0*nu))
    return nu, phi_alpha

def get_subset(records, seed=1, n=10000):
    results = []
    random.seed(seed)
    for i in random.sample(range(0, len(records)), n):
        results.append( records[i] )
    return results
