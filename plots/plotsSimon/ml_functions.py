import numpy as np

# Function to calculate sum of squared weights for each bin
def calc_squared_weights(data, weights, bins):
    squared_weights = np.square(weights)
    hist_squared, _ = np.histogram(data, bins=bins, weights=squared_weights)
    return hist_squared

def calculate_chisq(hist1, hist2, hist_squared1):
    # Create a mask to exclude rows where hist_squared1 is 0
    valid_mask = hist_squared1 != 0
    # Apply the mask to the arrays
    chi_squared = np.sum(
        np.square(hist1[valid_mask] - hist2[valid_mask]) / hist_squared1[valid_mask]
    )
    return chi_squared

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_k(Gamma,m):
    gam = np.sqrt(m**2 *(m**2 + Gamma**2))
    k = 2 * np.sqrt(2) * Gamma * gam / ( np.pi * np.sqrt(m**2 + gam) )
    return k

def bw_reweight(s,m_old,m_new,Gamma,k_old,k_new):
    k = k_new / k_old
    
    a =  ((s**2 - m_old**2)**2 + (m_old**2 * Gamma**2)) / ((s**2 - m_new**2)**2+ (m_new**2 * Gamma**2)) #SH: Beware of denomminator
    return a*k
    
    #(calc_squared_weights,calculate_chisq,count_parameters,get_k,bw_reweight)