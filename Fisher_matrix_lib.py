
#
# This is based on the stuff from Carlos and Livia
#
#
import numpy as np
import matplotlib.pyplot as plt

def partial(ff, k, p, H=1e-6):
    '''
    Returns the partial derivative of a function
    
    ff = function to be differentiated
    k  = index of the variable with respect to which the derivative is taken
    p  = point where to calculate the derivative
    H  = approximation step size
    '''
    
    inf,sup = np.copy(p),np.copy(p) # points where to calculate ff
    inf[k] -= H/2 # (..., x_i - H/2, ...)
    sup[k] += H/2 # (..., x_i + H/2, ...)
    
    return (ff(sup) - ff(inf)) / H


def Fisher_matrix(p, ffs, sigmas, priors=None):
    '''
    Returns the Fisher matrix of a model
    
    p      = list of model parameters
    ffs    = array of the functions that describe the observables
    sigmas = array of the Gaussian errors of the observables
    priors = array of the priors to be used (same length and order as p)
    '''

    m = len(p)
    F = np.zeros([m,m]) # matrix size
    
    for i in range(m):
        for j in range(m):

            part_i, part_j = partial(ffs,i,p), partial(ffs,j,p)
            F[i,j] = np.sum((part_i * part_j) / sigmas**2) # calculates each entry
    
    if priors is None or np.all(priors==0):
        return F
    
    else:
        D = np.zeros_like(priors, dtype = float)
        mask = priors != 0
        D[mask] = priors[mask]**-2
        F_priors = np.diag(D) # creates a diagonal matrix with the priors

        return F + F_priors # adds priors to initial matrix


def cov_matrix(F):
    '''
    Returns the covariance matrix of a model

    F = Fisher matrix of the model
    '''
    
    # HRK edit ! return np.mat(F).I # inverse of the Fisher matrix
    #
    return np.asmatrix(F).I # inverse of the Fisher matrix


def unc(F):
    '''
    Returns the uncertainties of the model parameters
    
    F = Fisher matrix of the model
    '''

    C = cov_matrix(F)
    return np.sqrt(abs(np.diag(C)))


def rho(F, i, j):
    '''
    Returns the correlation coefficient between parameters i and j

    F    = Fisher matrix of the model
    i, j = indexes of the parameters for which to calculate the correlation coefficient
    '''
    
    C = cov_matrix(F)
    rho = C[i,j]/np.sqrt(C[i,i]*C[j,j])

    return rho


def ellipse(F, i, j):

    '''
    Returns the axes and the angle from the x-axis of the confidence ellipse of parameters i and j

    F    = Fisher matrix of the model
    i, j = indexes of the parameters for which to calculate the correlation coefficient´
    '''

    C = cov_matrix(F)
    C_ij = C[np.ix_([i, j], [i, j])]
    
    A2 = (C_ij[0,0]+C_ij[1,1]) / 2. + np.sqrt((C_ij[0,0]-C_ij[1,1])**2 / 4. + C_ij[0,1]**2)
    a2 = (C_ij[0,0]+C_ij[1,1]) / 2. - np.sqrt((C_ij[0,0]-C_ij[1,1])**2 / 4. + C_ij[0,1]**2)
    if C_ij[0,0] >= C_ij[1,1]:
        ai2, aj2 = A2, a2
    else:
        ai2, aj2 = a2, A2

    angle = 0.5*np.arctan((2*C_ij[0,1])/(C_ij[0,0]-C_ij[1,1])) # rad
    
    return np.sqrt(abs(ai2)), np.sqrt(abs(aj2)), angle


def FoM(F, i, j, delta_chi2=2.3):
    '''
    Returns the Figure of Merit of p_i and p_j

    F          = Fisher matrix of the model
    i, j       = indexes of the parameters for which to calculate the correlation coefficient
    delta_chi2 = confidence interval of interest
    '''

    ai, aj = ellipse(F, i, j)[:2]
    
    return 1 / (ai * aj * delta_chi2)


def draw_ellipse(F, i, j, delta_chi2=2.3, center=(0, 0), ax=None, **kwargs):
    '''
    Returns the confidence ellipse of p_i (x) and p_j (y)
    
    F = Fisher matrix of the model
    i, j = indexes of the parameters for the ellipse
    delta_chi2 = confidence interval of interest
    **kwargs = specifications of the ellipse
    '''

    ai, aj, angle = ellipse(F, i, j)
    
    theta = np.linspace(0, 2*np.pi, 200) # ellipse
    x = ai * np.cos(theta) * delta_chi2
    y = aj * np.sin(theta) * delta_chi2

    x_rot = center[0] + x * np.cos(angle) - y * np.sin(angle) # rotation
    y_rot = center[1] + x * np.sin(angle) + y * np.cos(angle)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    ax.plot(x_rot, y_rot, **kwargs)
    return fig, ax


