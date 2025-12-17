
import numpy as np

def Simpson(ff, a, b, N=100):
    '''
    Returns the integral of a function between a and b

    ff   = function to be integrated
    a, b = limits of integration
    N    = number of steps
    '''
    
    h = (b-a)/N
    x = np.linspace(a,b,N+1)
    y = ff(x)
    I = h/3 * (y[0] + y[-1] + 4*np.sum(y[1:N:2]) + 2*np.sum(y[2:N-1:2]))
    return I

def N_func(z, S_rms, S_area, delta_z=0.05):
    '''
    Returns the number of detected HI sources and the lower boundary of the emission line amplitude [𝜇Jy]

    z       = observed redshift
    S_rms   = array's noise [𝜇Jy]
    S_area  = observed survey area [sq deg]
    delta_z = delta z for dN/dz integration
    '''
    #
    # Yahya et al. MNRAS, 450, 3 2015
    # https://doi.org/10.1093/mnras/stv695
    #
    
    if S_rms < 1.:
        c1 = 6.21
        c2 = 1.72
        c3 = 0.79
        A  = .1

    elif 1. <= S_rms < 3.:
        c1 = 6.55
        c2 = 2.02
        c3 = 3.81
        A  = 1.

    elif 3. <= S_rms < 5.:
        c1 = 6.53
        c2 = 1.93
        c3 = 6.22
        A  = 3.

    elif 5. <= S_rms < 6.:
        c1 = 6.55
        c2 = 1.93
        c3 = 6.22
        A  = 5.

    elif 6. <= S_rms < 7.3:
        c1 = 6.58
        c2 = 1.95
        c3 = 6.69
        A  = 6.
    
    elif 7.3 <= S_rms < 10:
        c1 = 6.55
        c2 = 1.92
        c3 = 7.08
        A  = 7.3

    elif 10 <= S_rms < 23:
        c1 = 6.44
        c2 = 1.83
        c3 = 7.59
        A  = 10

    elif 23 <= S_rms < 40:
        c1 = 6.02
        c2 = 1.43
        c3 = 9.03
        A  = 23

    elif 40 <= S_rms < 70:
        c1 = 5.74
        c2 = 1.22
        c3 = 10.58
        A  = 40

    elif 70 <= S_rms < 150:
        c1 = 5.63
        c2 = 1.41
        c3 = 15.49
        A  = 70

    elif 150 <= S_rms < 200:
        c1 = 5.48
        c2 = 1.33
        c3 = 16.62
        A  = 150

    elif 200 <= S_rms < 900:
        c1 = 5.00
        c2 = 1.04
        c3 = 17.52
        A  = 200

    else: # in case there is too much noise, we consider only 5 sources will be detected
        return 10e-3 * S_area, 900

    dN_dz = lambda z: 10**c1 * z**c2 * np.exp(-c3*z)
    a, b = z-delta_z, z+delta_z
    N = Simpson(dN_dz,a,b) * S_area # integration over a bin of 0.2 centered in z

    #print('Simpson ',Simpson(dN_dz,a,b),'N ',np.array(N).astype(int), A,a,b)

    
    return np.array(N).astype(int), A


def N_func_SAX(z, S_rms, S_area, delta_z=0.05):
    '''
    Returns the number of detected HI sources and the lower boundary of the emission line amplitude [𝜇Jy]

    z       = observed redshift
    S_rms   = array's noise [𝜇Jy]
    S_area  = observed survey area [sq deg]
    delta_z = delta z for dN/dz integration
    '''
    #
    # Obreschkow  et al. ApJ, 703, 1890 2009
    # DOI 10.1088/0004-637X/703/2/1890
    #
    # values from Table 1 Peak flux densities for HI
    
    if S_rms < 0.01:
        c1 = 6.55
        c2 = 2.54
        c3 = 1.42
        A  = .01

    elif 0.01 <= S_rms < 0.1:
        c1 = 6.87
        c2 = 2.85
        c3 = 2.17
        A  = 0.01

    elif 0.1 <= S_rms < 1:
        c1 = 6.73
        c2 = 2.32
        c3 = 3.09
        A  = 0.1

    elif 1 <= S_rms < 10:
        c1 = 5.75
        c2 = 1.14
        c3 = 3.95
        A  = 1

    elif 10 <= S_rms < 100:
        c1 = 4.56
        c2 = 0.43
        c3 = 6.86
        A  = 10

    elif 100 <= S_rms < 1000:
        c1 = 6.62
        c2 = 2.64
        c3 = 35.49
        A  = 100

    else: # in case there is too much noise, we consider only 5 sources will be detected
         return 10e-3 * S_area, 1000
        
    dN_dz = lambda z: 10**c1 * z**c2 * np.exp(-c3*z)
    a, b = z-delta_z, z+delta_z
    N = Simpson(dN_dz,a,b) * S_area # integration over a bin of 0.2 centered in z

    #print('Simpson ',Simpson(dN_dz,a,b),'N ',np.array(N).astype(int), A,a,b)
    
    return np.array(N).astype(int), A
