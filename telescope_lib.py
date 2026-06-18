
import math
import numpy as np


def SEFD(D_dish):
    '''
    Returns the SEFD of a dish [W m⁻² Hz⁻¹]
    
    D_dish = diameter of the dish [m]
    '''

    kB       = 1.380649e-23 # J/K
    T_sys    = 23
    eta_dish = 1.
    return 2*kB*T_sys / (eta_dish*np.pi*D_dish**2/4)


def S_rms_func(t_obs, N_ant, Dnu):
    '''
    Returns the RMS noise of the array [𝜇Jy]

    t_obs = observation time (per pointing) [s]
    N_ant = total number of antennas
    Dnu   = channel width [Hz]
    '''

    eta_sys = 1.
    n_13_5 = 64
    n_15 = N_ant - 64
    SEFD_arr = (n_13_5*(n_13_5-1)/SEFD(13.5)**2 + 2*n_13_5*n_15/(SEFD(13.5)*SEFD(15.)) + n_15*(n_15-1)/SEFD(15.)**2)**-.5
    return SEFD_arr / (eta_sys*np.sqrt(2*Dnu*t_obs)) * 1e32 # W m⁻² Hz⁻¹ to 𝜇Jy


def t_obs_tot(t_obs, S_area):
    '''Returns the total observation time spent in all pointings, based on an hexagonal mosaicking [days]
        
    t_obs  = observation time (per pointing) [s]
    S_area = observed survey area [sq deg]'''

    theta_hex = 0.35
    theta_row = 0.31
    side = np.sqrt(S_area)
    N_hex_s = math.ceil(side/theta_hex)
    N_hex_l = N_hex_s + 1
    N_rows = math.ceil(side/theta_row)
    N_rows_s = N_rows // 2
    N_rows_l = N_rows - N_rows_s
    N_p = N_rows_s*N_hex_s + N_rows_l*N_hex_l
    return t_obs * N_p / (3600*24)


##########################################################################################################################################

def S_rms_func_hrk(T_obs,N_MKplus,Dnu):
    '''
    Returns the RMS noise of the array [𝜇Jy]

    t_obs = observation time (per pointing) [s]
    N_ant = total number of antennas
    Dnu = channel width [Hz]
    '''
    
    N_MK             = 64
    BW               = Dnu #       875E6 [Hz]
    N_pol            = 2
    SEFD_MK          = 426   #[Jy]
    SEFD_SKA         = 0.7332 * SEFD_MK
    array_eff_mkplus = 1

    N_tot           = N_MK + N_MKplus
    N_bsl_tot       = N_tot * (N_tot -1) / 2 # total number of baselines
    N_bsl_MK        = N_MK * (N_MK -1) / 2 # pure MK Antenna baselines
    N_bsl_MKplus    = N_MKplus * (N_MKplus -1) / 2 # pure SKAMID antenna baselines
    N_bsl_MK_MKplus = N_bsl_tot - N_bsl_MK - N_bsl_MKplus # intermixed baselines

    image_sensitivity_MKplus = 1/array_eff_mkplus * np.sqrt( 1 / (N_pol * N_tot * (N_tot - 1) * BW * T_obs) * (SEFD_MK**2 * N_bsl_MK + SEFD_SKA**2 * N_bsl_MKplus + SEFD_SKA * SEFD_MK * N_bsl_MK_MKplus) / N_bsl_tot )

    return image_sensitivity_MKplus