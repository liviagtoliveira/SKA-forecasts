
from telescope_lib import *
from HI_galaxy_lib import *

c_light = 29979245800    # cm/s
HI_freq = 1420405751.768 # Hz


def delta_v_exct_func(z, model, t_exp):
    '''
    Returns the exact spectroscopic velocity shift [cm/s]

    z     = observed redshift
    model = array of H0 [km/s/Mpc], OmegaM, w0 and wa values
    t_exp = total experiment time (time span between observations) [yrs]
    '''

    H0, OmegaM, w0, wa = model
    H0_s           = H0 / 3.08567758128e19 # km/s/Mpc to 1/s
    E_z_exct       = np.sqrt(OmegaM*(1+z)**3 + (1-OmegaM)*(1+z)**(3*(1+w0+wa))*np.exp(-3*wa*(z/(1+z))))
    Z1_exct        = 1 + z - E_z_exct

    return c_light * H0_s * t_exp*365*24*3600 * Z1_exct / (1 + z)


def delta_v_0_func(z, p, t_exp):
    '''
    Returns the spectroscopic velocity shift [cm/s]

    z     = observed redshift
    p     = array of H0 [km/s/Mpc], q0 and j0 values
    t_exp = total experiment time (time span between observations) [yrs]
    '''

    H0, q0, j0 = p
    E = 1 + (q0 + 1) * z + 1/2 * (j0 - q0**2) * z**2
    Z1 = 1 + z - E
    H0_s       = H0 / 3.08567758128e19 # km/s/Mpc to 1/s
    return c_light * H0_s * t_exp*365*24*3600 * Z1 / (1 + z)


def E_q_z(z, OmegaM, w0, wa):
    '''
    Returns the rescaled Hubble parameter squared and deceleration parameter for a given model

    z      = observed redshift
    OmegaM = matter density parameter
    w0     = equation of state parameter
    wa     = dark energy dynamical parameter
    '''
    E2 = OmegaM*(1+z)**3 + (1-OmegaM)*(1+z)**(3*(1+w0+wa))*np.exp(-3*wa*z/(1+z))
    q = 1/2 + 3/(2*E2)*(w0+wa*z/(1+z))*(1-OmegaM)*(1+z)**(3*(1+w0+wa))*np.exp(-3*wa*z/(1+z))
    return E2, q


def param(z0, model):
    '''
    Returns the rescaled Hubble parameter squared, deceleration parameter and jerk parameter for a given model

    z0    = center redshift
    model = array of H0 [km/s/Mpc], OmegaM, w0 and wa values 
    '''
    H0, OmegaM, w0, wa = model
    E2_z0, q_z0 = E_q_z(z0, OmegaM, w0, wa)
    H_z0 = np.sqrt(E2_z0)*H0

    h = 1e-5 * abs(q_z0) if q_z0 != 0 else 1e-5
    q_plus = E_q_z(z0+h/2, OmegaM, w0, wa)[1]
    q_minus = E_q_z(z0-h/2, OmegaM, w0, wa)[1]
    dq_dz = (q_plus-q_minus)/h

    j_z0 = q_z0*(1+2*q_z0) + (1+z0)*dq_dz

    #a = 3*OmegaM*(1+z0)**2 + 3*(1+w0+wa*z0/(1+z0))*(1-OmegaM)*(1+z0)**(3*(1+w0+wa)-1)*np.exp(3*wa*z0/(1+z0))
    #b = 6*OmegaM*(1+z0) + 3*wa/(1+z0)**3*(1-OmegaM)*(1+z0)**(3*(1+w0+wa))*np.exp(-3*wa*z0/(1+z0)) + 3*(1+w0+wa*z0/(1+z0))*(1-OmegaM)*(1+3*w0+3*wa)*(1+z0)**(1+3*w0+3*wa)*np.exp(-3*wa*z0/(1+z0)) - 9*wa/(1+z0)**2*(1+w0+wa*z0/(1+z0))*(1-OmegaM)*(1+z0)**(2+3*w0+3*wa)*np.exp(-3*wa*z0/(1+z0))
    #j_z0 = (1+z0)**2 * (-a/(2*E2_z0**1.5) + b/(2*E2_z0) + a**2/(4*E2_z0**2)) - (1+z0)/E2_z0 * a + 1

    return np.array([H_z0, q_z0, j_z0, H0])


def delta_v_cosmog_func(z, z0, p, t_exp):
    '''
    Returns the spectroscopic velocity shift in the cosmographic expansion centered at a certain redshift [cm/s]

    z     = observed redshift
    z0    = redshift to center expansion
    p     = array of H(z0) [km/s/Mpc], q(z0), j(z0) and H0
    t_exp = total experiment time (time span between observations) [yrs]
    '''
    H_z0, q_z0, j_z0, H0 = p
    dz = z - z0
    E_z = H_z0/H0 * (1 + (1+q_z0)/(1+z0) * dz + (j_z0-q_z0**2)/(2*(1+z0)**2) * dz**2)
    H0_s = H0 / 3.08567758128e19 # km/s/Mpc to 1/s
    t_exp_s = t_exp*365*24*3600
    Z1 = 1 + z - E_z

    return c_light * H0_s * t_exp_s * Z1 / (1 + z)


def relvelo(obsfreq):
    '''
    Returns the relativistic velocity [cm/s]
    (equation 10-77 and 10-78a Kraus, Radio Astronomy)

    obsfreq = observed frequency [Hz]
    '''

    z     = HI_freq/obsfreq -1.
    m     = (z + 1)**2
    rvelo = c_light * (m - 1)/(m + 1)
    return rvelo

    
def Dnu2dv(Dnu, z):
    '''
    Returns the channel width in velocity [cm/s]

    Dnu = channel width in frequency [Hz]
    z   = observed redshift
    '''
    
    lower_freq = HI_freq/(z+1) - Dnu/2
    upper_freq = HI_freq/(z+1) + Dnu/2
    upper_v    = relvelo(lower_freq)
    lower_v    = relvelo(upper_freq)
    dv = (upper_v - lower_v)
    return dv


def sigma_v_func(z, t_obs, N_ant, Dnu, S_area, fwhm, delta_z):
    '''
    Returns the uncertainty sigma_v in the cosmological velocity drift [cm/s]
    
    z       = observed redshift
    t_obs   = observation time (per pointing) [s]
    N_ant   = total number of antennas
    Dnu     = channel width [Hz]
    S_area  = observed survey area [sq deg]
    fwhm    = HI line width [cm/s]
    delta_z = delta z for dN/dz integration
    '''

    # determine dispersion of the HI line
    # sigma_HI or p4.
    #
    # Note this factor is based on the definition
    # of the Gaussian function in the Minin and Kamalabadi
    # and can not be used in general !
    #
    p4            = fwhm / (2*np.sqrt(np.log(2))) 

    # determine the channel width in velocity
    #
    dv            = Dnu2dv(Dnu,z)

    # Here we determine the scaling factor for
    # detecting a galaxy at a single channel, so 
    # essentially determine the galaxies we can work
    # with. Note that is not related to the detection
    # which we assume to stack.
    #
    # HI profile at 20 percent level see Obreschow et al. 2009
    #
    w20_HI_galaxy = 3.6 * p4 * 1/np.sqrt(2)
    #
    # Essentially this factor scales the Dnu to a single
    # channel.
    #
    Single_channel_detection  = w20_HI_galaxy/dv
    #
    # If Dnu is so large that a HI profile is just confined in a
    # single channel.
    # 
    if Single_channel_detection > 1:
        Single_channel_detection_factor  = 1/np.sqrt(Single_channel_detection)
    else:
        Single_channel_detection_factor  = 1

    # Determine the number of HI galaxies using the 
    # single channel detection
    #
    rms                                 = S_rms_func(t_obs, N_ant, Dnu) * Single_channel_detection_factor
    #print(rms)
    N_galaxies, Minimum_peak_Amplitude  = N_func(z, rms, S_area, delta_z)

    # Error on the line center parameter of a pure Gaussian will 
    # be used to estimate the accuray of determine the
    #
    # redshift_drift
    #
    # Based on the equation 14 of Minin and Kamalabadi (2009)
    # and assuming that the sigma in the equation reduces by
    # 1/sqrt(number_of_staked_spectra)
    #
    # Caution the Minimum_peak_Amplitude is a powerfull factor 
    # here and we use the boundaries of defined in N_func and
    # assuming that all, even the weakest line will
    # contribute to reduce the rms. For the future this needs to
    # be evaluated properly.
    #
    sigma_v = rms/np.sqrt(N_galaxies) * (Minimum_peak_Amplitude**2 * np.sqrt(np.pi) / (p4*dv*np.sqrt(2)))**-.5 

    return sigma_v