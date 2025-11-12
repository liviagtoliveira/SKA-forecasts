
#
# This is based on the stuff from Carlos and Livia
#
#
import numpy as np
import matplotlib.pyplot as plt

import results_plotting_lib as dvplt
import redshiftdrift_lib as dvlib
import Fisher_matrix_lib as fmlib

p_LCDM  = np.array([70., -.55, 1.]) # H0, q0, j0
priors_baseline = np.array([10., 0., 0.]) # prior on H0

#
#  Here start the playing around
#

fwhm_def   = 150E5                       # cm/s (see Obreschkow 2009)
z_val      = [0.1, 0.2, 0.3, 0.4, 0.5]   # redshifts
z_eg       = 0.25                        # example redshift
t_obs_def  = 3600*1                      # s
Dnu_val    = [1E-3,1E-2,1E-1]             # [0.1.285e3, 1e4]              # Hz
S_area_val = [5000,10000,30000]          # sq deg

t_exp_def  = 12                          # yr
N_ant_def  = 144                         # integer


# === Plot number counts ===
#
# this should be flat and should not change since we use the channel width
#
#dvplt.plot_Nz(z_val, Dnu_val, S_area_val, t_obs_def, N_ant_def)


# === Plot colour image of error estimates (image dimension sky area (horizontal) and channel width (vertical)) ===
#
#dvplt.im_sigmav(z_eg, Dnu_val, S_area_val, t_obs_def, N_ant_def, fwhm_def,doprtinfo=False,doplot=True)

# === Plot colour image of significance of error estimates versus theoretical model (image dimension sky area (horizontal) and channel width (vertical)) ===
#
#dvplt.im_vsignificance(z_eg, Dnu_val[0], Dnu_val[-1], S_area_val[0], S_area_val[-1], t_obs_def, t_exp_def, N_ant_def, fwhm_def, p_LCDM)


# === Plot diagram error estimates versus redshift per sky area  ===
#
#dvplt.plot_sigmav(z_val, Dnu_val, S_area_val, t_obs_def, N_ant_def, fwhm_def)

# === Plot diagram significance of error estimates/theoretical model versus redshift per sky area  ===
#
#dvplt.plot_vsignificance(z_val, Dnu_val, S_area_val, t_obs_def, t_exp_def, N_ant_def, fwhm_def, p_LCDM)

# === Plot diagram theoretical model redshiftdrifts and uncertainties versus redshift per sky area  ===
#
#dvplt.plot_Dv(z_val, Dnu_val, S_area_val, t_obs_def, t_exp_def, N_ant_def, fwhm_def, p_LCDM)


# === Plot confidence ellipses  ===
#
dvplt.plot_ellipses(np.array([0.1, 0.3, 0.5]), Dnu_val, S_area_val, t_obs_def, t_exp_def, N_ant_def, fwhm_def, p_LCDM, priors_baseline, savefig=True)


def analysis_FoM(z, t_obs, t_exp, N_ant, Dnu, S_area, fwhm, priors=None):
    
    sigma_v_vect = np.vectorize(lambda z_i: dvlib.sigma_v_func(z_i, t_obs, N_ant, Dnu, S_area, fwhm))
    sigma_v = sigma_v_vect(z)
    delta_v = lambda p: dvlib.delta_v_func(z, p, t_exp)

    F = fmlib.Fisher_matrix(p_LCDM, delta_v, sigma_v, priors)
    
    return F, fmlib.FoM(F, 1, 2), fmlib.unc(F), delta_v(p_LCDM), sigma_v # Fisher Matrix, Figure of Merit between q0 and j0 and uncertainties


def hrk_analysis_results(z, t_obs, t_exp, N_ant, Dnu, S_area, fwhm, priors=None): # t_obs in s, t_exp in yrs, Dnu in Hz, S_area in sq deg and fwhm in cm/s

    FMANA = analysis_FoM(z, t_obs, t_exp, N_ant, Dnu, S_area, fwhm, priors)
    FoM   = FMANA[1]
    unc   = FMANA[2]

    delta_v = dvlib.delta_v_func(z, p_LCDM, t_exp)
    sigma_v = dvlib.sigma_v_func(z, t_obs, N_ant, Dnu, S_area, fwhm)
    
    return f'\n=== Analysis ===\nRedshift: {z}\nDnu: {Dnu}\nS_area: {S_area}\nExpected drift: {delta_v}\nMeasured Error: {sigma_v}\nFigure of Merit: {FoM} \nUncertainties of H0, q0, j0: {unc}'


def analysis(t_obs, t_exp, N_ant, Dnu, S_area, fwhm, priors=None, doplot=False): # t_obs in s, t_exp in yrs, Dnu in Hz, S_area in sq deg and fwhm in cm/s
    z1 = np.array([.3])
    z2 = np.array([.3, .5])
    z3 = np.array([.1, .3, .5])

    z_bins  = [z1, z2, z3]
    F       = [analysis_FoM(z_i, t_obs, t_exp, N_ant, Dnu, S_area, fwhm, priors)[0] for z_i in z_bins]
    FoM     = [analysis_FoM(z_i, t_obs, t_exp, N_ant, Dnu, S_area, fwhm, priors)[1] for z_i in z_bins]
    unc     = [analysis_FoM(z_i, t_obs, t_exp, N_ant, Dnu, S_area, fwhm, priors)[2] for z_i in z_bins]
    delta_v = [analysis_FoM(z_i, t_obs, t_exp, N_ant, Dnu, S_area, fwhm, priors)[3] for z_i in z_bins]
    sigma_v = [analysis_FoM(z_i, t_obs, t_exp, N_ant, Dnu, S_area, fwhm, priors)[4] for z_i in z_bins]
    i       = np.argmax(FoM)

    if doplot:
        fig, ax = plt.subplots()
        fmlib.draw_ellipse(F[i], 1, 2, delta_chi2=2.3, center=(p_LCDM[1], p_LCDM[2]), ax=ax, label=r'1$\sigma$')
        fmlib.draw_ellipse(F[i], 1, 2, delta_chi2=6.17, center=(p_LCDM[1], p_LCDM[2]), ax=ax, label=r'2$\sigma$')
        fmlib.draw_ellipse(F[i], 1, 2, delta_chi2=11.8, center=(p_LCDM[1], p_LCDM[2]), ax=ax, label=r'3$\sigma$')
        ax.legend()
        ax.set_xlabel(r'$q_0$')
        ax.set_ylabel(r'$j_0$')
        plt.show()

    return f'\n=== Analysis ===\nPriors: {priors}\nDnu: {Dnu}\nS_area: {S_area}\nArray of redshifts: {z_bins[i]}\nExpected drift: {delta_v[i]}\nMeasured Error: {sigma_v[i]}\nFigure of Merit (1 sigma): {FoM[1]} \nUncertainties of H0, q0, j0: {unc[1]}'    
    

#
# === First attemp to do the analyis ===
#

best_values = dvplt.im_sigmav(z_eg, Dnu_val, S_area_val, t_obs_def, N_ant_def, fwhm_def,doprtinfo=True,doplot=False)

best_dnu    = best_values[0][0]
best_area   = best_values[0][1]


print(analysis(t_obs_def, t_exp_def, N_ant_def, best_dnu, best_area, fwhm_def))
print(analysis(t_obs_def, t_exp_def, N_ant_def, best_dnu, best_area, fwhm_def, priors_baseline))