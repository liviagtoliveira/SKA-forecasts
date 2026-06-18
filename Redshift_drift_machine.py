
#
# This is based on the stuff from Carlos and Livia
#
#
import numpy as np
import matplotlib.pyplot as plt

import telescope_lib as tellib
import results_plotting_lib as dvplt
import redshiftdrift_lib as dvlib
import Fisher_matrix_lib as fmlib

# Center redshift
z0 = 0.4

# Models
LCDM  = np.array([70., .3, -1, 0]) # H0, OmegaM, w0, wa
p_LCDM = dvlib.param(z0, LCDM) # H(z0), q(z0), j(z0), H0
p_LCDM_0 = np.array([70., -.55, 1.]) # H0, q0, j0
DCD   = np.array([66.74, .3191, -.752, -.86])
p_DCD = dvlib.param(z0, DCD)
p_DCD_0 = np.array([66.74, -.27, -.45])

# Prior on H0
priors_baseline = np.array([0., 0., 0., 10.])
priors_baseline_0 = np.array([10., 0., 0.])

#
#  Here start the playing around
#

fwhm_def    = 150E5                     # cm/s (see Obreschkow 2009)
z_val       = [0.1, 0.2, 0.3, 0.4, 0.5] # redshifts
z_eg        = 0.3                       # example redshift
delta_z_def = 0.1                       # integration delta z
t_obs_def   = 169                       # s
Dnu_val     = [1e-5, 1e-3]              # Hz
S_area_val  = [3500, 5000]              # sq deg
t_exp_def   = 32                        # yr
N_ant_def   = 197                       # integer


# === Plot rms noise ===
#
#dvplt.plot_rms_panel(z_val, Dnu_val, t_obs_def, [144, 170, 197], fwhm_def)


# === Plot colour image of rms noise ===
#
#dvplt.im_rms(z_eg, 1e6, 100, 500, 144, 197, fwhm_def, savefig=True)

# === Plot colour image of number of sources per sq deg ===
#
#dvplt.im_Nz(z_eg, 1e-3, 100, 500, 144, 197, fwhm_def, delta_z_def)


# === Plot colour image of how the uncertainty of the spectroscopic velocity goes ===
#
#dvplt.im_sigmav_tN(z_eg, Dnu_val[-1], 100, 500, 144, 197, fwhm_def, delta_z_def)


# === Plot number counts ===
#
#dvplt.plot_Nz_panel(z_val, Dnu_val, S_area_val, t_obs_def, N_ant_def, fwhm_def, delta_z_def) # this should be flat and should not change since we use the channel width

#dvplt.plot_Nz(z_val, 1, delta_z_def, color='tab:blue', label=r'$S_{rms}$ = 1 $\mu$Jy')
#dvplt.plot_Nz(z_val, 5, delta_z_def, color='tab:orange', label=r'$S_{rms}$ = 5 $\mu$Jy')
#dvplt.plot_Nz(z_val, 10, delta_z_def, color='tab:green', label=r'$S_{rms}$ = 10 $\mu$Jy')
#dvplt.plot_Nz(z_val, 40, delta_z_def, color='tab:red', label=r'$S_{rms}$ = 40 $\mu$Jy')
#dvplt.plot_Nz(z_val, 100, delta_z_def, color='tab:purple', label=r'$S_{rms}$ = 100 $\mu$Jy')
#plt.legend(loc="upper left", bbox_to_anchor=(0.65, 0.57))
#plt.xlabel('z')
#plt.ylabel(r'$N_{HI}$ / sq deg')
#plt.title('Number density of detected sources and RMS noise')
#plt.savefig('N_z.png', dpi=300, bbox_inches="tight")
#plt.show()


# === Plot diagram error estimates versus redshift per sky area  ===
#
#dvplt.plot_sigmav(z_val, Dnu_val, S_area_val, t_obs_def, N_ant_def, fwhm_def, delta_z_def)


# === Plot colour image of error estimates (image dimension sky area (horizontal) and channel width (vertical)) ===
#
#dvplt.im_sigmav(z_eg, Dnu_val, S_area_val, t_obs_def, N_ant_def, fwhm_def, delta_z_def, doprtinfo=False, doplot=True)


# === Plot diagram theoretical model redshiftdrifts and uncertainties versus redshift per sky area  ===
#
#z_plt = np.linspace(.1,1,10)
#t_exp_plt = 12
#fig, ax = plt.subplots()
#dvplt.plot_Dv_exct(z_plt, t_exp_plt, LCDM, 'tab:blue', ax=ax, label=r'Exact $\Lambda$CDM')
#dvplt.plot_Dv_0(z_plt, t_exp_plt, p_LCDM_0, 'tab:blue', ax=ax, label=r'Cosmographic $\Lambda$CDM')
#dvplt.plot_Dv_exct(z_plt, t_exp_plt, DCD, 'tab:orange', ax=ax, label='Exact CPL (DESI+CMB+DESY5)')
#dvplt.plot_Dv_0(z_plt, t_exp_plt, p_DCD_0, 'tab:orange', ax=ax, label='Cosmographic CPL (DESI+CMB+DESY5)')
#ax.legend()
#ax.set_xlabel('z')
#ax.set_ylabel(r'$\Delta$v [cm/s]')
#ax.set_title(f'$t_{{exp}}$ = {t_exp_plt} yrs')
#fig.savefig('Dv_plot.png', dpi=300)
#plt.show()

#dvplt.plot_Dv_errbar(z_val, Dnu_val, S_area_val, t_obs_def, t_exp_def, N_ant_def, fwhm_def, p_LCDM_0, delta_z_def)


# === Plot diagram significance of error estimates/theoretical model versus redshift per sky area  ===
#
#dvplt.plot_vsignificance(z_val, Dnu_val, S_area_val, t_obs_def, t_exp_def, N_ant_def, fwhm_def, p_LCDM_0, delta_z_def)


# === Plot colour image of significance of error estimates versus theoretical model (image dimension sky area (horizontal) and channel width (vertical)) ===
#
#dvplt.im_vsignificance(z_eg, Dnu_val[0], Dnu_val[-1], S_area_val[0], S_area_val[-1], t_obs_def, t_exp_def, N_ant_def, fwhm_def, p_LCDM_0, delta_z_def)


# === Plot confidence ellipses  ===
#
#dvplt.plot_ellps_panel(np.array([0.1, 0.3, 0.5]), Dnu_val, S_area_val, t_obs_def, t_exp_def, N_ant_def, fwhm_def, p_LCDM_0, delta_z_def, priors=priors_baseline)
#dvplt.plot_ellipses(np.array([0.1, 0.3, 0.5]), Dnu_val, 5000, t_obs_def, t_exp_def, N_ant_def, fwhm_def, p_LCDM_0, delta_z_def, priors=priors_baseline)


# === Plot the final image of the SKA Book  ===
#
#dvplt.plot_dv_ellipses_SKA_book([0.1, 0.2, 0.3, 0.4, 0.5], [1E-3,1E-2,1E-1], [5000,10000,30000], 3600*1, 12, 144, 150E5, p_LCDM_0, 0.1, priors=np.array([10.,0.,0.]), savefig=False)



def analysis_FoM(z, t_obs, t_exp, N_ant, Dnu, S_area, fwhm, delta_z, p, priors=None):
    '''Returns the Fisher matrix, FoM between q0 and j0, parameter constraints, expected drifts and their uncertainties
    
    z       = array of redshift bins
    t_obs   = observation time [s]
    t_exp   = total experiment time [yrs]
    N_ant   = number of antennas
    Dnu     = channel width [Hz]
    S_area  = survey area [sq deg]
    fwhm    = HI line width [cm/s]
    delta_z = interval of redshift to integrate dN/dz
    p       = array of model parameters
    priors  = array of priors for H0, q0, j0'''
    
    sigma_v_vect = np.vectorize(lambda z_i: dvlib.sigma_v_func(z_i, t_obs, N_ant, Dnu, S_area, fwhm, delta_z))
    sigma_v = sigma_v_vect(z)
    delta_v = lambda params: dvlib.delta_v_0_func(z, params, t_exp) # for cosmography centered at z=0
    #delta_v = lambda params: dvlib.delta_v_cosmog_func(z, z0, params, t_exp) # for generalized cosmography

    F = fmlib.Fisher_matrix(p, delta_v, sigma_v, priors)
    
    return F, fmlib.FoM(F, 1, 2), fmlib.unc(F), delta_v(p), sigma_v


def analysis(t_obs, t_exp, N_ant, Dnu, S_area, fwhm, delta_z, p, criterion, priors=None, ellipse=False, all_ellipses=False, rtrn_res=False, rtrn_allres=False, rtrn_ellps=False, ax=None, **kwargs):
    '''Returns analysis information (optimzed redshift bins, expected drifts, their uncertainties, FoM and parameter constraints) for the input setup
    If ellipse=True, it plots the optimized confidence ellipse
    If all_ellipses=True, it plots the confidence ellipses for all combinations of redshifts
    If rtrn_sig=True, it returns the optimized q_0 significance
    If rtrn_allsig=True, it returns the q_0 significances for all combinations of redshifts
    
    t_obs     = observation time [s]
    t_exp     = total experiment time [yrs]
    N_ant     = number of antennas
    Dnu       = channel width [Hz]
    S_area    = survey area [sq deg]
    fwhm      = HI line width [cm/s]
    delta_z   = interval of redshift to integrate dN/dz
    p         = array of model parameters
    priors    = array of priors for H, q, j, H0
    criterion = q0 or fom'''
    
    z1 = np.array([.1, .3])
    z2 = np.array([.1, .5])
    z3 = np.array([.2, .4])
    z4 = np.array([.1, .3, .5])
    z5 = np.array([.2, .4, .6])
    z6 = np.array([.1, .3, .5, .7])

    z_bins  = [z1, z2, z3, z4, z5, z6]
    F       = [analysis_FoM(z_i, t_obs/len(z_i), t_exp, N_ant, Dnu, S_area, fwhm, delta_z, p, priors)[0] for z_i in z_bins]
    FoM     = [analysis_FoM(z_i, t_obs/len(z_i), t_exp, N_ant, Dnu, S_area, fwhm, delta_z, p, priors)[1] for z_i in z_bins]
    unc     = [analysis_FoM(z_i, t_obs/len(z_i), t_exp, N_ant, Dnu, S_area, fwhm, delta_z, p, priors)[2] for z_i in z_bins]
    delta_v = [analysis_FoM(z_i, t_obs/len(z_i), t_exp, N_ant, Dnu, S_area, fwhm, delta_z, p, priors)[3] for z_i in z_bins]
    sigma_v = [analysis_FoM(z_i, t_obs/len(z_i), t_exp, N_ant, Dnu, S_area, fwhm, delta_z, p, priors)[4] for z_i in z_bins]
    if criterion=='q0':
        i       = np.argmin(np.array([sub[1] for sub in unc])) # where the uncertainty of q_z0 is minimum
    if criterion=='fom':
        i       = np.argmax(FoM) # where the FoM is maximum
    print(f't_obs = {t_obs}, N_ant = {N_ant}, z = {z_bins[i]}')

    if ellipse:
        fig, ax = plt.subplots()
        fmlib.draw_ellipse(F[i], 1, 2, delta_chi2=2.3, center=(p[1], p[2]), ax=ax, label=r'1$\sigma$')
        fmlib.draw_ellipse(F[i], 1, 2, delta_chi2=6.17, center=(p[1], p[2]), ax=ax, label=r'2$\sigma$')
        fmlib.draw_ellipse(F[i], 1, 2, delta_chi2=11.8, center=(p[1], p[2]), ax=ax, label=r'3$\sigma$')
        ax.legend()
        ax.set_xlabel(r'$q_0$')
        ax.set_ylabel(r'$j_0$')
        plt.show()

    if all_ellipses:
        fig, ax = plt.subplots()
        for n in range(len(F)):
            fmlib.draw_ellipse(F[n], 1, 2, delta_chi2=2.3, center=(p[1], p[2]), ax=ax, label=z_bins[n])
        ax.legend()
        ax.set_xlabel(r'$q_0$', fontsize=12)
        ax.set_ylabel(r'$j_0$', fontsize=12)
        ax.axvline(x=p[1], color='gray', linestyle=':', linewidth=1)
        ax.axhline(y=p[2], color='gray', linestyle=':', linewidth=1)
        #ax.set_title(f'$\\Lambda$CDM ($q_0$), $N_{{ant}}={N_ant}$, $t_{{obs}}={t_obs}$ s')
        #fig.savefig('all_ellipses_LCDM_q0.png', dpi=300)
        plt.show()

    if rtrn_res:
        if criterion=='q0':
            return abs(p[1]/unc[i][1])
        if criterion=='fom':
            return FoM[i]
    
    if rtrn_allres:
        if criterion=='q0':
            return np.array([abs(p[1]/sub[1]) for sub in unc])
        if criterion=='fom':
            return FoM
        
    if rtrn_ellps:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        fmlib.draw_ellipse(F[i], 1, 2, delta_chi2=2.3, center=(p[1], p[2]), ax=ax, **kwargs)
        return fig, ax

    return f'\n=== Analysis ===\nDnu: {Dnu} Hz\nS_area: {S_area} sq deg\nArray of redshifts: {z_bins[i]}\nTotal observation time: {tellib.t_obs_tot(t_obs, S_area*len(z_bins[i])):.0f} days\nExpected drift: {delta_v[i]} [cm/s]\nMeasured Error: {sigma_v[i]} [cm/s]\nSignificance: {delta_v[i]/sigma_v[i]}\nFigure of Merit: {FoM[i]}\nUncertainties of H0, q0, j0: {unc[i]}\nSignificances of H0, q0, j0: {abs(p/unc[i])}\n'
    
#fig, ax = plt.subplots()
#analysis(330, t_exp_def, 144, 1e-3, 3500, fwhm_def, delta_z_def, p_LCDM_0, 'q0', priors_baseline_0, rtrn_ellps=True, ax=ax, label=r'$N_{ant}=144$, $t_{obs}=330$ s')
#analysis(169, t_exp_def, 197, 1e-3, 3500, fwhm_def, delta_z_def, p_LCDM_0, 'q0', priors_baseline_0, rtrn_ellps=True, ax=ax, label=r'$N_{ant}=197$, $t_{obs}=169$ s')
#analysis(338, t_exp_def, 197, 1e-3, 3500, fwhm_def, delta_z_def, p_LCDM_0, 'q0', priors_baseline_0, rtrn_ellps=True, ax=ax, label=r'$N_{ant}=197$, $t_{obs}=338$ s')
#analysis(169, t_exp_def, 197, 1e-3, 3500, fwhm_def, delta_z_def, p_LCDM_0, 'q0', priors_baseline_0, rtrn_ellps=True, ax=ax, label=r'$\Lambda$CDM ($q_0$), $N_{ant}=197$, $t_{obs}=169$ s', color='tab:blue', linestyle='--')
#analysis(253, t_exp_def, 197, 1e-3, 3500, fwhm_def, delta_z_def, p_DCD_0, 'fom', priors_baseline_0, rtrn_ellps=True, ax=ax, label=r'DESI+CMB+DESY5 (FoM), $N_{ant}=197$, $t_{obs}=253$ s', color='tab:orange', linestyle='-')
#analysis(169, t_exp_def, 197, 1e-3, 3500, fwhm_def, delta_z_def, p_DCD_0, 'q0', priors_baseline_0, rtrn_ellps=True, ax=ax, label=r'DESI+CMB+DESY5 ($q_0$), $N_{ant}=197$, $t_{obs}=169$ s', color='tab:orange', linestyle='--')
#ax.legend()
#ax.axvline(x=p_LCDM_0[1], color='gray', linestyle=':', linewidth=1)
#ax.axhline(y=p_LCDM_0[2], color='gray', linestyle=':', linewidth=1)
#ax.axvline(x=p_DCD_0[1], color='gray', linestyle=':', linewidth=1)
#ax.axhline(y=p_DCD_0[2], color='gray', linestyle=':', linewidth=1)
#ax.set_xlabel(r'$q_0$', fontsize=12)
#ax.set_ylabel(r'$j_0$', fontsize=12)
#ax.set_title(r'$\Lambda$CDM ($q_0$)')
#fig.savefig('ellipses_LCDM_q0.png', dpi=300)
#plt.show()


print(analysis(169, t_exp_def, 197, 1e-3, 3500, fwhm_def, delta_z_def, p_LCDM_0, 'q0', priors_baseline_0)) # for cosmography centered at z=0
#print(analysis(253, t_exp_def, 197, 1e-3, 3500, fwhm_def, delta_z_def, p_LCDM, 'fom', priors_baseline, all_ellipses=True)) # for generalized cosmography


#
# === unc plots ===
#

t_obs_val = np.linspace(100,500,100)
N_ant_val = np.arange(144,198)

def im_res(t_exp, Dnu, S_area, delta_z, p, criterion, priors, savefig=False):

    arr = np.zeros((len(t_obs_val),len(N_ant_val)), dtype=float)

    for j, t_obs in enumerate(t_obs_val):
        for k, N_ant in enumerate(N_ant_val):
            arr[j,k] = analysis(t_obs, t_exp, N_ant, Dnu, S_area, fwhm_def, delta_z, p, criterion, priors, rtrn_res=True)

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(arr,
                   origin='lower',
                   extent=(N_ant_val[0], N_ant_val[-1], t_obs_val[0], t_obs_val[-1]),
                   aspect='auto')
    if criterion=='q0':
        label = '$q_0$ significance'
    if criterion=='fom':
        label = 'FoM'
    cbar = plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel(r'$N_{ant}$')
    ax.set_xticks(np.linspace(150,190,5))
    ax.set_ylabel(r'$t_{obs}$ [s]')
    ax.set_title(f'$\\Lambda$CDM (FoM)\n$t_{{exp}}$ = {t_exp} yrs, $\\Delta\\nu$ = {Dnu} Hz, $S_{{area}}$ = {S_area} sq deg', fontsize=10, pad=15)
    fig.tight_layout()
    if savefig:
        fig.savefig(f'im_res_{p[0]}_{criterion}.png', dpi=300, bbox_inches="tight")
    plt.show()

#im_res(t_exp_def, 1e-3, 3500, delta_z_def, p_LCDM_0, 'fom', priors_baseline_0, savefig=False) # for cosmography centered at z=0
#im_res(t_exp_def, 1e-3, 3500, delta_z_def, p_LCDM, 'fom', priors_baseline, savefig=False) # for generalized cosmography


#
# === unc plots for each strategy ===
#

def plot_res_each(t_obs_fixed, N_ant_fixed, t_exp, Dnu, S_area, delta_z, p, criterion):
    fig, ax = plt.subplots(1, 2, figsize=(6,4))
    for n in range(9):
        sig_t = [analysis(t_obs, t_exp, N_ant_fixed, Dnu, S_area, fwhm_def, delta_z, p, criterion, priors_baseline, rtrn_allres=True)[n] for t_obs in t_obs_val]
        sig_N = [analysis(t_obs_fixed, t_exp, N_ant, Dnu, S_area, fwhm_def, delta_z, p, criterion, priors_baseline, rtrn_allres=True)[n] for N_ant in N_ant_val]
        ax[0].plot(t_obs_val, np.array(sig_t), label=f'Strategy {n+1}')
        ax[1].plot(N_ant_val, sig_N, label=f'Strategy {n+1}')
    ax[0].legend()
    if criterion=='q0':
        label = '$q_0$ significance'
    if criterion=='fom':
        label = 'FoM'
    ax[0].set_xlabel(r't_{obs} [s]')
    ax[0].set_ylabel(label)
    ax[0].set_title(f'$N_{{ant}}$ = {N_ant_fixed}')
    ax[1].legend()
    ax[1].set_xlabel(r'$N_{ant}$')
    ax[1].set_ylabel(label)
    ax[1].set_title(f'$t_{{obs}}$ = {t_obs_fixed} s')
    plt.tight_layout()
    plt.show()

#plot_res_each(169, 197, t_exp_def, 1e-3, 10000, delta_z_def, p_LCDM_0, 'q0') # for cosmography centered at z=0
#plot_res_each(169, 197, t_exp_def, 1e-3, 10000, delta_z_def, p_LCDM, 'q0') # for generalized cosmography



##########################################################################################################################################

def hrk_analysis_results(z, t_obs, t_exp, N_ant, Dnu, S_area, fwhm, delta_z, p, priors=None):
    '''Returns analysis information (expected drifts, their uncertainties, FoM and parameter constraints) for the input setup
    
    z       = array of redshift bins
    t_obs   = observation time [s]
    t_exp   = total experiment time [yrs]
    N_ant   = number of antennas
    Dnu     = channel width [Hz]
    S_area  = survey area [sq deg]
    fwhm    = HI line width [cm/s]
    delta_z = interval of redshift to integrate dN/dz
    p       = array of model parameters
    priors  = array of priors for H0, q0, j0'''

    FMANA   = analysis_FoM(z, t_obs, t_exp, N_ant, Dnu, S_area, fwhm, delta_z, p, priors)
    FoM     = FMANA[1]
    unc     = FMANA[2]
    delta_v = FMANA[3]
    sigma_v = FMANA[4]
    
    return f'\n=== Analysis ===\nPriors: {priors}\nRedshift: {z}\nDnu: {Dnu}\nS_area: {S_area}\nExpected drift: {delta_v}\nMeasured Error: {sigma_v}\nFigure of Merit: {FoM} \nUncertainties of H0, q0, j0: {unc}'

#
# === First attempt to do the analyis ===
#

#best_values = dvplt.im_sigmav(z_eg, Dnu_val, S_area_val, t_obs_def, N_ant_def, fwhm_def, delta_z_def, doplot=False)

#best_dnu    = best_values[0][0]
#best_area   = best_values[0][1]


#print(hrk_analysis_results(np.array([.1, .3, .5]), t_obs_def, t_exp_def, N_ant_def, best_dnu, best_area, fwhm_def, delta_z_def, p_LCDM, priors=priors_baseline))