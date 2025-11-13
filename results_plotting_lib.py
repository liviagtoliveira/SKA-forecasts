import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
np.set_printoptions(linewidth=np.inf)

from redshiftdrift_lib import *
from telescope_lib import *
from Fisher_matrix_lib import *


def plot_Nz(z_val, Dnu_val, S_area_val, t_obs, N_ant):
    '''
    Creates a panel of plots N(z) for different values of channel width and survey area (Dnu in the lines and S_area in the columns)

    z_val      = values of redshift to calculate N
    Dnu_val    = values of channel width to plot [Hz]
    S_area_val = values of survey area to plot [sq deg]
    t_obs      = observation time used to calculate N [s]
    N_ant      = number of antennas used to calculate N
    '''
    
    n       = len(Dnu_val)
    m       = len(S_area_val)
    fig, ax = plt.subplots(n,m,figsize=(6*m,4*n))

    for i, Dnu in enumerate(Dnu_val):
        rms = S_rms_func(t_obs, N_ant, Dnu)
        for j, S_area in enumerate(S_area_val):
            N = [N_func(z, rms,S_area)[0] for z in z_val]
            ax[i,j].scatter(z_val, N)
            ax[i,j].set_xlabel('z')
            ax[i,j].set_ylabel('N')

    for i, a in enumerate(ax[:, 0]):
        a.set_ylabel(f"N\nDnu = {Dnu_val[i]} Hz", fontsize=12, rotation=0, labelpad=50, va='center')
    for j, a in enumerate(ax[0]):
        a.set_title(f"S_area = {S_area_val[j]} sq deg", fontsize=12)

    fig.tight_layout()
    plt.show()
    print()


def plot_sigmav(z_val, Dnu_val, S_area_val, t_obs, N_ant, fwhm):
    '''
    Creates a panel of plots sigma_v(z) for different values of channel width and survey area (Dnu in the lines and S_area in the columns)

    z_val      = values of redshift to calculate sigma_v
    Dnu_val    = values of channel width to plot [Hz]
    S_area_val = values of survey area to plot [sq deg]
    t_obs      = observation time used to calculate sigma_v [s]
    N_ant      = number of antennas used to calculate sigma_v
    fwhm       = HI line width [cm/s]
    '''

    n = len(Dnu_val)
    m = len(S_area_val)
    fig, ax = plt.subplots(n,m,figsize=(6*m,4*n))

    for i, Dnu in enumerate(Dnu_val):
        for j, S_area in enumerate(S_area_val):
            sigmav = [sigma_v_func(z, t_obs, N_ant, Dnu, S_area, fwhm) for z in z_val]
            ax[i,j].scatter(z_val, sigmav)

    for i, a in enumerate(ax[:, 0]):
        a.set_ylabel(f"sigma_v (cm/s)\n[Dnu = {Dnu_val[i]} Hz]")
    for j, a in enumerate(ax[0]):
        a.set_title(f"S_area = {S_area_val[j]} sq deg")
    for j, a in enumerate(ax[-1]):
        a.set_xlabel(f"redshift z")

    fig.tight_layout()
    plt.show()


def im_sigmav(z_eg, Dnu_range, S_area_range, t_obs, N_ant, fwhm, doprtinfo=False, doplot=True):
    '''
    Creates an image of sigma_v values for different channel widths and survey areas if doplot=True
    Otherwise, returns the values of Dnu, S_area and sigma_v for which sigma_v is minimum and maximum

    z_eg         = redshift used to calculate sigma_v
    Dnu_range    = channel width [Hz]
    S_area_range = survey area [sq deg]
    t_obs        = observation time used to calculate sigma_v [s]
    N_ant        = number of antennas used to calculate sigma_v
    fwhm         = HI line width [cm/s]
    doprtinfo    = print information of sigma_v and input parameters
    doplot       = plot the image of sigma_v
    '''

    if len(Dnu_range) == len(S_area_range):
        Dnu_val    = Dnu_range
        S_area_val = S_area_range
        N          = len(Dnu_range) 
    else:
        print('CAUTION ranges are not aquidistant')
        N          = np.max([len(Dnu_range),len(S_area_range)])
        Dnu_val    = np.linspace(np.min(Dnu_range),np.max(Dnu_range), N)
        S_area_val = np.linspace(np.min(S_area_range),np.max(S_area_range), N)

    arr        = np.zeros((N,N), dtype=float)
    for i, Dnu in enumerate(Dnu_val):
        for j, S_area in enumerate(S_area_val):
            arr[i,j] = sigma_v_func(z_eg, t_obs, N_ant, Dnu, S_area, fwhm)

    # Determine the minimum and maximum
    #
    ind_min = np.unravel_index(np.argmin(arr, axis=None), arr.shape)
    ind_max = np.unravel_index(np.argmax(arr, axis=None), arr.shape)
    
    if doprtinfo:
        print('=== Info of sigma_v and the input parameters ===\n')
        print('\tDnu range      [Hz] : ',Dnu_val)
        print('\tArea range [sq deg] : ',S_area_val)
        print('\tMaximum  ','Dnu [Hz]',Dnu_val[ind_max[0]],' | Sky Area [sq deg] ',S_area_val[ind_max[1]],' | sigma_v [cm]',arr[ind_max] )
        print('\tMinimum  ','Dnu [Hz]',Dnu_val[ind_min[0]],' | Sky Area [sq deg] ',S_area_val[ind_min[1]],' | sigma_v [cm]',arr[ind_min] )

    if doplot:
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(arr,
                           origin='lower',
                           extent=(S_area_val[0], S_area_val[-1], Dnu_val[0], Dnu_val[-1]),
                           aspect='auto')
        cbar = plt.colorbar(im, ax=ax, label=r'$\sigma_v$ [cm/s]')
        ax.set_xlabel(r'$S_{area}$ [sq deg]')
        ax.set_ylabel(r'$\Delta \nu$ [Hz]')
        plt.show()

    else:
        return [Dnu_val[ind_min[0]],S_area_val[ind_min[1]],arr[ind_min]],[Dnu_val[ind_max[0]],S_area_val[ind_max[1]],arr[ind_max]]
        

def plot_Dv(z_val, Dnu_val, S_area_val, t_obs, t_exp, N_ant, fwhm, p):
    '''
    Creates a panel of plots ∆v(z) for different values of channel width and survey area (Dnu in the lines and S_area in the columns)

    z_val      = values of redshift to calculate ∆v
    Dnu_val    = values of channel width to plot [Hz]
    S_area_val = values of survey area to plot [sq deg]
    t_obs      = observation time used to calculate ∆v [s]
    t_exp      = total experiment time [yrs]
    N_ant      = number of antennas used to calculate ∆v
    fwhm       = HI line width [cm/s]
    p          = array of H0, q0, j0 values
    '''

    n = len(Dnu_val)
    m = len(S_area_val)
    fig, ax = plt.subplots(n,m,figsize=(6*m,4*n))
    Dv = [delta_v_func(z, p, t_exp) for z in z_val]
    
    for i, Dnu in enumerate(Dnu_val):
        for j, S_area in enumerate(S_area_val):
            sigmav = [sigma_v_func(z, t_obs, N_ant, Dnu, S_area, fwhm) for z in z_val]
            ax[i,j].errorbar(z_val, Dv, yerr=sigmav, fmt='.')

    for i, a in enumerate(ax[:, 0]):
        a.set_ylabel(f"Dv (cm/s)\n[Dnu = {Dnu_val[i]} Hz]")
    for j, a in enumerate(ax[0]):
        a.set_title(f"S_area = {S_area_val[j]} sq deg")
    for j, a in enumerate(ax[-1]):
        a.set_xlabel(f"redshift z")

    fig.tight_layout()
    plt.show()


def plot_vsignificance(z_val, Dnu_val, S_area_val, t_obs, t_exp, N_ant, fwhm, p):
    '''
    Creates a panel of plots sigma_v(z) for different values of channel width and survey area (Dnu in the lines and S_area in the columns)

    z_val.     = values of redshift to calculate ∆v and sigma_v
    Dnu_val    = values of channel width to plot [Hz]
    S_area_val = values of survey area to plot [sq deg]
    t_obs      = observation time used to calculate sigma_v [s]
    t_exp      = total experiment time used to calulate ∆v [yrs]
    N_ant      = number of antennas used to calculate sigma_v
    fwhm       = HI line width [cm/s]
    p          = array of H0, q0, j0 values
    '''

    n = len(Dnu_val)
    m = len(S_area_val)
    fig, ax = plt.subplots(n,m,figsize=(6*m,4*n))
    Dv = np.array([delta_v_func(z, p, t_exp) for z in z_val])
    
    for i, Dnu in enumerate(Dnu_val):
        for j, S_area in enumerate(S_area_val):
            sigmav = np.array([sigma_v_func(z, t_obs, N_ant, Dnu, S_area, fwhm) for z in z_val])
            ax[i,j].scatter(z_val, Dv/sigmav)

    for i, a in enumerate(ax[:, 0]):
        a.set_ylabel(f"Dv significance\n[Dnu = {Dnu_val[i]} Hz]")
    for j, a in enumerate(ax[0]):
        a.set_title(f"S_area = {S_area_val[j]} sq deg")
    for j, a in enumerate(ax[-1]):
        a.set_xlabel(f"redshift z")

    fig.tight_layout()
    plt.show()

    
def im_vsignificance(z_eg, Dnu_min, Dnu_max, S_area_min, S_area_max, t_obs, t_exp, N_ant, fwhm, p, N=100):
    '''
    Creates an image of v significance values for different channel widths and survey areas

    z_eg                   = redshift used to calculate ∆v and sigma_v
    Dnu_min, Dnu_max       = limits of channel width [Hz]
    S_area_min, S_area_max = limits of survey area [sq deg]
    t_obs                  = observation time used to calculate sigma_v [s]
    t_exp                  = total experiment time used to calulate ∆v [yrs]
    N_ant                  = number of antennas used to calculate sigma_v
    fwhm                   = HI line width [cm/s]
    p                      = array of H0, q0, j0 values
    N                      = number of points in Dnu and S_area to create image
    '''

    Dv          = delta_v_func(z_eg, p, t_exp)
    Dnu_val     = np.linspace(Dnu_min, Dnu_max, N)
    S_area_val  = np.linspace(S_area_min, S_area_max, N)
    arr         = np.zeros((N,N), dtype=float)

    for i, Dnu in enumerate(Dnu_val):
        for j, S_area in enumerate(S_area_val):
            arr[i,j] = Dv/sigma_v_func(z_eg, t_obs, N_ant, Dnu, S_area, fwhm)

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(arr,
                   origin='lower',
                   extent=(S_area_val[0], S_area_val[-1], Dnu_val[0], Dnu_val[-1]),
                   aspect='auto')
    cbar = plt.colorbar(im, ax=ax, label=r'$\Delta$v significance')
    ax.set_xlabel(r'$S_{area}$ [sq deg]')
    ax.set_ylabel(r'$\Delta \nu$ [Hz]')
    plt.show()


def plot_ellipses(z_val, Dnu_val, S_area_val, t_obs, t_exp, N_ant, fwhm, p, priors=None, savefig=False):
    '''
    Creates a panel of confidence ellipses for different values of channel width and survey area (Dnu in the lines and S_area in the columns)

    z_val      = values of redshift to calculate ∆v and sigma_v
    Dnu_val    = values of channel width to plot [Hz]
    S_area_val = values of survey area to plot [sq deg]
    t_obs      = observation time used to calculate sigma_v [s]
    t_exp      = total experiment time used to calulate ∆v [yrs]
    N_ant      = number of antennas used to calculate sigma_v
    fwhm       = HI line width [cm/s]
    p          = array of H0, q0, j0 values
    priors     = array of the priors to be used (same length and order as p)
    '''
    
    n = len(Dnu_val)
    m = len(S_area_val)
    fig, ax = plt.subplots(n,m,figsize=(6*m,4*n), sharex=True, sharey=True)

    for i, Dnu in enumerate(Dnu_val):
        for j, S_area in enumerate(S_area_val):
            sigma_v_vect = np.vectorize(lambda z_i: sigma_v_func(z_i, t_obs, N_ant, Dnu, S_area, fwhm))
            sigma_v = sigma_v_vect(z_val)
            delta_v = lambda p: delta_v_func(z_val, p, t_exp)
            F = Fisher_matrix(p, delta_v, sigma_v, priors)
            fom = FoM(F, 1, 2)
            draw_ellipse(F, 1, 2, delta_chi2=2.3, center=(p[1], p[2]), ax=ax[i,j], label=r'1$\sigma$')
            draw_ellipse(F, 1, 2, delta_chi2=6.17, center=(p[1], p[2]), ax=ax[i,j], label=r'2$\sigma$')
            draw_ellipse(F, 1, 2, delta_chi2=11.8, center=(p[1], p[2]), ax=ax[i,j], label=r'3$\sigma$')
            ax[i,j].legend()
            ax[i,j].text(-8, -40, f'FoM = {fom:.2f}')

    for i, a in enumerate(ax[:, 0]):
        a.set_ylabel(f"$j_0$\n[Dnu = {Dnu_val[i]} Hz]", fontsize=12)
    for j, a in enumerate(ax[0]):
        a.set_title(f"S_area = {S_area_val[j]} sq deg")
    for j, a in enumerate(ax[-1]):
        a.set_xlabel(f"$q_0$")

    fig.tight_layout()
    if savefig:
        fig.savefig('ellipses.png')
    plt.show()