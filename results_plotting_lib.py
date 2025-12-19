import math
import numpy as np
import matplotlib.pyplot as plt

from redshiftdrift_lib import *
from telescope_lib import *
from Fisher_matrix_lib import *


def plot_rms(z_val, Dnu_val, t_obs, N_ant_val, fwhm):
    '''
    Creates a panel of plots S_rms for different values of channel width and survey area (Dnu in the lines and S_area in the columns)

    z_val      = values of redshift to calculate dv
    Dnu_val    = values of channel width to plot [Hz]
    S_area_val = values of survey area to plot [sq deg]
    t_obs      = observation time used to calculate S_rms [s]
    N_ant_val  = numbers of antennas used to calculate S_rms
    fwhm       = HI line width [cm/s]
    '''
    
    n       = len(Dnu_val)
    m       = len(N_ant_val)
    fig, ax = plt.subplots(n,m,figsize=(6*m,4*n))

    for i, Dnu in enumerate(Dnu_val):
        for j, N_ant in enumerate(N_ant_val):
            for z in z_val:
                p4 = fwhm / (2*np.sqrt(np.log(2))) 
                dv = Dnu2dv(Dnu,z)
                w20_HI_galaxy = 3.6 * p4 * 1/np.sqrt(2)
                Single_channel_detection  = w20_HI_galaxy/dv
                if Single_channel_detection > 1:
                    Single_channel_detection_factor  = 1/np.sqrt(Single_channel_detection)
                else:
                    Single_channel_detection_factor  = 1
                rms = S_rms_func(t_obs, N_ant, Dnu) * Single_channel_detection_factor
                ax[i,j].scatter(z, rms, color='tab:blue')
            ax[i,j].set_xlabel('z')
            ax[i,j].set_ylabel(r'S_rms [$\mu$Jy]')

    for i, a in enumerate(ax[:, 0]):
        a.set_ylabel(f"S_rms\nDnu = {Dnu_val[i]} Hz", fontsize=12, rotation=0, labelpad=50, va='center')
    for j, a in enumerate(ax[0]):
        a.set_title(f"N_ant = {N_ant_val[j]}", fontsize=12)

    fig.tight_layout()
    plt.show()


def im_rms(z_eg, Dnu, t_obs_min, t_obs_max, N_ant_min, N_ant_max, fwhm, N=100):
    '''
    Creates an image of v significance values for different channel widths and survey areas

    z_eg                 = redshift used to calculate ∆v and sigma_v
    Dnu                  = channel width used to calculate S_rms [Hz]
    t_obs_min, t_obs_max = limits of observation time [h]
    N_ant_min, N_ant_max = limits of number of antennas 
    fwhm                 = HI line width [cm/s]
    N                    = number of points in Dnu and S_area to create image
    '''

    t_obs_val = np.linspace(t_obs_min, t_obs_max, N)
    N_ant_val = np.arange(N_ant_min, N_ant_max+1)
    arr       = np.zeros((N,len(N_ant_val)), dtype=float)

    for i, t_obs in enumerate(t_obs_val):
        for j, N_ant in enumerate(N_ant_val):
            p4 = fwhm / (2*np.sqrt(np.log(2))) 
            dv = Dnu2dv(Dnu,z_eg)
            w20_HI_galaxy = 3.6 * p4 * 1/np.sqrt(2)
            Single_channel_detection  = w20_HI_galaxy/dv
            if Single_channel_detection > 1:
                Single_channel_detection_factor  = 1/np.sqrt(Single_channel_detection)
            else:
                Single_channel_detection_factor  = 1
            arr[i,j] = S_rms_func(t_obs*3600, N_ant, Dnu) * Single_channel_detection_factor

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(arr,
                   origin='lower',
                   extent=(N_ant_val[0], N_ant_val[-1], t_obs_val[0], t_obs_val[-1]),
                   aspect='auto')
    cbar = plt.colorbar(im, ax=ax, label=r'$S_{rms}$ [$\mu$Jy]')
    ax.set_xlabel(r'$N_{ant}$')
    ax.set_ylabel(r'$t_{obs}$ [h]')
    ax.set_title(f'z = {z_eg}, Dnu = {Dnu} Hz')
    plt.show()


def plot_Nz_panel(z_val, Dnu_val, S_area_val, t_obs, N_ant, fwhm, delta_z):
    '''
    Creates a panel of plots N(z) for different values of channel width and survey area (Dnu in the lines and S_area in the columns)

    z_val      = values of redshift to calculate N
    Dnu_val    = values of channel width to plot [Hz]
    S_area_val = values of survey area to plot [sq deg]
    t_obs      = observation time used to calculate N [s]
    N_ant      = number of antennas used to calculate N
    fwhm       = HI line width [cm/s]
    delta_z    = delta z for dN/dz integration
    '''
    
    n       = len(Dnu_val)
    m       = len(S_area_val)
    fig, ax = plt.subplots(n,m,figsize=(6*m,4*n))

    for i, Dnu in enumerate(Dnu_val):
        for j, S_area in enumerate(S_area_val):
            for z in z_val:
                p4 = fwhm / (2*np.sqrt(np.log(2))) 
                dv = Dnu2dv(Dnu,z)
                w20_HI_galaxy = 3.6 * p4 * 1/np.sqrt(2)
                Single_channel_detection  = w20_HI_galaxy/dv
                if Single_channel_detection > 1:
                    Single_channel_detection_factor  = 1/np.sqrt(Single_channel_detection)
                else:
                    Single_channel_detection_factor  = 1
                rms = S_rms_func(t_obs, N_ant, Dnu) * Single_channel_detection_factor
                N = N_func(z, rms, S_area, delta_z)[0]
                ax[i,j].scatter(z, N, color='tab:blue')
            ax[i,j].set_xlabel('z')
            ax[i,j].set_ylabel('N')

    for i, a in enumerate(ax[:, 0]):
        a.set_ylabel(f"N\nDnu = {Dnu_val[i]} Hz", fontsize=12, rotation=0, labelpad=50, va='center')
    for j, a in enumerate(ax[0]):
        a.set_title(f"S_area = {S_area_val[j]} sq deg", fontsize=12)

    fig.tight_layout()
    plt.show()


def plot_Nz(z_val, rms, delta_z, color, label):
    '''
    Creates a panel of plots N(z) for different values of channel width and survey area (Dnu in the lines and S_area in the columns)

    z_val   = values of redshift to calculate N
    Dnu     = channel width used to calculate S_rms [Hz]
    S_area  = survey area used to calculate N [sq deg]
    t_obs   = observation time used to calculate N [s]
    N_ant   = number of antennas used to calculate N
    fwhm    = HI line width [cm/s]
    delta_z = delta z for dN/dz integration
    '''
    
    N_vect = np.vectorize(lambda z: N_func(z, rms, 1, delta_z)[0])
    N = N_vect(z_val)
    plt.scatter(z_val, N, color=color)

    z_cont = np.linspace(z_val[0], z_val[-1], 100)
    N_cont = N_vect(z_cont)
    plt.plot(z_cont, N_cont, color=color, label=label)


def plot_sigmav(z_val, Dnu_val, S_area_val, t_obs, N_ant, fwhm, delta_z):
    '''
    Creates a panel of plots sigma_v(z) for different values of channel width and survey area (Dnu in the lines and S_area in the columns)

    z_val      = values of redshift to calculate sigma_v
    Dnu_val    = values of channel width to plot [Hz]
    S_area_val = values of survey area to plot [sq deg]
    t_obs      = observation time used to calculate sigma_v [s]
    N_ant      = number of antennas used to calculate sigma_v
    fwhm       = HI line width [cm/s]
    delta_z    = delta z for dN/dz integration
    '''

    n = len(Dnu_val)
    m = len(S_area_val)
    fig, ax = plt.subplots(n,m,figsize=(6*m,4*n))

    for i, Dnu in enumerate(Dnu_val):
        for j, S_area in enumerate(S_area_val):
            sigmav = [sigma_v_func(z, t_obs, N_ant, Dnu, S_area, fwhm, delta_z) for z in z_val]
            ax[i,j].scatter(z_val, sigmav)

    for i, a in enumerate(ax[:, 0]):
        a.set_ylabel(f"sigma_v (cm/s)\n[Dnu = {Dnu_val[i]} Hz]")
    for j, a in enumerate(ax[0]):
        a.set_title(f"S_area = {S_area_val[j]} sq deg")
    for j, a in enumerate(ax[-1]):
        a.set_xlabel(f"redshift z")

    fig.tight_layout()
    plt.show()


def im_sigmav(z_eg, Dnu_range, S_area_range, t_obs, N_ant, fwhm, delta_z, doprtinfo=False, doplot=True):
    '''
    Creates an image of sigma_v values for different channel widths and survey areas if doplot=True
    Otherwise, returns the values of Dnu, S_area and sigma_v for which sigma_v is minimum and maximum

    z_eg         = redshift used to calculate sigma_v
    Dnu_range    = channel width [Hz]
    S_area_range = survey area [sq deg]
    t_obs        = observation time used to calculate sigma_v [s]
    N_ant        = number of antennas used to calculate sigma_v
    fwhm         = HI line width [cm/s]
    delta_z      = delta z for dN/dz integration
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
            arr[i,j] = sigma_v_func(z_eg, t_obs, N_ant, Dnu, S_area, fwhm, delta_z)

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
        

def plot_Dv(z_val, t_exp, p):
    '''
    Creates a panel of plots ∆v(z) for different values of channel width and survey area (Dnu in the lines and S_area in the columns)

    z_val      = values of redshift to calculate ∆v
    t_exp      = total experiment time [yrs]
    p          = array of H0, q0, j0 values
    '''

    Dv = [delta_v_func(z, p, t_exp) for z in z_val]
    z_cont = np.linspace(z_val[0], z_val[-1], 100)
    Dv_cont = [delta_v_func(z, p, t_exp) for z in z_cont]
    plt.scatter(z_val, Dv)
    plt.plot(z_cont, Dv_cont)
    plt.xlabel('z')
    plt.ylabel(r'$\Delta$v [cm/s]')
    plt.title(f't_exp = {t_exp} yrs')
    plt.show()


def plot_Dv_errbar(z_val, Dnu_val, S_area_val, t_obs, t_exp, N_ant, fwhm, p, delta_z):
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
    delta_z    = delta z for dN/dz integration
    '''

    n = len(Dnu_val)
    m = len(S_area_val)
    fig, ax = plt.subplots(n,m,figsize=(6*m,4*n))
    Dv = [delta_v_func(z, p, t_exp) for z in z_val]
    
    for i, Dnu in enumerate(Dnu_val):
        for j, S_area in enumerate(S_area_val):
            sigmav = [sigma_v_func(z, t_obs, N_ant, Dnu, S_area, fwhm, delta_z) for z in z_val]
            ax[i,j].errorbar(z_val, Dv, yerr=sigmav, fmt='.')

    for i, a in enumerate(ax[:, 0]):
        a.set_ylabel(f"Dv (cm/s)\n[Dnu = {Dnu_val[i]} Hz]")
    for j, a in enumerate(ax[0]):
        a.set_title(f"S_area = {S_area_val[j]} sq deg")
    for j, a in enumerate(ax[-1]):
        a.set_xlabel(f"redshift z")

    fig.tight_layout()
    plt.show()


def plot_vsignificance(z_val, Dnu_val, S_area_val, t_obs, t_exp, N_ant, fwhm, p, delta_z):
    '''
    Creates a panel of plots sigma_v(z) for different values of channel width and survey area (Dnu in the lines and S_area in the columns)

    z_val      = values of redshift to calculate ∆v and sigma_v
    Dnu_val    = values of channel width to plot [Hz]
    S_area_val = values of survey area to plot [sq deg]
    t_obs      = observation time used to calculate sigma_v [s]
    t_exp      = total experiment time used to calulate ∆v [yrs]
    N_ant      = number of antennas used to calculate sigma_v
    fwhm       = HI line width [cm/s]
    p          = array of H0, q0, j0 values
    delta_z    = delta z for dN/dz integration
    '''

    n = len(Dnu_val)
    m = len(S_area_val)
    fig, ax = plt.subplots(n,m,figsize=(6*m,4*n))
    Dv = np.array([delta_v_func(z, p, t_exp) for z in z_val])
    
    for i, Dnu in enumerate(Dnu_val):
        for j, S_area in enumerate(S_area_val):
            sigmav = np.array([sigma_v_func(z, t_obs, N_ant, Dnu, S_area, fwhm, delta_z) for z in z_val])
            ax[i,j].scatter(z_val, Dv/sigmav)

    for i, a in enumerate(ax[:, 0]):
        a.set_ylabel(f"Dv significance\n[Dnu = {Dnu_val[i]} Hz]")
    for j, a in enumerate(ax[0]):
        a.set_title(f"S_area = {S_area_val[j]} sq deg")
    for j, a in enumerate(ax[-1]):
        a.set_xlabel(f"redshift z")
    
    fig.suptitle(f't_obs = {t_obs} s, t_exp = {t_exp} yrs, N_ant = {N_ant}')

    fig.tight_layout()
    plt.show()

    
def im_vsignificance(z_eg, Dnu_min, Dnu_max, S_area_min, S_area_max, t_obs, t_exp, N_ant, fwhm, p, delta_z, N=100):
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
    delta_z                = delta z for dN/dz integration
    N                      = number of points in Dnu and S_area to create image
    '''

    Dv          = delta_v_func(z_eg, p, t_exp)
    Dnu_val     = np.linspace(Dnu_min, Dnu_max, N)
    S_area_val  = np.linspace(S_area_min, S_area_max, N)
    arr         = np.zeros((N,N), dtype=float)

    for i, Dnu in enumerate(Dnu_val):
        for j, S_area in enumerate(S_area_val):
            arr[i,j] = Dv/sigma_v_func(z_eg, t_obs, N_ant, Dnu, S_area, fwhm, delta_z)

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(arr,
                   origin='lower',
                   extent=(S_area_val[0], S_area_val[-1], Dnu_val[0], Dnu_val[-1]),
                   aspect='auto')
    cbar = plt.colorbar(im, ax=ax, label=r'$\Delta$v significance')
    ax.set_xlabel(r'$S_{area}$ [sq deg]')
    ax.set_ylabel(r'$\Delta \nu$ [Hz]')
    ax.set_title(f'z = {z_eg}, t_obs = {t_obs/3600} h, t_exp = {t_exp} yrs, N_ant = {N_ant}')
    plt.show()


def plot_ellps_panel(z_val, Dnu_val, S_area_val, t_obs, t_exp, N_ant, fwhm, p, delta_z, priors=None, savefig=False):
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
    delta_z    = delta z for dN/dz integration
    priors     = array of the priors to be used (same length and order as p)
    savefig    = save the panel as PNG
    '''
    
    n = len(Dnu_val)
    m = len(S_area_val)
    fig, ax = plt.subplots(n,m,figsize=(6*m,4*n), sharex=True, sharey=True)

    for i, Dnu in enumerate(Dnu_val):
        for j, S_area in enumerate(S_area_val):
            sigma_v_vect = np.vectorize(lambda z_i: sigma_v_func(z_i, t_obs, N_ant, Dnu, S_area, fwhm, delta_z))
            sigma_v = sigma_v_vect(z_val)
            delta_v = lambda p: delta_v_func(z_val, p, t_exp)
            F = Fisher_matrix(p, delta_v, sigma_v, priors)
            fom = FoM(F, 1, 2)
            draw_ellipse(F, 1, 2, delta_chi2=2.3, center=(p[1], p[2]), ax=ax[i,j], label=r'1$\sigma$')
            draw_ellipse(F, 1, 2, delta_chi2=6.17, center=(p[1], p[2]), ax=ax[i,j], label=r'2$\sigma$')
            draw_ellipse(F, 1, 2, delta_chi2=11.8, center=(p[1], p[2]), ax=ax[i,j], label=r'3$\sigma$')
            ax[i,j].legend()
            ax[i,j].text(0.1, 0.1, f'FoM = {fom:.2f}', fontsize=12, transform=ax[i,j].transAxes, verticalalignment='bottom', horizontalalignment='left')

    for i, a in enumerate(ax[:, 0]):
        a.set_ylabel(f"Dnu = {Dnu_val[i]} Hz\n$\\mathbf{{j_0}}$", fontsize=12)
    for j, a in enumerate(ax[0]):
        a.set_title(f"S_area = {S_area_val[j]} sq deg")
    for j, a in enumerate(ax[-1]):
        a.set_xlabel(f"$\\mathbf{{q_0}}$")

    fig.tight_layout()
    if savefig:
        fig.savefig('ellipses_panel.png')
    plt.show()


def plot_ellipses(z_val, Dnu_val, S_area, t_obs, t_exp, N_ant, fwhm, p, delta_z, priors=None, savefig=False):
    '''
    Creates a panel of confidence ellipses for different values of channel width and survey area (Dnu in the lines and S_area in the columns)

    z_val      = values of redshift to calculate ∆v and sigma_v
    Dnu_val    = values of channel width to plot [Hz]
    S_area     = survey area to plot [sq deg]
    t_obs      = observation time used to calculate sigma_v [s]
    t_exp      = total experiment time used to calulate ∆v [yrs]
    N_ant      = number of antennas used to calculate sigma_v
    fwhm       = HI line width [cm/s]
    p          = array of H0, q0, j0 values
    delta_z    = delta z for dN/dz integration
    priors     = array of the priors to be used (same length and order as p)
    savefig    = save ellipses as PNG
    '''

    fig, ax = plt.subplots()
    for Dnu in Dnu_val:
        sigma_v_vect = np.vectorize(lambda z_i: sigma_v_func(z_i, t_obs, N_ant, Dnu, S_area, fwhm, delta_z))
        sigma_v = sigma_v_vect(z_val)
        delta_v = lambda p: delta_v_func(z_val, p, t_exp)
        F = Fisher_matrix(p, delta_v, sigma_v, priors)
        fom = FoM(F, 1, 2)
        draw_ellipse(F, 1, 2, delta_chi2=2.3, center=(p[1], p[2]), ax=ax, label=f'{Dnu} Hz (FoM = {fom:.2f})')
    ax.legend()
    ax.set_xlabel(r"$q_0$", fontsize=12)
    ax.set_ylabel(r"$j_0$", fontsize=12)

    if savefig:
        fig.savefig('ellipses.png')
    plt.show()


def plot_dv_ellipses_SKA_book(z_val, Dnu_val, S_area_val, t_obs, t_exp, N_ant, fwhm, p, delta_z, priors=None, savefig=False):
    '''
    Creates a panel of plots sigma_v(z) for different values of channel width and survey area (Dnu in the lines and S_area in the columns)

    z_val      = values of redshift to calculate ∆v and sigma_v
    Dnu_val    = values of channel width to plot [Hz]
    S_area_val = values of survey area to plot [sq deg]
    t_obs      = observation time used to calculate sigma_v [s]
    t_exp      = total experiment time used to calulate ∆v [yrs]
    N_ant      = number of antennas used to calculate sigma_v
    fwhm       = HI line width [cm/s]
    p          = array of H0, q0, j0 values
    delta_z    = delta z for dN/dz integration
    priors     = array of the priors to be used (same length and order as p)
    savefig    = save image as PNG
    '''

    fig, ax = plt.subplots(1,4,figsize=(20,5))

    fontsize = 15
    
    Dv = np.array([delta_v_func(z, p, t_exp) for z in z_val])

    z_val_org = z_val

    for j, S_area in enumerate(S_area_val):
        for i, Dnu in enumerate(Dnu_val):
            z_val  = z_val - j* np.ones(len(z_val)) * 0.001
            sigmav = np.array([sigma_v_func(z, t_obs, N_ant, Dnu, S_area, fwhm,delta_z) for z in z_val])
            ax[j].plot(z_val_org,Dv,marker='',alpha=0.3,color='black',linestyle='--')
            ax[j].errorbar(z_val,Dv,sigmav,marker="x",capsize=7,alpha=0.3,linestyle='',label=str(Dnu)+' Hz')
        ax[j].set_xlabel('z',fontsize=fontsize)
        #ax[i,j].set_ylabel(r'$\Delta$v theoretical and errors [cm/s]')
        ax[j].set_ylim([0,7])
        ax[j].set_xlim([0.15,0.55])
        ax[j].legend(loc="upper left")
        ax[j].xaxis.set_tick_params(labelsize=fontsize)
        ax[j].yaxis.set_tick_params(labelsize=fontsize)
        ax[j].set_title(r'S_area = '+str(S_area_val[j])+' $deg^{2}$', fontsize=fontsize)
        
    ax[0].set_ylabel(r'$Dv \pm \Delta Dv \, [cms^{-1}]$', fontsize=fontsize, rotation=90, labelpad=20, va='center')


    
    for Dnu in Dnu_val:
        sigma_v_vect = np.vectorize(lambda z_i: sigma_v_func(z_i, t_obs, N_ant, Dnu, S_area, fwhm, delta_z))
        sigma_v = sigma_v_vect(z_val)
        delta_v = lambda p: delta_v_func(z_val, p, t_exp)
        F = Fisher_matrix(p, delta_v, sigma_v, priors)
        fom = FoM(F, 1, 2)        
        draw_ellipse(F, 1, 2, delta_chi2=2.3, center=(p[1], p[2]), ax=ax[-1], label=f'{Dnu} Hz (FoM = {fom:.2f})')

    ax[-1].yaxis.set_label_position("right")
    ax[-1].yaxis.tick_right()

    ax[-1].xaxis.set_tick_params(labelsize=fontsize)
    ax[-1].yaxis.set_tick_params(labelsize=fontsize)

    ax[-1].legend()
    ax[-1].set_xlabel(r"$q_0$", fontsize=fontsize)
    ax[-1].set_ylabel(r"$j_0$", fontsize=fontsize)

    if savefig:
        fig.savefig('redshiftdrift_skabook.png')
    plt.show()