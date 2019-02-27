# Systematic testing of Bayesian unfolding using simulated spectra from LaBr3, Plastic, and PIPS detector 
# 
# Created by Andrei R. Hanu

# Libraries to handle ROOT files
import ROOT
import root_numpy

# Theano
import theano
import theano.tensor

# Copy function
import copy

# NumPy
import numpy as np

# PyMC3
import pymc3 as pm

# Color palette library for Python
# How to choose a colour scheme for your data:
# http://earthobservatory.nasa.gov/blogs/elegantfigures/2013/08/05/subtleties-of-color-part-1-of-6/
import palettable

# Matplotlib - 2D plotting library
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
#import seaborn.apionly as sns
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import Grid, AxesGrid

##########################################################################################
# Setting rcParams for publication quality graphs
fig_size =  np.array([7.3,4.2])*1.5
params = {'backend': 'pdf',
        'axes.labelsize': 12,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.major.size': 7,
        'xtick.major.width': 1,
        'xtick.minor.size': 3.5,
        'xtick.minor.width': 1.25,
        'ytick.major.size': 7,
        'ytick.major.width': 1.25,
        'ytick.minor.size': 3.5,
        'ytick.minor.width': 1.25,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Bitstream Vera Sans',
        'font.size': 11,
        'figure.figsize': fig_size}

# Update rcParams
rcParams.update(params)


# Threshold Energy, in keV, above which unfolding will occur
thld_e = 0.

# Configuration
det1 = 'Saint Gobain B380 LaBr3'
det2 = 'Eljen Plastic Detector'
det3 = 'Canberra PD450-15-500AM'

#isotope = 'Cl36'
#isotope = 'Kr85'
isotope = 'Sr90Y90'
#isotope = 'Cs137'
#isotope = 'gamma_Power_10_10000_keV_alpha_-4_electron_Gauss_3000_100_keV'
#isotope = 'gamma_Power_10_10000_keV_alpha_-3_electron_Gauss_600_600_keV'
f_data = isotope + '_R_25_cm_Nr_100000000_ISO.root'


# ## STEP 1 - Import the detector response matrices


# Load the ROOT file containing the response matrix for the detector
f_rspns = ROOT.TFile.Open('./TestData/'+det1+'/Response Matrix/'+det1+'.root')

# Retrieve the electron and gamma-ray energy migration matrices and source vectors (i.e. the true spectrum from which the response matrix was simulated)
# NOTE: Index 0 contains the bin values
#       Index 1 contains the bin edges
src_vec_e = np.asarray(root_numpy.hist2array(f_rspns.Get('Source Spectrum (Electron)'), 
                                             include_overflow=False, copy=True, return_edges=True))

src_vec_gam = np.asarray(root_numpy.hist2array(f_rspns.Get('Source Spectrum (Gamma)'), 
                                               include_overflow=False, copy=True, return_edges=True))

mig_mat_e = np.asarray(root_numpy.hist2array(f_rspns.Get('Energy Migration Matrix (Electron)'), 
                                             include_overflow=False, copy=True, return_edges=True))
    
mig_mat_gam = np.asarray(root_numpy.hist2array(f_rspns.Get('Energy Migration Matrix (Gamma)'), 
                                               include_overflow=False, copy=True, return_edges=True))

# Calculate the response matrices by normalizing the energy migration matrices by the source vectors
rspns_mat_det1_e = copy.deepcopy(mig_mat_e)
rspns_mat_det1_e[0] = np.nan_to_num(rspns_mat_det1_e[0]/src_vec_e[0])
rspns_mat_det1_gam = copy.deepcopy(mig_mat_gam)
rspns_mat_det1_gam[0] = np.nan_to_num(rspns_mat_det1_gam[0]/src_vec_e[0])

# Remove response matrix elements below threshold energy
rspns_mat_det1_e[0] = np.delete(rspns_mat_det1_e[0], np.where(rspns_mat_det1_e[1][0] < thld_e), axis=0)
rspns_mat_det1_e[0] = np.delete(rspns_mat_det1_e[0], np.where(rspns_mat_det1_e[1][0] < thld_e), axis=1)
rspns_mat_det1_e[1] = np.delete(rspns_mat_det1_e[1], np.where(rspns_mat_det1_e[1][0] < thld_e), axis=1)
rspns_mat_det1_gam[0] = np.delete(rspns_mat_det1_gam[0], np.where(rspns_mat_det1_gam[1][0] < thld_e), axis=0)
rspns_mat_det1_gam[0] = np.delete(rspns_mat_det1_gam[0], np.where(rspns_mat_det1_gam[1][0] < thld_e), axis=1)
rspns_mat_det1_gam[1] = np.delete(rspns_mat_det1_gam[1], np.where(rspns_mat_det1_gam[1][0] < thld_e), axis=1)

# Create a combined response matrix
rspns_mat_det1_comb = copy.deepcopy(rspns_mat_det1_e)
rspns_mat_det1_comb[0] += rspns_mat_det1_gam[0]


# Load the ROOT file containing the response matrix for the detector
f_rspns = ROOT.TFile.Open('./TestData/'+det2+'/Response Matrix/'+det2+'.root')

# Retrieve the electron and gamma-ray energy migration matrices and source vectors (i.e. the true spectrum from which the response matrix was simulated)
# NOTE: Index 0 contains the bin values
#       Index 1 contains the bin edges
src_vec_e = np.asarray(root_numpy.hist2array(f_rspns.Get('Source Spectrum (Electron)'), 
                                             include_overflow=False, copy=True, return_edges=True))

src_vec_gam = np.asarray(root_numpy.hist2array(f_rspns.Get('Source Spectrum (Gamma)'), 
                                               include_overflow=False, copy=True, return_edges=True))

mig_mat_e = np.asarray(root_numpy.hist2array(f_rspns.Get('Energy Migration Matrix (Electron)'), 
                                             include_overflow=False, copy=True, return_edges=True))
    
mig_mat_gam = np.asarray(root_numpy.hist2array(f_rspns.Get('Energy Migration Matrix (Gamma)'), 
                                               include_overflow=False, copy=True, return_edges=True))

# Calculate the response matrices by normalizing the energy migration matrices by the source vectors
rspns_mat_det2_e = copy.deepcopy(mig_mat_e)
rspns_mat_det2_e[0] = np.nan_to_num(rspns_mat_det2_e[0]/src_vec_e[0])
rspns_mat_det2_gam = copy.deepcopy(mig_mat_gam)
rspns_mat_det2_gam[0] = np.nan_to_num(rspns_mat_det2_gam[0]/src_vec_e[0])

# Remove response matrix elements below threshold energy
rspns_mat_det2_e[0] = np.delete(rspns_mat_det2_e[0], np.where(rspns_mat_det2_e[1][0] < thld_e), axis=0)
rspns_mat_det2_e[0] = np.delete(rspns_mat_det2_e[0], np.where(rspns_mat_det2_e[1][0] < thld_e), axis=1)
rspns_mat_det2_e[1] = np.delete(rspns_mat_det2_e[1], np.where(rspns_mat_det2_e[1][0] < thld_e), axis=1)
rspns_mat_det2_gam[0] = np.delete(rspns_mat_det2_gam[0], np.where(rspns_mat_det2_gam[1][0] < thld_e), axis=0)
rspns_mat_det2_gam[0] = np.delete(rspns_mat_det2_gam[0], np.where(rspns_mat_det2_gam[1][0] < thld_e), axis=1)
rspns_mat_det2_gam[1] = np.delete(rspns_mat_det2_gam[1], np.where(rspns_mat_det2_gam[1][0] < thld_e), axis=1)

# Create a combined response matrix
rspns_mat_det2_comb = copy.deepcopy(rspns_mat_det2_e)
rspns_mat_det2_comb[0] += rspns_mat_det2_gam[0]


# Load the ROOT file containing the response matrix for the detector
f_rspns = ROOT.TFile.Open('./TestData/'+det3+'/Response Matrix/'+det3+'.root')

# Retrieve the electron and gamma-ray energy migration matrices and source vectors (i.e. the true spectrum from which the response matrix was simulated)
# NOTE: Index 0 contains the bin values
#       Index 1 contains the bin edges
src_vec_e = np.asarray(root_numpy.hist2array(f_rspns.Get('Source Spectrum (Electron)'), 
                                             include_overflow=False, copy=True, return_edges=True))

src_vec_gam = np.asarray(root_numpy.hist2array(f_rspns.Get('Source Spectrum (Gamma)'), 
                                               include_overflow=False, copy=True, return_edges=True))

mig_mat_e = np.asarray(root_numpy.hist2array(f_rspns.Get('Energy Migration Matrix (Electron)'), 
                                             include_overflow=False, copy=True, return_edges=True))
    
mig_mat_gam = np.asarray(root_numpy.hist2array(f_rspns.Get('Energy Migration Matrix (Gamma)'), 
                                               include_overflow=False, copy=True, return_edges=True))

# Calculate the response matrices by normalizing the energy migration matrices by the source vectors
rspns_mat_det3_e = copy.deepcopy(mig_mat_e)
rspns_mat_det3_e[0] = np.nan_to_num(rspns_mat_det3_e[0]/src_vec_e[0])
rspns_mat_det3_gam = copy.deepcopy(mig_mat_gam)
rspns_mat_det3_gam[0] = np.nan_to_num(rspns_mat_det3_gam[0]/src_vec_e[0])

# Remove response matrix elements below threshold energy
rspns_mat_det3_e[0] = np.delete(rspns_mat_det3_e[0], np.where(rspns_mat_det3_e[1][0] < thld_e), axis=0)
rspns_mat_det3_e[0] = np.delete(rspns_mat_det3_e[0], np.where(rspns_mat_det3_e[1][0] < thld_e), axis=1)
rspns_mat_det3_e[1] = np.delete(rspns_mat_det3_e[1], np.where(rspns_mat_det3_e[1][0] < thld_e), axis=1)
rspns_mat_det3_gam[0] = np.delete(rspns_mat_det3_gam[0], np.where(rspns_mat_det3_gam[1][0] < thld_e), axis=0)
rspns_mat_det3_gam[0] = np.delete(rspns_mat_det3_gam[0], np.where(rspns_mat_det3_gam[1][0] < thld_e), axis=1)
rspns_mat_det3_gam[1] = np.delete(rspns_mat_det3_gam[1], np.where(rspns_mat_det3_gam[1][0] < thld_e), axis=1)

# Create a combined response matrix
rspns_mat_det3_comb = copy.deepcopy(rspns_mat_det3_e)
rspns_mat_det3_comb[0] += rspns_mat_det3_gam[0]


def plotResponseMatrix(rspns_mat_e, rspns_mat_gam, rspns_mat_comb, filename = 'Response Matrix.jpg'):
    # Plot the energy migration matrix
    fig_mig_mat = plt.figure()

    ax_mig_mat = AxesGrid(fig_mig_mat, 111,
                          nrows_ncols=(1, 3),
                          axes_pad=0.3,
                          aspect=False,
                          #label_mode = 'L',
                          cbar_mode='single',
                          cbar_location='right',
                          cbar_pad=0.2,
                          cbar_size = 0.3)

    # Color map
    cmap = palettable.matplotlib.Viridis_20.mpl_colormap
    cmap.set_bad(cmap(0.))
    cmap.set_over(cmap(1.))

    # Response Limits 
    rLimUp = np.ceil(np.abs(np.log10(np.maximum(rspns_mat_e[0].max(), rspns_mat_gam[0].max()))))
    rLimUp = 1E1
    rLimLow = rLimUp/1E3

    # Plot the response matrices
    X, Y = np.meshgrid(rspns_mat_e[1][0], rspns_mat_e[1][1])
    H0 = ax_mig_mat[0].pcolormesh(X, Y, rspns_mat_e[0].T, norm = colors.LogNorm(), cmap = cmap, rasterized = True) 

    X, Y = np.meshgrid(rspns_mat_gam[1][0], rspns_mat_gam[1][1])
    H1 = ax_mig_mat[1].pcolormesh(X, Y, rspns_mat_gam[0].T, norm = colors.LogNorm(), cmap = cmap, rasterized = True) 

    X, Y = np.meshgrid(rspns_mat_comb[1][0], rspns_mat_comb[1][1])
    H2 = ax_mig_mat[2].pcolormesh(X, Y, rspns_mat_comb[0].T, norm = colors.LogNorm(), cmap = cmap, rasterized = True) 

    # Color limits for the plot
    H0.set_clim(rLimLow, rLimUp)
    H1.set_clim(rLimLow, rLimUp)
    H2.set_clim(rLimLow, rLimUp)

    # Colorbar     
    from matplotlib.ticker import LogLocator
    ax_mig_mat.cbar_axes[0].colorbar(H2, spacing = 'uniform')
    ax_mig_mat.cbar_axes[0].set_yscale('log')
    ax_mig_mat.cbar_axes[0].axis[ax_mig_mat.cbar_axes[0].orientation].set_label('Omnidirectional Response (cm$^2$)')

    # Figure Properties
    ax_mig_mat[0].set_xscale('log')
    ax_mig_mat[0].set_yscale('log')
    ax_mig_mat[0].set_ylabel('Measured Energy (keV)')
    ax_mig_mat[0].set_xlabel('True Energy (keV)')
    ax_mig_mat[0].set_title('Beta-ray Response Matrix')

    ax_mig_mat[1].set_xscale('log')
    ax_mig_mat[1].set_yscale('log')
    ax_mig_mat[1].set_xlabel('True Energy (keV)')
    ax_mig_mat[1].set_title('Gamma-ray Response Matrix')

    ax_mig_mat[2].set_xscale('log')
    ax_mig_mat[2].set_yscale('log')
    ax_mig_mat[2].set_xlabel('True Energy (keV)')
    ax_mig_mat[2].set_title('Combined Response Matrix')

    # Fine-tune figure 
    fig_mig_mat.set_tight_layout(False)
    
    # Save the figure
    plt.savefig(filename, bbox_inches="tight")

    # Show the figure
    #plt.show(fig_mig_mat)
    plt.close(fig_mig_mat)
    
print('Response Matrix - Detector 1 - ' + det1)
plotResponseMatrix(rspns_mat_det1_e, rspns_mat_det1_gam, rspns_mat_det1_comb, det1 + ' Response Matrix.jpg')
print('Response Matrix - Detector 2 - ' + det2)
plotResponseMatrix(rspns_mat_det2_e, rspns_mat_det2_gam, rspns_mat_det2_comb, det2 + ' Response Matrix.jpg')
print('Response Matrix - Detector 3 - ' + det3)
plotResponseMatrix(rspns_mat_det3_e, rspns_mat_det3_gam, rspns_mat_det3_comb, det3 + ' Response Matrix.jpg')


# ## STEP 2 - Import the measured spectra from each detector

# Load the ROOT file containing the measured spectrum
f_meas = ROOT.TFile.Open('./TestData/'+det1+'/'+isotope+'/'+f_data)

# Retrieve the measured spectrum
# NOTE: Index 0 contains the bin values
#       Index 1 contains the bin edges
meas_vec_det1 = np.asarray(root_numpy.hist2array(f_meas.Get('Detector Measured Spectrum'), 
                                            include_overflow=False, copy=True, return_edges=True))

truth_vec_det1_e = np.asarray(root_numpy.hist2array(f_meas.Get('Source Spectrum (Electron)'),
                                               include_overflow=False, copy=True, return_edges=True))

truth_vec_det1_gam = np.asarray(root_numpy.hist2array(f_meas.Get('Source Spectrum (Gamma)'),
                                                 include_overflow=False, copy=True, return_edges=True))

# Remove elements below threshold energy
meas_vec_det1[0] = np.delete(meas_vec_det1[0], np.where(meas_vec_det1[1][0] < thld_e), axis=0)
meas_vec_det1[1] = np.delete(meas_vec_det1[1], np.where(meas_vec_det1[1][0] < thld_e), axis=1)
truth_vec_det1_e[0] = np.delete(truth_vec_det1_e[0], np.where(truth_vec_det1_e[1][0] < thld_e), axis=0)
truth_vec_det1_e[1] = np.delete(truth_vec_det1_e[1], np.where(truth_vec_det1_e[1][0] < thld_e), axis=1)
truth_vec_det1_gam[0] = np.delete(truth_vec_det1_gam[0], np.where(truth_vec_det1_gam[1][0] < thld_e), axis=0)
truth_vec_det1_gam[1] = np.delete(truth_vec_det1_gam[1], np.where(truth_vec_det1_gam[1][0] < thld_e), axis=1)

# Load the ROOT file containing the measured spectrum
f_meas = ROOT.TFile.Open('./TestData/'+det2+'/'+isotope+'/'+f_data)

# Retrieve the measured spectrum
# NOTE: Index 0 contains the bin values
#       Index 1 contains the bin edges
meas_vec_det2 = np.asarray(root_numpy.hist2array(f_meas.Get('Detector Measured Spectrum'), 
                                            include_overflow=False, copy=True, return_edges=True))

truth_vec_det2_e = np.asarray(root_numpy.hist2array(f_meas.Get('Source Spectrum (Electron)'),
                                               include_overflow=False, copy=True, return_edges=True))

truth_vec_det2_gam = np.asarray(root_numpy.hist2array(f_meas.Get('Source Spectrum (Gamma)'),
                                                 include_overflow=False, copy=True, return_edges=True))

# Remove elements below threshold energy
meas_vec_det2[0] = np.delete(meas_vec_det2[0], np.where(meas_vec_det2[1][0] < thld_e), axis=0)
meas_vec_det2[1] = np.delete(meas_vec_det2[1], np.where(meas_vec_det2[1][0] < thld_e), axis=1)
truth_vec_det2_e[0] = np.delete(truth_vec_det2_e[0], np.where(truth_vec_det2_e[1][0] < thld_e), axis=0)
truth_vec_det2_e[1] = np.delete(truth_vec_det2_e[1], np.where(truth_vec_det2_e[1][0] < thld_e), axis=1)
truth_vec_det2_gam[0] = np.delete(truth_vec_det2_gam[0], np.where(truth_vec_det2_gam[1][0] < thld_e), axis=0)
truth_vec_det2_gam[1] = np.delete(truth_vec_det2_gam[1], np.where(truth_vec_det2_gam[1][0] < thld_e), axis=1)


# Load the ROOT file containing the measured spectrum
f_meas = ROOT.TFile.Open('./TestData/'+det3+'/'+isotope+'/'+f_data)

# Retrieve the measured spectrum
# NOTE: Index 0 contains the bin values
#       Index 1 contains the bin edges
meas_vec_det3 = np.asarray(root_numpy.hist2array(f_meas.Get('Detector Measured Spectrum'), 
                                            include_overflow=False, copy=True, return_edges=True))

truth_vec_det3_e = np.asarray(root_numpy.hist2array(f_meas.Get('Source Spectrum (Electron)'),
                                               include_overflow=False, copy=True, return_edges=True))

truth_vec_det3_gam = np.asarray(root_numpy.hist2array(f_meas.Get('Source Spectrum (Gamma)'),
                                                 include_overflow=False, copy=True, return_edges=True))

# Remove elements below threshold energy
meas_vec_det3[0] = np.delete(meas_vec_det3[0], np.where(meas_vec_det3[1][0] < thld_e), axis=0)
meas_vec_det3[1] = np.delete(meas_vec_det3[1], np.where(meas_vec_det3[1][0] < thld_e), axis=1)
truth_vec_det3_e[0] = np.delete(truth_vec_det3_e[0], np.where(truth_vec_det3_e[1][0] < thld_e), axis=0)
truth_vec_det3_e[1] = np.delete(truth_vec_det3_e[1], np.where(truth_vec_det3_e[1][0] < thld_e), axis=1)
truth_vec_det3_gam[0] = np.delete(truth_vec_det3_gam[0], np.where(truth_vec_det3_gam[1][0] < thld_e), axis=0)
truth_vec_det3_gam[1] = np.delete(truth_vec_det3_gam[1], np.where(truth_vec_det3_gam[1][0] < thld_e), axis=1)

# Plot the measured spectrum
def plotMeasuredSpectrum(meas_vec, filename = 'Measured Spectrum.jpg'):
    # Plot the measured spectrum
    fig_meas_vec, ax_meas_vec = plt.subplots()

    # Plot the raw spectrum
    ax_meas_vec.plot(sorted(np.append(meas_vec[1][0][:-1], meas_vec[1][0][1:])),
                    np.repeat(meas_vec[0], 2),
                    lw=1.25,
                    color='black',
                    linestyle="-",
                    drawstyle='steps')

    # Figure properties
    ax_meas_vec.set_xlabel('Measured Energy (keV)')
    ax_meas_vec.set_ylabel('Counts')
    ax_meas_vec.set_xlim(min(meas_vec[1][0]),max(meas_vec[1][0]))
    ax_meas_vec.set_xscale('log')
    ax_meas_vec.set_yscale('log')

    # Fine-tune figure 
    fig_meas_vec.set_tight_layout(True)
    
    # Save the figure
    plt.savefig(filename, bbox_inches="tight")
    
    # Show the figure
    #plt.show(fig_meas_vec)
    plt.close(fig_meas_vec)
    
print('Measured Spectrum - Detector 1 - ' + det1)
plotMeasuredSpectrum(meas_vec_det1, isotope + ' - ' + det1 + ' - Measured Spectrum.jpg')
print('Measured Spectrum - Detector 2 - ' + det2)
plotMeasuredSpectrum(meas_vec_det2, isotope + ' - ' + det2 + ' - Measured Spectrum.jpg')
print('Measured Spectrum - Detector 3 - ' + det3)
plotMeasuredSpectrum(meas_vec_det3, isotope + ' - ' + det3 + ' - Measured Spectrum.jpg')


# ## STEP 3  - Build the generative models
# Generally, when a detector is exposed to a homogeneous radiation field, the relationship between the incoming particle fluence spectrum and the measured energy spectrum, $D(E)$, can be described by the following Fredholm integral equation of the first kind:
# 
# $$D\left(E\right) = \int_{0}^{\infty}R\left(E, E^{'}\right)\Phi\left(E^{'}\right)dE^{'} \ ,\ 0 \leq E \leq \infty$$
# 
# where $R\left(E, E^{'}\right)$ is a kernel describing the detector response in terms of the measured energy, $E$, and the true energy, $E^{'}$, of the incoming particle and $\Phi\left(E^{'}\right)$ is the incoming particle fluence spectrum.
# 
# Within the context of Bayesian inference, the above equation is often refered to as the generative model that describes how the measured data was generated when the detector was exposed to the radiation field. 
# 
# For this systematic testing, the following generative models are available:
# - **model_det1** - this model uses **only** the response matrix and measured spectra from Detector 1 (det1)

DRAWS = 10000
TUNE = 20000

def plotReconstructedSpectrum(trace, filename = 'Unfolded Fluence Spectrum.jpg'):
    # Create a Pandas dataframe of summary information from the sampling
    df_reco = pm.summary(trace, alpha=0.005)

    # Create a figure and axis to plot the unfolded (aka. reconstructed) beta-ray and gamma-ray fluence spectra
    fig_reco_vec = plt.figure()

    ax_reco_vec = Grid(fig_reco_vec, 
                        111,
                        nrows_ncols=(2, 1),
                        axes_pad=(0.35, 0.35),
                        add_all=True,
                        label_mode = 'L')

    # Plot the unfolded spectrum
    pMeanBeta, = ax_reco_vec[0].plot(sorted(np.append(rspns_mat_det1_e[1][0][:-1], rspns_mat_det1_e[1][0][1:])),
                                     np.repeat(df_reco[df_reco.index.str.startswith('phi_e')]['mean'], 2),
                                     lw=1.5,
                                     color='black',
                                     linestyle="-",
                                     drawstyle='steps')

    pBCIBeta = ax_reco_vec[0].fill_between(sorted(np.append(rspns_mat_det1_e[1][0][:-1], rspns_mat_det1_e[1][0][1:])), 
                                           np.repeat(df_reco[df_reco.index.str.startswith('phi_e')]['hpd_0.25'], 2), 
                                           np.repeat(df_reco[df_reco.index.str.startswith('phi_e')]['hpd_99.75'], 2),
                                           color='black',
                                           alpha=0.2)

    pMeanGamma, = ax_reco_vec[1].plot(sorted(np.append(rspns_mat_det1_gam[1][0][:-1], rspns_mat_det1_gam[1][0][1:])),
                                      np.repeat(df_reco[df_reco.index.str.startswith('phi_gam')]['mean'], 2),
                                      lw=1.5,
                                      color='black',
                                      linestyle="-",
                                      drawstyle='steps')

    pBCIGamma = ax_reco_vec[1].fill_between(sorted(np.append(rspns_mat_det1_gam[1][0][:-1], rspns_mat_det1_gam[1][0][1:])), 
                                            np.repeat(df_reco[df_reco.index.str.startswith('phi_gam')]['hpd_0.25'], 2), 
                                            np.repeat(df_reco[df_reco.index.str.startswith('phi_gam')]['hpd_99.75'], 2),
                                            color='black',
                                            alpha=0.2)

    # Plot the truth spectrum (if known)
    pTruthBeta, = ax_reco_vec[0].plot(sorted(np.append(truth_vec_det1_e[1][0][:-1], truth_vec_det1_e[1][0][1:])),
                                      np.repeat(truth_vec_det1_e[0], 2),
                                      lw=1.5,
                                      color='blue',
                                      linestyle="-",
                                      drawstyle='steps')

    pTruthGamma, = ax_reco_vec[1].plot(sorted(np.append(truth_vec_det1_gam[1][0][:-1], truth_vec_det1_gam[1][0][1:])),
                                       np.repeat(truth_vec_det1_gam[0], 2),
                                       lw=1.5,
                                       color='blue',
                                       linestyle="-",
                                       drawstyle='steps')

    # Find min and max y value for scaling the plot
    y_lim_up = np.max([truth_vec_det1_e[0].max(),
                       truth_vec_det1_gam[0].max(),
                       df_reco[df_reco.index.str.startswith('phi_e')]['hpd_99.75'].max(),
                       df_reco[df_reco.index.str.startswith('phi_gam')]['hpd_99.75'].max()])
    y_lim_up = 10**np.ceil(np.abs(np.log10(y_lim_up)))
    y_lim_up = 1E6
    y_lim_down = y_lim_up/1E6

    # Plot statistics text
    print('\nStatistics from reconstructed Beta-ray Fluence Spectrum            \n-------------------------------------------------------            \nRMSE \t{:.2E} ({:.2E} - {:.2E})            \nMAE \t{:.2E} ({:.2E} - {:.2E})'
          .format(np.sqrt(((df_reco[df_reco.index.str.startswith('phi_e')]['mean'] - truth_vec_det1_e[0])**2).sum()/truth_vec_det1_e[0].size),
                  np.sqrt(((df_reco[df_reco.index.str.startswith('phi_e')]['hpd_0.25'] - truth_vec_det1_e[0])**2).sum()/truth_vec_det1_e[0].size),
                  np.sqrt(((df_reco[df_reco.index.str.startswith('phi_e')]['hpd_99.75'] - truth_vec_det1_e[0])**2).sum()/truth_vec_det1_e[0].size),
                  np.abs(truth_vec_det1_e[0] - df_reco[df_reco.index.str.startswith('phi_e')]['mean']).sum()/truth_vec_det1_e[0].size,
                  np.abs(truth_vec_det1_e[0] - df_reco[df_reco.index.str.startswith('phi_e')]['hpd_0.25']).sum()/truth_vec_det1_e[0].size,
                  np.abs(truth_vec_det1_e[0] - df_reco[df_reco.index.str.startswith('phi_e')]['hpd_99.75']).sum()/truth_vec_det1_e[0].size))
    
    print('\nStatistics from reconstructed Gamma-ray Fluence Spectrum            \n-------------------------------------------------------            \nRMSE \t{:.2E} ({:.2E} - {:.2E})            \nMAE \t{:.2E} ({:.2E} - {:.2E})'
          .format(np.sqrt(((df_reco[df_reco.index.str.startswith('phi_gam')]['mean'] - truth_vec_det1_gam[0])**2).sum()/truth_vec_det1_gam[0].size),
                  np.sqrt(((df_reco[df_reco.index.str.startswith('phi_gam')]['hpd_0.25'] - truth_vec_det1_gam[0])**2).sum()/truth_vec_det1_gam[0].size),
                  np.sqrt(((df_reco[df_reco.index.str.startswith('phi_gam')]['hpd_99.75'] - truth_vec_det1_gam[0])**2).sum()/truth_vec_det1_gam[0].size),
                  np.abs(truth_vec_det1_gam[0] - df_reco[df_reco.index.str.startswith('phi_gam')]['mean']).sum()/truth_vec_det1_gam[0].size, 
                  np.abs(truth_vec_det1_gam[0] - df_reco[df_reco.index.str.startswith('phi_gam')]['hpd_0.25']).sum()/truth_vec_det1_gam[0].size, 
                  np.abs(truth_vec_det1_gam[0] - df_reco[df_reco.index.str.startswith('phi_gam')]['hpd_99.75']).sum()/truth_vec_det1_gam[0].size))

    # Figure properties
    ax_reco_vec[0].set_xlabel('True Energy (keV)')
    ax_reco_vec[0].set_ylabel('Fluence (cm$^{-2}$)')
    ax_reco_vec[0].set_xlim(min(rspns_mat_det1_e[1][0]), max(rspns_mat_det1_e[1][0]))
    ax_reco_vec[0].set_ylim(y_lim_down, y_lim_up)
    ax_reco_vec[0].set_xscale('log')
    ax_reco_vec[0].set_yscale('log')
    ax_reco_vec[0].set_title('Beta-ray Fluence Spectrum')
    ax_reco_vec[0].legend([pTruthBeta, (pBCIBeta, pMeanBeta)], ['True distribution','Unfolded dist. (99.5% BCI)'], loc='upper right')

    ax_reco_vec[1].set_xlabel('True Energy (keV)')
    ax_reco_vec[1].set_ylabel('Fluence (cm$^{-2}$)')
    ax_reco_vec[1].set_xlim(min(rspns_mat_det1_gam[1][0]),max(rspns_mat_det1_gam[1][0]))
    ax_reco_vec[1].set_ylim(y_lim_down, y_lim_up)
    ax_reco_vec[1].set_xscale('log')
    ax_reco_vec[1].set_yscale('log')
    ax_reco_vec[1].set_title('Gamma-ray Fluence Spectrum')
    ax_reco_vec[1].legend([pTruthGamma, (pBCIGamma, pMeanGamma)], ['True distribution','Unfolded dist. (99.5% BCI)'], loc='upper right')

    # Fine-tune figure 
    fig_reco_vec.set_tight_layout(True)
    
    # Save the figure
    plt.savefig(filename, bbox_inches="tight")
    
    # Show the figure
    #plt.show(fig_reco_vec)
    plt.close(fig_reco_vec)


# - **model_det1** - this model uses **only** the response matrix and measured spectra from Detector 1 (det1)
print('\nUnfolding using model_det1\n--------------------------------------')
with pm.Model() as model_det1:
    ''' 
    Define the upper and lower bounds of the uniform prior based on the measured data and the response matrix
    
    For an ideal radiation detector, the response matrix would a diagonal meaning that the measured spectrum would be an exact if not close approximation of the true particle spectrum incident on the detector. However, real detectors have response matrices which often have non-diagonal components due to physical interactions (e.g. compton scaterring) which result in only partial energy depositions. As a result, the measured spectrum is not an accurate representation of the true particle spectrum incident on the detector. Nevertheless, we can use the measured spectrum in combination with the response matrix to provide a good initial or "guess" spectrum for the Bayesian inference.
    '''

    # Define the upper and lower bounds for the priors
    gf_det1_e = np.sum(rspns_mat_det1_e[0], axis=1)
    gf_det1_gam = np.sum(rspns_mat_det1_gam[0], axis=1)
    
    gf_det1_e[np.isclose(gf_det1_e, 0)] = np.min(gf_det1_e[np.nonzero(gf_det1_e)])
    gf_det1_gam[np.isclose(gf_det1_gam, 0)] = np.min(gf_det1_gam[np.nonzero(gf_det1_gam)])

    lb_phi_e = np.zeros(rspns_mat_det1_e[1][0].size - 1)
    ub_phi_e = np.ones(rspns_mat_det1_e[1][0].size - 1)*np.sum(meas_vec_det1[0])/gf_det1_e
    lb_phi_gam = np.zeros(rspns_mat_det1_gam[1][0].size - 1)
    ub_phi_gam = np.ones(rspns_mat_det1_gam[1][0].size - 1)*np.sum(meas_vec_det1[0])/gf_det1_gam
    
    # Define the prior probability densities
    phi_e = pm.Uniform('phi_e', lower = lb_phi_e, upper = ub_phi_e, shape = (ub_phi_e.size))
    phi_gam = pm.Uniform('phi_gam', lower = lb_phi_gam, upper = ub_phi_gam, shape = (ub_phi_gam.size))
    
    # Define the generative models
    M_det1 = theano.dot(rspns_mat_det1_e[0].T, phi_e) + theano.dot(rspns_mat_det1_gam[0].T, phi_gam)

    # Define the likelihood (aka. posterior probability function)
    PPF_det1 = pm.Poisson('PPF_det1', mu = M_det1, observed = meas_vec_det1[0], shape = (meas_vec_det1[0].size))

with model_det1:
    print 'Sampling the posterior distribution ...'

    # Sample
    trace = pm.sample(draws = DRAWS,
                      tune = TUNE,
                      step = pm.HamiltonianMC(target_accept=0.99),
                      start = pm.find_MAP(),
                      use_mmap = True,
                      compute_convergence_checks = True)
    
    plotReconstructedSpectrum(trace, isotope + ' - ' + det1 + ' - Unfolded Fluence Spectrum.jpg')

# - **model_det2** - this model uses **only** the response matrix and measured spectra from Detector 2 (det2)
print('\nUnfolding using model_det2\n--------------------------------------')
with pm.Model() as model_det2:
    ''' 
    Define the upper and lower bounds of the uniform prior based on the measured data and the response matrix
    
    For an ideal radiation detector, the response matrix would a diagonal meaning that the measured spectrum would be an exact if not close approximation of the true particle spectrum incident on the detector. However, real detectors have response matrices which often have non-diagonal components due to physical interactions (e.g. compton scaterring) which result in only partial energy depositions. As a result, the measured spectrum is not an accurate representation of the true particle spectrum incident on the detector. Nevertheless, we can use the measured spectrum in combination with the response matrix to provide a good initial or "guess" spectrum for the Bayesian inference.
    '''

    # Define the upper and lower bounds for the priors
    gf_det2_e = np.sum(rspns_mat_det2_e[0], axis=1)
    gf_det2_gam = np.sum(rspns_mat_det2_gam[0], axis=1)
    
    gf_det2_e[np.isclose(gf_det2_e, 0)] = np.min(gf_det2_e[np.nonzero(gf_det2_e)])
    gf_det2_gam[np.isclose(gf_det2_gam, 0)] = np.min(gf_det2_gam[np.nonzero(gf_det2_gam)])

    lb_phi_e = np.zeros(rspns_mat_det2_e[1][0].size - 1)
    ub_phi_e = np.ones(rspns_mat_det2_e[1][0].size - 1)*np.sum(meas_vec_det2[0])/gf_det2_e
    lb_phi_gam = np.zeros(rspns_mat_det2_gam[1][0].size - 1)
    ub_phi_gam = np.ones(rspns_mat_det2_gam[1][0].size - 1)*np.sum(meas_vec_det2[0])/gf_det2_gam
    
    # Define the prior probability densities
    phi_e = pm.Uniform('phi_e', lower = lb_phi_e, upper = ub_phi_e, shape = (ub_phi_e.size))
    phi_gam = pm.Uniform('phi_gam', lower = lb_phi_gam, upper = ub_phi_gam, shape = (ub_phi_gam.size))
    
    # Define the generative models
    M_det2 = theano.dot(rspns_mat_det2_e[0].T, phi_e) + theano.dot(rspns_mat_det2_gam[0].T, phi_gam)

    # Define the likelihood (aka. posterior probability function)
    PPF_det2 = pm.Poisson('PPF_det2', mu = M_det2, observed = meas_vec_det2[0], shape = (meas_vec_det2[0].size))

with model_det2:
    print 'Sampling the posterior distribution ...'

    # Sample
    trace = pm.sample(draws = DRAWS,
                      tune = TUNE,
                      step = pm.HamiltonianMC(target_accept=0.99),
                      start = pm.find_MAP(),
                      use_mmap = True,
                      compute_convergence_checks = True)
    
    plotReconstructedSpectrum(trace, isotope + ' - ' + det2 + ' - Unfolded Fluence Spectrum.jpg')

# - **model_det3** - this model uses **only** the response matrix and measured spectra from Detector 3 (det3)
print('\nUnfolding using model_det3\n--------------------------------------')
with pm.Model() as model_det3:
    ''' 
    Define the upper and lower bounds of the uniform prior based on the measured data and the response matrix
    
    For an ideal radiation detector, the response matrix would a diagonal meaning that the measured spectrum would be an exact if not close approximation of the true particle spectrum incident on the detector. However, real detectors have response matrices which often have non-diagonal components due to physical interactions (e.g. compton scaterring) which result in only partial energy depositions. As a result, the measured spectrum is not an accurate representation of the true particle spectrum incident on the detector. Nevertheless, we can use the measured spectrum in combination with the response matrix to provide a good initial or "guess" spectrum for the Bayesian inference.
    '''

    # Define the upper and lower bounds for the priors
    gf_det3_e = np.sum(rspns_mat_det3_e[0], axis=1)
    gf_det3_gam = np.sum(rspns_mat_det3_gam[0], axis=1)
    
    gf_det3_e[np.isclose(gf_det3_e, 0)] = np.min(gf_det3_e[np.nonzero(gf_det3_e)])
    gf_det3_gam[np.isclose(gf_det3_gam, 0)] = np.min(gf_det3_gam[np.nonzero(gf_det3_gam)])

    lb_phi_e = np.zeros(rspns_mat_det3_e[1][0].size - 1)
    ub_phi_e = np.ones(rspns_mat_det3_e[1][0].size - 1)*np.sum(meas_vec_det3[0])/gf_det3_e
    lb_phi_gam = np.zeros(rspns_mat_det3_gam[1][0].size - 1)
    ub_phi_gam = np.ones(rspns_mat_det3_gam[1][0].size - 1)*np.sum(meas_vec_det3[0])/gf_det3_gam
    
    # Define the prior probability densities
    phi_e = pm.Uniform('phi_e', lower = lb_phi_e, upper = ub_phi_e, shape = (ub_phi_e.size))
    phi_gam = pm.Uniform('phi_gam', lower = lb_phi_gam, upper = ub_phi_gam, shape = (ub_phi_gam.size))
    
    # Define the generative models
    M_det3 = theano.dot(rspns_mat_det3_e[0].T, phi_e) + theano.dot(rspns_mat_det3_gam[0].T, phi_gam)

    # Define the likelihood (aka. posterior probability function)
    PPF_det3 = pm.Poisson('PPF_det3', mu = M_det3, observed = meas_vec_det3[0], shape = (meas_vec_det3[0].size))

with model_det3:
    print 'Sampling the posterior distribution ...'

    # Sample
    trace = pm.sample(draws = DRAWS,
                      tune = TUNE,
                      step = pm.HamiltonianMC(target_accept=0.99),
                      start = pm.find_MAP(),
                      use_mmap = True,
                      compute_convergence_checks = True)
    
    plotReconstructedSpectrum(trace, isotope + ' - ' + det3 + ' - Unfolded Fluence Spectrum.jpg')

# - **model_det1_det2** - this model uses the response matrix and measured spectra from Detectors 1 and 2
print('\nUnfolding using model_det1_det2\n--------------------------------------')
with pm.Model() as model_det1_det2:
    ''' 
    Define the upper and lower bounds of the uniform prior based on the measured data and the response matrix
    
    For an ideal radiation detector, the response matrix would a diagonal meaning that the measured spectrum would be an exact if not close approximation of the true particle spectrum incident on the detector. However, real detectors have response matrices which often have non-diagonal components due to physical interactions (e.g. compton scaterring) which result in only partial energy depositions. As a result, the measured spectrum is not an accurate representation of the true particle spectrum incident on the detector. Nevertheless, we can use the measured spectrum in combination with the response matrix to provide a good initial or "guess" spectrum for the Bayesian inference.
    '''

    # Define the upper and lower bounds for the priors
    gf_det1_e = np.sum(rspns_mat_det1_e[0], axis=1)
    gf_det1_gam = np.sum(rspns_mat_det1_gam[0], axis=1)
    gf_det2_e = np.sum(rspns_mat_det2_e[0], axis=1)
    gf_det2_gam = np.sum(rspns_mat_det2_gam[0], axis=1)
    
    gf_det1_e[np.isclose(gf_det1_e, 0)] = np.min(gf_det1_e[np.nonzero(gf_det1_e)])
    gf_det1_gam[np.isclose(gf_det1_gam, 0)] = np.min(gf_det1_gam[np.nonzero(gf_det1_gam)])
    gf_det2_e[np.isclose(gf_det2_e, 0)] = np.min(gf_det2_e[np.nonzero(gf_det2_e)])
    gf_det2_gam[np.isclose(gf_det2_gam, 0)] = np.min(gf_det2_gam[np.nonzero(gf_det2_gam)])

    # Find the ROIs where each detector has the highest Geometric Factor
    roi_det1_e = np.intersect1d(np.where(gf_det1_e >= gf_det2_e), np.where(gf_det1_e >= gf_det1_e))
    roi_det1_gam = np.intersect1d(np.where(gf_det1_gam >= gf_det2_gam), np.where(gf_det1_gam >= gf_det1_gam))
    roi_det2_e = np.intersect1d(np.where(gf_det2_e >= gf_det1_e), np.where(gf_det2_e >= gf_det1_e))
    roi_det2_gam = np.intersect1d(np.where(gf_det2_gam >= gf_det1_gam), np.where(gf_det2_gam >= gf_det1_gam))

    if roi_det1_e.size != 0: print 'Detector 1 ROI bins for beta-rays: ',roi_det1_e.min(),' to ',roi_det1_e.max() 
    if roi_det2_e.size != 0: print 'Detector 2 ROI bins for beta-rays: ',roi_det2_e.min(),' to ',roi_det2_e.max() 

    if roi_det1_gam.size != 0: print 'Detector 1 ROI bins for gamma-rays: ',roi_det1_gam.min(),' to ',roi_det1_gam.max() 
    if roi_det2_gam.size != 0: print 'Detector 2 ROI bins for gamma-rays: ',roi_det2_gam.min(),' to ',roi_det2_gam.max()

    lb_phi_e = np.zeros(rspns_mat_det3_e[1][0].size - 1)

    ub_phi_e = np.ones(rspns_mat_det3_e[1][0].size - 1)
    if roi_det1_e.size != 0: ub_phi_e[roi_det1_e.min():roi_det1_e.max() + 1] *= np.sum(meas_vec_det1[0])/gf_det1_e[roi_det1_e.min():roi_det1_e.max() + 1]
    if roi_det2_e.size != 0: ub_phi_e[roi_det2_e.min():roi_det2_e.max() + 1] *= np.sum(meas_vec_det2[0])/gf_det2_e[roi_det2_e.min():roi_det2_e.max() + 1]

    lb_phi_gam = np.zeros(rspns_mat_det3_gam[1][0].size - 1)
    
    ub_phi_gam = np.ones(rspns_mat_det3_gam[1][0].size - 1)
    if roi_det1_gam.size != 0: ub_phi_gam[roi_det1_gam.min():roi_det1_gam.max() + 1] *= np.sum(meas_vec_det1[0])/gf_det1_gam[roi_det1_gam.min():roi_det1_gam.max() + 1]
    if roi_det2_gam.size != 0: ub_phi_gam[roi_det2_gam.min():roi_det2_gam.max() + 1] *= np.sum(meas_vec_det2[0])/gf_det2_gam[roi_det2_gam.min():roi_det2_gam.max() + 1]

    # Define the prior probability densities
    phi_e = pm.Uniform('phi_e', lower = lb_phi_e, upper = ub_phi_e, shape = (ub_phi_e.size))
    phi_gam = pm.Uniform('phi_gam', lower = lb_phi_gam, upper = ub_phi_gam, shape = (ub_phi_gam.size))
    
    # Define the generative models
    M_det1 = theano.tensor.dot(rspns_mat_det1_e[0].T, phi_e) + theano.tensor.dot(rspns_mat_det1_gam[0].T, phi_gam)
    M_det2 = theano.tensor.dot(rspns_mat_det2_e[0].T, phi_e) + theano.tensor.dot(rspns_mat_det2_gam[0].T, phi_gam)

    # Define the likelihood (aka. posterior probability function)
    PPF_det1 = pm.Poisson('PPF_det1', mu = M_det1, observed = meas_vec_det1[0], shape = meas_vec_det1[0].size)
    PPF_det2 = pm.Poisson('PPF_det2', mu = M_det2, observed = meas_vec_det2[0], shape = meas_vec_det2[0].size)

with model_det1_det2:
    print 'Sampling the posterior distribution ...'

    # Sample
    trace = pm.sample(draws = DRAWS,
                      tune = TUNE,
                      step = pm.HamiltonianMC(target_accept=0.99),
                      start = pm.find_MAP(),
                      use_mmap = True,
                      compute_convergence_checks = True)
    
    plotReconstructedSpectrum(trace, isotope + ' - ' + det1 + ' - ' + det2 + ' - Unfolded Fluence Spectrum.jpg')

# - **model_det1_det3** - this model uses the response matrix and measured spectra from Detectors 1 and 3
print('\nUnfolding using model_det1_det3\n--------------------------------------')
with pm.Model() as model_det1_det3:
    ''' 
    Define the upper and lower bounds of the uniform prior based on the measured data and the response matrix
    
    For an ideal radiation detector, the response matrix would a diagonal meaning that the measured spectrum would be an exact if not close approximation of the true particle spectrum incident on the detector. However, real detectors have response matrices which often have non-diagonal components due to physical interactions (e.g. compton scaterring) which result in only partial energy depositions. As a result, the measured spectrum is not an accurate representation of the true particle spectrum incident on the detector. Nevertheless, we can use the measured spectrum in combination with the response matrix to provide a good initial or "guess" spectrum for the Bayesian inference.
    '''

    # Define the upper and lower bounds for the priors
    gf_det1_e = np.sum(rspns_mat_det1_e[0], axis=1)
    gf_det1_gam = np.sum(rspns_mat_det1_gam[0], axis=1)
    gf_det3_e = np.sum(rspns_mat_det3_e[0], axis=1)
    gf_det3_gam = np.sum(rspns_mat_det3_gam[0], axis=1)
    
    gf_det1_e[np.isclose(gf_det1_e, 0)] = np.min(gf_det1_e[np.nonzero(gf_det1_e)])
    gf_det1_gam[np.isclose(gf_det1_gam, 0)] = np.min(gf_det1_gam[np.nonzero(gf_det1_gam)])
    gf_det3_e[np.isclose(gf_det3_e, 0)] = np.min(gf_det3_e[np.nonzero(gf_det3_e)])
    gf_det3_gam[np.isclose(gf_det3_gam, 0)] = np.min(gf_det3_gam[np.nonzero(gf_det3_gam)])

    # Find the ROIs where each detector has the highest Geometric Factor
    roi_det1_e = np.intersect1d(np.where(gf_det1_e >= gf_det3_e), np.where(gf_det1_e >= gf_det3_e))
    roi_det1_gam = np.intersect1d(np.where(gf_det1_gam >= gf_det3_gam), np.where(gf_det1_gam >= gf_det3_gam))
    roi_det3_e = np.intersect1d(np.where(gf_det3_e >= gf_det1_e), np.where(gf_det3_e >= gf_det1_e))
    roi_det3_gam = np.intersect1d(np.where(gf_det3_gam >= gf_det1_gam), np.where(gf_det3_gam >= gf_det1_gam))

    if roi_det1_e.size != 0: print 'Detector 1 ROI bins for beta-rays: ',roi_det1_e.min(),' to ',roi_det1_e.max() 
    if roi_det3_e.size != 0: print 'Detector 3 ROI bins for beta-rays: ',roi_det3_e.min(),' to ',roi_det3_e.max() 

    if roi_det1_gam.size != 0: print 'Detector 1 ROI bins for gamma-rays: ',roi_det1_gam.min(),' to ',roi_det1_gam.max() 
    if roi_det3_gam.size != 0: print 'Detector 3 ROI bins for gamma-rays: ',roi_det3_gam.min(),' to ',roi_det3_gam.max()

    lb_phi_e = np.zeros(rspns_mat_det3_e[1][0].size - 1)

    ub_phi_e = np.ones(rspns_mat_det3_e[1][0].size - 1)
    if roi_det1_e.size != 0: ub_phi_e[roi_det1_e.min():roi_det1_e.max() + 1] *= np.sum(meas_vec_det1[0])/gf_det1_e[roi_det1_e.min():roi_det1_e.max() + 1]
    if roi_det3_e.size != 0: ub_phi_e[roi_det3_e.min():roi_det3_e.max() + 1] *= np.sum(meas_vec_det3[0])/gf_det3_e[roi_det3_e.min():roi_det3_e.max() + 1]

    lb_phi_gam = np.zeros(rspns_mat_det3_gam[1][0].size - 1)
    
    ub_phi_gam = np.ones(rspns_mat_det3_gam[1][0].size - 1)
    if roi_det1_gam.size != 0: ub_phi_gam[roi_det1_gam.min():roi_det1_gam.max() + 1] *= np.sum(meas_vec_det1[0])/gf_det1_gam[roi_det1_gam.min():roi_det1_gam.max() + 1]
    if roi_det3_gam.size != 0: ub_phi_gam[roi_det3_gam.min():roi_det3_gam.max() + 1] *= np.sum(meas_vec_det3[0])/gf_det3_gam[roi_det3_gam.min():roi_det3_gam.max() + 1]

    # Define the prior probability densities
    phi_e = pm.Uniform('phi_e', lower = lb_phi_e, upper = ub_phi_e, shape = (ub_phi_e.size))
    phi_gam = pm.Uniform('phi_gam', lower = lb_phi_gam, upper = ub_phi_gam, shape = (ub_phi_gam.size))
    
    # Define the generative models
    M_det1 = theano.tensor.dot(rspns_mat_det1_e[0].T, phi_e) + theano.tensor.dot(rspns_mat_det1_gam[0].T, phi_gam)
    M_det3 = theano.tensor.dot(rspns_mat_det3_e[0].T, phi_e) + theano.tensor.dot(rspns_mat_det3_gam[0].T, phi_gam)

    # Define the likelihood (aka. posterior probability function)
    PPF_det1 = pm.Poisson('PPF_det1', mu = M_det1, observed = meas_vec_det1[0], shape = meas_vec_det1[0].size)
    PPF_det3 = pm.Poisson('PPF_det3', mu = M_det3, observed = meas_vec_det3[0], shape = meas_vec_det3[0].size)

with model_det1_det3:
    print 'Sampling the posterior distribution ...'

    # Sample
    trace = pm.sample(draws = DRAWS,
                      tune = TUNE,
                      step = pm.HamiltonianMC(target_accept=0.99),
                      start = pm.find_MAP(),
                      use_mmap = True,
                      compute_convergence_checks = True)
    
    plotReconstructedSpectrum(trace, isotope + ' - ' + det1 + ' - ' + det3 + ' - Unfolded Fluence Spectrum.jpg')

# - **model_det2_det3** - this model uses the response matrix and measured spectra from Detectors 2 and 3
print('\nUnfolding using model_det2_det3\n--------------------------------------')
with pm.Model() as model_det2_det3:
    ''' 
    Define the upper and lower bounds of the uniform prior based on the measured data and the response matrix
    
    For an ideal radiation detector, the response matrix would a diagonal meaning that the measured spectrum would be an exact if not close approximation of the true particle spectrum incident on the detector. However, real detectors have response matrices which often have non-diagonal components due to physical interactions (e.g. compton scaterring) which result in only partial energy depositions. As a result, the measured spectrum is not an accurate representation of the true particle spectrum incident on the detector. Nevertheless, we can use the measured spectrum in combination with the response matrix to provide a good initial or "guess" spectrum for the Bayesian inference.
    '''

    # Define the upper and lower bounds for the priors
    gf_det2_e = np.sum(rspns_mat_det2_e[0], axis=1)
    gf_det2_gam = np.sum(rspns_mat_det2_gam[0], axis=1)
    gf_det3_e = np.sum(rspns_mat_det3_e[0], axis=1)
    gf_det3_gam = np.sum(rspns_mat_det3_gam[0], axis=1)
    
    gf_det2_e[np.isclose(gf_det2_e, 0)] = np.min(gf_det2_e[np.nonzero(gf_det2_e)])
    gf_det2_gam[np.isclose(gf_det2_gam, 0)] = np.min(gf_det2_gam[np.nonzero(gf_det2_gam)])
    gf_det3_e[np.isclose(gf_det3_e, 0)] = np.min(gf_det3_e[np.nonzero(gf_det3_e)])
    gf_det3_gam[np.isclose(gf_det3_gam, 0)] = np.min(gf_det3_gam[np.nonzero(gf_det3_gam)])

    # Find the ROIs where each detector has the highest Geometric Factor
    roi_det2_e = np.intersect1d(np.where(gf_det2_e >= gf_det3_e), np.where(gf_det2_e >= gf_det3_e))
    roi_det2_gam = np.intersect1d(np.where(gf_det2_gam >= gf_det3_gam), np.where(gf_det2_gam >= gf_det3_gam))
    roi_det3_e = np.intersect1d(np.where(gf_det3_e >= gf_det2_e), np.where(gf_det3_e >= gf_det2_e))
    roi_det3_gam = np.intersect1d(np.where(gf_det3_gam >= gf_det2_gam), np.where(gf_det3_gam >= gf_det2_gam))

    if roi_det2_e.size != 0: print 'Detector 2 ROI bins for beta-rays: ',roi_det2_e.min(),' to ',roi_det2_e.max() 
    if roi_det3_e.size != 0: print 'Detector 3 ROI bins for beta-rays: ',roi_det3_e.min(),' to ',roi_det3_e.max() 

    if roi_det2_gam.size != 0: print 'Detector 2 ROI bins for gamma-rays: ',roi_det2_gam.min(),' to ',roi_det2_gam.max() 
    if roi_det3_gam.size != 0: print 'Detector 3 ROI bins for gamma-rays: ',roi_det3_gam.min(),' to ',roi_det3_gam.max()

    lb_phi_e = np.zeros(rspns_mat_det3_e[1][0].size - 1)

    ub_phi_e = np.ones(rspns_mat_det3_e[1][0].size - 1)
    if roi_det2_e.size != 0: ub_phi_e[roi_det2_e.min():roi_det2_e.max() + 1] *= np.sum(meas_vec_det2[0])/gf_det2_e[roi_det2_e.min():roi_det2_e.max() + 1]
    if roi_det3_e.size != 0: ub_phi_e[roi_det3_e.min():roi_det3_e.max() + 1] *= np.sum(meas_vec_det3[0])/gf_det3_e[roi_det3_e.min():roi_det3_e.max() + 1]

    lb_phi_gam = np.zeros(rspns_mat_det3_gam[1][0].size - 1)
    
    ub_phi_gam = np.ones(rspns_mat_det3_gam[1][0].size - 1)
    if roi_det2_gam.size != 0: ub_phi_gam[roi_det2_gam.min():roi_det2_gam.max() + 1] *= np.sum(meas_vec_det2[0])/gf_det2_gam[roi_det2_gam.min():roi_det2_gam.max() + 1]
    if roi_det3_gam.size != 0: ub_phi_gam[roi_det3_gam.min():roi_det3_gam.max() + 1] *= np.sum(meas_vec_det3[0])/gf_det3_gam[roi_det3_gam.min():roi_det3_gam.max() + 1]

    # Define the prior probability densities
    phi_e = pm.Uniform('phi_e', lower = lb_phi_e, upper = ub_phi_e, shape = (ub_phi_e.size))
    phi_gam = pm.Uniform('phi_gam', lower = lb_phi_gam, upper = ub_phi_gam, shape = (ub_phi_gam.size))
    
    # Define the generative models
    M_det2 = theano.tensor.dot(rspns_mat_det2_e[0].T, phi_e) + theano.tensor.dot(rspns_mat_det2_gam[0].T, phi_gam)
    M_det3 = theano.tensor.dot(rspns_mat_det3_e[0].T, phi_e) + theano.tensor.dot(rspns_mat_det3_gam[0].T, phi_gam)

    # Define the likelihood (aka. posterior probability function)
    PPF_det2 = pm.Poisson('PPF_det2', mu = M_det2, observed = meas_vec_det2[0], shape = meas_vec_det2[0].size)
    PPF_det3 = pm.Poisson('PPF_det3', mu = M_det3, observed = meas_vec_det3[0], shape = meas_vec_det3[0].size)

with model_det2_det3:
    print 'Sampling the posterior distribution ...'

    # Sample
    trace = pm.sample(draws = DRAWS,
                      tune = TUNE,
                      step = pm.HamiltonianMC(target_accept=0.99),
                      start = pm.find_MAP(),
                      use_mmap = True,
                      compute_convergence_checks = True)
    
    plotReconstructedSpectrum(trace, isotope + ' - ' + det2 + ' - ' + det3 + ' - Unfolded Fluence Spectrum.jpg')

# - **model_det1_det2_det3** - this model uses the response matrix and measured spectra from Detectors 1, 2, and 3
print('\nUnfolding using model_det1_det2_det3\n--------------------------------------')
with pm.Model() as model_det1_det2_det3:
    ''' 
    Define the upper and lower bounds of the uniform prior based on the measured data and the response matrix
    
    For an ideal radiation detector, the response matrix would a diagonal meaning that the measured spectrum would be an exact if not close approximation of the true particle spectrum incident on the detector. However, real detectors have response matrices which often have non-diagonal components due to physical interactions (e.g. compton scaterring) which result in only partial energy depositions. As a result, the measured spectrum is not an accurate representation of the true particle spectrum incident on the detector. Nevertheless, we can use the measured spectrum in combination with the response matrix to provide a good initial or "guess" spectrum for the Bayesian inference.
    '''

    # Define the upper and lower bounds for the priors
    gf_det1_e = np.sum(rspns_mat_det1_e[0], axis=1)
    gf_det1_gam = np.sum(rspns_mat_det1_gam[0], axis=1)
    gf_det2_e = np.sum(rspns_mat_det2_e[0], axis=1)
    gf_det2_gam = np.sum(rspns_mat_det2_gam[0], axis=1)
    gf_det3_e = np.sum(rspns_mat_det3_e[0], axis=1)
    gf_det3_gam = np.sum(rspns_mat_det3_gam[0], axis=1)
    
    gf_det1_e[np.isclose(gf_det1_e, 0)] = np.min(gf_det1_e[np.nonzero(gf_det1_e)])
    gf_det1_gam[np.isclose(gf_det1_gam, 0)] = np.min(gf_det1_gam[np.nonzero(gf_det1_gam)])
    gf_det2_e[np.isclose(gf_det2_e, 0)] = np.min(gf_det2_e[np.nonzero(gf_det2_e)])
    gf_det2_gam[np.isclose(gf_det2_gam, 0)] = np.min(gf_det2_gam[np.nonzero(gf_det2_gam)])
    gf_det3_e[np.isclose(gf_det3_e, 0)] = np.min(gf_det3_e[np.nonzero(gf_det3_e)])
    gf_det3_gam[np.isclose(gf_det3_gam, 0)] = np.min(gf_det3_gam[np.nonzero(gf_det3_gam)])

    # Find the ROIs where each detector has the highest Geometric Factor
    roi_det1_e = np.intersect1d(np.where(gf_det1_e >= gf_det2_e), np.where(gf_det1_e >= gf_det3_e))
    roi_det1_gam = np.intersect1d(np.where(gf_det1_gam >= gf_det2_gam), np.where(gf_det1_gam >= gf_det3_gam))
    roi_det2_e = np.intersect1d(np.where(gf_det2_e >= gf_det1_e), np.where(gf_det2_e >= gf_det3_e))
    roi_det2_gam = np.intersect1d(np.where(gf_det2_gam >= gf_det1_gam), np.where(gf_det2_gam >= gf_det3_gam))
    roi_det3_e = np.intersect1d(np.where(gf_det3_e >= gf_det1_e), np.where(gf_det3_e >= gf_det2_e))
    roi_det3_gam = np.intersect1d(np.where(gf_det3_gam >= gf_det1_gam), np.where(gf_det3_gam >= gf_det2_gam))

    if roi_det1_e.size != 0: print 'Detector 1 ROI bins for beta-rays: ',roi_det1_e.min(),' to ',roi_det1_e.max() 
    if roi_det2_e.size != 0: print 'Detector 2 ROI bins for beta-rays: ',roi_det2_e.min(),' to ',roi_det2_e.max() 
    if roi_det3_e.size != 0: print 'Detector 3 ROI bins for beta-rays: ',roi_det3_e.min(),' to ',roi_det3_e.max() 

    if roi_det1_gam.size != 0: print 'Detector 1 ROI bins for gamma-rays: ',roi_det1_gam.min(),' to ',roi_det1_gam.max() 
    if roi_det2_gam.size != 0: print 'Detector 2 ROI bins for gamma-rays: ',roi_det2_gam.min(),' to ',roi_det2_gam.max() 
    if roi_det3_gam.size != 0: print 'Detector 3 ROI bins for gamma-rays: ',roi_det3_gam.min(),' to ',roi_det3_gam.max()

    lb_phi_e = np.zeros(rspns_mat_det3_e[1][0].size - 1)

    ub_phi_e = np.ones(rspns_mat_det3_e[1][0].size - 1)
    if roi_det1_e.size != 0: ub_phi_e[roi_det1_e.min():roi_det1_e.max() + 1] *= np.sum(meas_vec_det1[0])/gf_det1_e[roi_det1_e.min():roi_det1_e.max() + 1]
    if roi_det2_e.size != 0: ub_phi_e[roi_det2_e.min():roi_det2_e.max() + 1] *= np.sum(meas_vec_det2[0])/gf_det2_e[roi_det2_e.min():roi_det2_e.max() + 1]
    if roi_det3_e.size != 0: ub_phi_e[roi_det3_e.min():roi_det3_e.max() + 1] *= np.sum(meas_vec_det3[0])/gf_det3_e[roi_det3_e.min():roi_det3_e.max() + 1]

    lb_phi_gam = np.zeros(rspns_mat_det3_gam[1][0].size - 1)
    
    ub_phi_gam = np.ones(rspns_mat_det3_gam[1][0].size - 1)
    if roi_det1_gam.size != 0: ub_phi_gam[roi_det1_gam.min():roi_det1_gam.max() + 1] *= np.sum(meas_vec_det1[0])/gf_det1_gam[roi_det1_gam.min():roi_det1_gam.max() + 1]
    if roi_det2_gam.size != 0: ub_phi_gam[roi_det2_gam.min():roi_det2_gam.max() + 1] *= np.sum(meas_vec_det2[0])/gf_det2_gam[roi_det2_gam.min():roi_det2_gam.max() + 1]
    if roi_det3_gam.size != 0: ub_phi_gam[roi_det3_gam.min():roi_det3_gam.max() + 1] *= np.sum(meas_vec_det3[0])/gf_det3_gam[roi_det3_gam.min():roi_det3_gam.max() + 1]

    # Define the prior probability densities
    phi_e = pm.Uniform('phi_e', lower = lb_phi_e, upper = ub_phi_e, shape = (ub_phi_e.size))
    phi_gam = pm.Uniform('phi_gam', lower = lb_phi_gam, upper = ub_phi_gam, shape = (ub_phi_gam.size))
    
    # Define the generative models
    M_det1 = theano.tensor.dot(rspns_mat_det1_e[0].T, phi_e) + theano.tensor.dot(rspns_mat_det1_gam[0].T, phi_gam)
    M_det2 = theano.tensor.dot(rspns_mat_det2_e[0].T, phi_e) + theano.tensor.dot(rspns_mat_det2_gam[0].T, phi_gam)
    M_det3 = theano.tensor.dot(rspns_mat_det3_e[0].T, phi_e) + theano.tensor.dot(rspns_mat_det3_gam[0].T, phi_gam)

    # Define the likelihood (aka. posterior probability function)
    PPF_det1 = pm.Poisson('PPF_det1', mu = M_det1, observed = meas_vec_det1[0], shape = meas_vec_det1[0].size)
    PPF_det2 = pm.Poisson('PPF_det2', mu = M_det2, observed = meas_vec_det2[0], shape = meas_vec_det2[0].size)
    PPF_det3 = pm.Poisson('PPF_det3', mu = M_det3, observed = meas_vec_det3[0], shape = meas_vec_det3[0].size)

with model_det1_det2_det3:
    print 'Sampling the posterior distribution ...'

    # Sample
    trace = pm.sample(draws = DRAWS,
                      tune = TUNE,
                      step = pm.HamiltonianMC(target_accept=0.99),
                      start = pm.find_MAP(),
                      use_mmap = True,
                      compute_convergence_checks = True)
    
    plotReconstructedSpectrum(trace, isotope + ' - ' + det1 + ' - ' + det2 + ' - ' + det3 + ' - Unfolded Fluence Spectrum.jpg')
