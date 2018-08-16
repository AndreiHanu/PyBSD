# Command Line Parsing Module
import argparse

import numpy as np
import pymc3 as pm
import ROOT 

# PyMC3 SMC Module
from pymc3.step_methods import smc
from tempfile import mkdtemp
import shutil

# ROOT-Numpy
from root_numpy import root2array, hist2array, matrix

# Theano
import theano
import theano.tensor

# Copy function
import copy

# Matplotlib - 2D plotting library
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
#import seaborn.apionly as sns
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import Grid, AxesGrid

# Pandas
import pandas

# Scipy
import scipy.stats as st
from scipy.stats.mstats import mode
from scipy import interpolate

# Color palette library for Python
# How to choose a colour scheme for your data:
# http://earthobservatory.nasa.gov/blogs/elegantfigures/2013/08/05/subtleties-of-color-part-1-of-6/
import palettable

##########################################################################################
# Setting rcParams for publication quality graphs
fig_width_pt = 246.0                    # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0      # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # Width in inches
fig_height = fig_width*golden_mean      # Height in inches
fig_size =  [fig_width, fig_height]
fig_size =  np.array([7.3,4.2])*1.25
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

class PyMBSD(object):
    
    def __init__(self,
                MigBetaPlastic, MigGammaPlastic, MigBetaLaBr3, MigGammaLaBr3,
                SourceBetaPlastic, SourceGammaPlastic, SourceBetaLaBr3,SourceGammaLaBr3,
                ThresholdEnergy):
        '''
        Object initialization function
        '''

        # Check the input instance type as follows:
        if not isinstance(MigBetaPlastic, ROOT.TH2): raise TypeError("Beta migration matrix for Plastic detector must be of type ROOT.TH2")
        if not isinstance(MigGammaPlastic, ROOT.TH2): raise TypeError("Gamma migration matrix for Plastic detector must be of type ROOT.TH2")
        if not isinstance(MigBetaLaBr3, ROOT.TH2): raise TypeError("Beta migration matrix for LaBr3 detector must be of type ROOT.TH2")
        if not isinstance(MigGammaLaBr3, ROOT.TH2): raise TypeError("Gamma migration matrix for LaBr3 detector must be of type ROOT.TH2")
        if not isinstance(SourceBetaPlastic, ROOT.TH1): raise TypeError("Beta source spectrum for the Plastic detector must be of type ROOT.TH1")
        if not isinstance(SourceGammaPlastic, ROOT.TH1): raise TypeError("Gamma source spectrum for the Plastic detector must be of type ROOT.TH1")
        if not isinstance(SourceBetaLaBr3, ROOT.TH1): raise TypeError("Beta source spectrum for the LaBr3 detector must be of type ROOT.TH1")
        if not isinstance(SourceGammaLaBr3, ROOT.TH1): raise TypeError("Gamma source spectrum for the LaBr3 detector must be of type ROOT.TH1")

        # Set the threshold energy
        self.ThresholdEnergy = np.double(ThresholdEnergy) if ThresholdEnergy > 0. else np.double(0.)

        # Copy the inputs to the object
        # NOTE: Convert tuple to ndarray so we can remove channels below threshold energy
        self.MigBetaPlastic = np.asarray(hist2array(MigBetaPlastic, include_overflow=False, copy=True, return_edges=True))
        self.MigGammaPlastic = np.asarray(hist2array(MigGammaPlastic, include_overflow=False, copy=True, return_edges=True))
        self.MigBetaLaBr3 = np.asarray(hist2array(MigBetaLaBr3, include_overflow=False, copy=True, return_edges=True))
        self.MigGammaLaBr3 = np.asarray(hist2array(MigGammaLaBr3, include_overflow=False, copy=True, return_edges=True))
        self.SourceBetaPlastic = np.asarray(hist2array(SourceBetaPlastic, include_overflow=False, copy=True, return_edges=True))
        self.SourceGammaPlastic = np.asarray(hist2array(SourceGammaPlastic, include_overflow=False, copy=True, return_edges=True))
        self.SourceBetaLaBr3 = np.asarray(hist2array(SourceBetaLaBr3, include_overflow=False, copy=True, return_edges=True))
        self.SourceGammaLaBr3 = np.asarray(hist2array(SourceGammaLaBr3, include_overflow=False, copy=True, return_edges=True))

        # Calculate the response matrix (aka. conditional probability) using Eq. 5 from the Choudalakis paper
        # Response[i,j] = P(d = j|t = i) = P(t = i, d = j)/P(t = i)
        # Response[j|i] = M[d = j, t = i] / Truth[i]
        self.ResponseBetaPlastic = copy.deepcopy(self.MigBetaPlastic)
        for i in np.arange(self.ResponseBetaPlastic[1][0].size - 1):
            for j in np.arange(self.ResponseBetaPlastic[1][1].size - 1):
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Normalize with Truth
                    self.ResponseBetaPlastic[0][i,j]=(self.ResponseBetaPlastic[0][i,j]/self.SourceBetaPlastic[0][i] if np.nonzero(self.SourceBetaPlastic[0][i]) else 0.)
        
        self.ResponseGammaPlastic = copy.deepcopy(self.MigGammaPlastic)
        for i in np.arange(self.ResponseGammaPlastic[1][0].size - 1):
            for j in np.arange(self.ResponseGammaPlastic[1][1].size - 1):
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Normalize with Truth
                    self.ResponseGammaPlastic[0][i,j]=(self.ResponseGammaPlastic[0][i,j]/self.SourceGammaPlastic[0][i] if np.nonzero(self.SourceGammaPlastic[0][i]) else 0.)

        self.ResponseBetaLaBr3 = copy.deepcopy(self.MigBetaLaBr3)
        for i in np.arange(self.ResponseBetaLaBr3[1][0].size - 1):
            for j in np.arange(self.ResponseBetaLaBr3[1][1].size - 1):
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Normalize with Truth
                    self.ResponseBetaLaBr3[0][i,j]=(self.ResponseBetaLaBr3[0][i,j]/self.SourceBetaLaBr3[0][i] if np.nonzero(self.SourceBetaLaBr3[0][i]) else 0.)

        self.ResponseGammaLaBr3 = copy.deepcopy(self.MigGammaLaBr3)
        for i in np.arange(self.ResponseGammaLaBr3[1][0].size - 1):
            for j in np.arange(self.ResponseGammaLaBr3[1][1].size - 1):
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Normalize with Truth
                    self.ResponseGammaLaBr3[0][i,j]=(self.ResponseGammaLaBr3[0][i,j]/self.SourceGammaLaBr3[0][i] if np.nonzero(self.SourceGammaLaBr3[0][i]) else 0.)

        # Round the response matrix bin edges to avoid the floating point rounding error when comparing against threshold
        self.ResponseBetaPlastic[1] = np.round(self.ResponseBetaPlastic[1], 5)
        self.ResponseGammaPlastic[1] = np.round(self.ResponseGammaPlastic[1], 5)
        self.ResponseBetaLaBr3[1] = np.round(self.ResponseBetaLaBr3[1], 5)
        self.ResponseGammaLaBr3[1] = np.round(self.ResponseGammaLaBr3[1], 5)
        
        # Remove response matrix elements below threshold energy (high pass filter)
        self.ResponseBetaPlastic[0] = np.delete(self.ResponseBetaPlastic[0], np.where(self.ResponseBetaPlastic[1][0] < self.ThresholdEnergy), axis=0)
        self.ResponseBetaPlastic[0] = np.delete(self.ResponseBetaPlastic[0], np.where(self.ResponseBetaPlastic[1][0] < self.ThresholdEnergy), axis=1)
        self.ResponseBetaPlastic[1] = np.delete(self.ResponseBetaPlastic[1], np.where(self.ResponseBetaPlastic[1][0] < self.ThresholdEnergy), axis=1)
        self.ResponseGammaPlastic[0] = np.delete(self.ResponseGammaPlastic[0], np.where(self.ResponseGammaPlastic[1][0] < self.ThresholdEnergy), axis=0)
        self.ResponseGammaPlastic[0] = np.delete(self.ResponseGammaPlastic[0], np.where(self.ResponseGammaPlastic[1][0] < self.ThresholdEnergy), axis=1)
        self.ResponseGammaPlastic[1] = np.delete(self.ResponseGammaPlastic[1], np.where(self.ResponseGammaPlastic[1][0] < self.ThresholdEnergy), axis=1)
        self.ResponseBetaLaBr3[0] = np.delete(self.ResponseBetaLaBr3[0], np.where(self.ResponseBetaLaBr3[1][0] < self.ThresholdEnergy), axis=0)
        self.ResponseBetaLaBr3[0] = np.delete(self.ResponseBetaLaBr3[0], np.where(self.ResponseBetaLaBr3[1][0] < self.ThresholdEnergy), axis=1)
        self.ResponseBetaLaBr3[1] = np.delete(self.ResponseBetaLaBr3[1], np.where(self.ResponseBetaLaBr3[1][0] < self.ThresholdEnergy), axis=1)
        self.ResponseGammaLaBr3[0] = np.delete(self.ResponseGammaLaBr3[0], np.where(self.ResponseGammaLaBr3[1][0] < self.ThresholdEnergy), axis=0)
        self.ResponseGammaLaBr3[0] = np.delete(self.ResponseGammaLaBr3[0], np.where(self.ResponseGammaLaBr3[1][0] < self.ThresholdEnergy), axis=1)
        self.ResponseGammaLaBr3[1] = np.delete(self.ResponseGammaLaBr3[1], np.where(self.ResponseGammaLaBr3[1][0] < self.ThresholdEnergy), axis=1)

    def plotResponse(self,  fName='ResponseMatrix.jpg'):
        '''
        Function to plot the response matrices
        '''

       # Create a figure to plot the spectrum
        figResp = plt.figure()

        axResp = AxesGrid(figResp, 111,
                        nrows_ncols=(2, 2),
                        axes_pad=0.3,
                        aspect=False,
                        label_mode = 'L',
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1,
                        cbar_size = 0.2)

        # Color map
        cmap = palettable.matplotlib.Viridis_20.mpl_colormap
        cmap.set_bad(cmap(0.))
        cmap.set_over(cmap(1.))

        # Response Limits
        rLimLow = 1E-2
        rLimUp = 1E2

        # Plot the response matrices
        X, Y = np.meshgrid(self.ResponseBetaPlastic[1][0], self.ResponseBetaPlastic[1][1])
        H0 = axResp[0].pcolormesh(X, Y, self.ResponseBetaPlastic[0].T, norm = colors.LogNorm(), cmap = cmap, linewidth = 0, rasterized = True) 

        X, Y = np.meshgrid(self.ResponseGammaPlastic[1][0], self.ResponseGammaPlastic[1][1])
        H1 = axResp[1].pcolormesh(X, Y, self.ResponseGammaPlastic[0].T, norm = colors.LogNorm(), cmap = cmap, linewidth = 0, rasterized = True) 

        X, Y = np.meshgrid(self.ResponseBetaLaBr3[1][0], self.ResponseBetaLaBr3[1][1])
        H2 = axResp[2].pcolormesh(X, Y, self.ResponseBetaLaBr3[0].T, norm = colors.LogNorm(), cmap = cmap, linewidth = 0, rasterized = True) 

        X, Y = np.meshgrid(self.ResponseGammaLaBr3[1][0], self.ResponseGammaLaBr3[1][1])
        H3 = axResp[3].pcolormesh(X, Y, self.ResponseGammaLaBr3[0].T, norm = colors.LogNorm(), cmap = cmap, linewidth = 0, rasterized = True) 

        # Color limits for the plot
        H0.set_clim(rLimLow, rLimUp)
        H1.set_clim(rLimLow, rLimUp)
        H2.set_clim(rLimLow, rLimUp)
        H3.set_clim(rLimLow, rLimUp)

        # Colorbar     
        from matplotlib.ticker import LogLocator
        axResp.cbar_axes[0].colorbar(H3, spacing = 'uniform')
        axResp.cbar_axes[0].set_yscale('log')
        axResp.cbar_axes[0].axis[axResp.cbar_axes[0].orientation].set_label('Omnidirectional Response (cm$^2$)')

        # Figure Properties
        axResp[0].set_xscale('log')
        axResp[0].set_yscale('log')
        axResp[0].set_ylabel('Measured Energy (keV)')
        axResp[0].text(0.03,0.95, 'Eljen M550-20x8-1 Plastic Detector', transform=axResp[0].transAxes, verticalalignment='top', color = 'white')

        axResp[1].set_xscale('log')
        axResp[1].set_yscale('log')
        axResp[1].text(0.03,0.95, 'Eljen M550-20x8-1 Plastic Detector', transform=axResp[1].transAxes, verticalalignment='top', color = 'white')

        axResp[2].set_xscale('log')
        axResp[2].set_yscale('log')
        axResp[2].set_xlabel('True Beta-ray Energy (keV)')
        axResp[2].set_ylabel('Measured Energy (keV)')
        axResp[2].text(0.03,0.95, 'Saint Gobain B380 LaBr3', transform=axResp[2].transAxes, verticalalignment='top', color = 'white')

        axResp[3].set_xscale('log')
        axResp[3].set_yscale('log')
        axResp[3].set_xlabel('True Gamma-ray Energy (keV)')
        axResp[3].text(0.03,0.95, 'Saint Gobain B380 LaBr3', transform=axResp[3].transAxes, verticalalignment='top', color = 'white')

        # Fine-tune figure 
        figResp.tight_layout()
        figResp.subplots_adjust(wspace=0.05,hspace=0.05)

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Response matrix plot saved to: ' + fName

        # Close the figure
        plt.close(figResp)
    
    def loadDoseCoeffGamma(self, fName=''):
        '''
        Function to load the ICRP 116 gamma-ray dose coefficients for the whole body, skin, and lens of the eye.
        '''

        # Import coefficients from files
        df_ICRP116_Photon_WholeBody = pandas.read_excel(fName, sheet_name = 'Effective Dose (Whole Body)')
        df_ICRP116_Photon_FemaleSkin = pandas.read_excel(fName, sheet_name = 'Absorbed Dose (Female Skin)')
        df_ICRP116_Photon_MaleSkin = pandas.read_excel(fName, sheet_name = 'Absorbed Dose (Male Skin)')
        df_ICRP116_Photon_EyeLens = pandas.read_excel(fName, sheet_name = 'Absorbed Dose (Lens of Eye)')

        # Interpolate the coefficients into the true log energy bins
        def logInterpCoeff(coeffBins, coeffX, coeffY):
            midBin = [np.log10(coeffBins[i]*coeffBins[i + 1])/2 for i in range(0, len(coeffBins)-1)]
            return np.nan_to_num(np.power(10, interpolate.interp1d(np.log10(coeffX), np.log10(coeffY), kind='linear')(midBin)))

        # Scaling factor to convert the magnitude of the dose coefficients
        coeffScalingFactor = 1E-12*1E9 # pSv/pGy to nSv/nGy

        # Load the dose coefficients into self object
        # NOTE: Energy scaling from MeV to keV
        self.coeffGammaWB = np.array([logInterpCoeff(self.ResponseBetaPlastic[1][0], 
                                                     df_ICRP116_Photon_WholeBody['Energy (MeV)'].values*1E3, 
                                                     df_ICRP116_Photon_WholeBody['ISO (pSv cm2)'].values*coeffScalingFactor),
                                      self.ResponseBetaPlastic[1][0]])
        
        self.coeffGammaSkin = np.array([logInterpCoeff(self.ResponseBetaPlastic[1][0], 
                                                     df_ICRP116_Photon_MaleSkin['Energy (MeV)'].values*1E3, 
                                                     df_ICRP116_Photon_MaleSkin['ISO (pGy cm2)'].values*coeffScalingFactor),
                                      self.ResponseBetaPlastic[1][0]])
        
        self.coeffGammaEye = np.array([logInterpCoeff(self.ResponseBetaPlastic[1][0], 
                                                     df_ICRP116_Photon_EyeLens['Energy (MeV)'].values*1E3, 
                                                     df_ICRP116_Photon_EyeLens['ISO (pGy cm2)'].values*coeffScalingFactor),
                                      self.ResponseBetaPlastic[1][0]])
    
    def loadDoseCoeffBeta(self, fName=''):
        '''
        Function to load the ICRP 116 beta-ray dose coefficients for the whole body, skin, and lens of the eye.
        '''

        # Import coefficients from files
        df_ICRP116_Beta_WholeBody = pandas.read_excel(fName, sheet_name = 'Effective Dose (Whole Body)')
        df_ICRP116_Beta_Skin = pandas.read_excel(fName, sheet_name = 'Absorbed Dose (Average Skin)')
        df_ICRP116_Beta_EyeLens = pandas.read_excel(fName, sheet_name = 'Absorbed Dose (Lens of Eye)')

        # Interpolate the coefficients into the true log energy bins
        def logInterpCoeff(coeffBins, coeffX, coeffY):
            midBin = [np.log10(coeffBins[i]*coeffBins[i + 1])/2 for i in range(0, len(coeffBins)-1)]
            return np.nan_to_num(np.power(10, interpolate.interp1d(np.log10(coeffX), np.log10(coeffY), kind='linear')(midBin)))

        # Scaling factor to convert the magnitude of the dose coefficients
        coeffScalingFactor = 1E-12*1E9 # pSv/pGy to nSv/nGy

        # Load the dose coefficients into self object
        # NOTE: Energy scaling from MeV to keV
        self.coeffBetaWB = np.array([logInterpCoeff(self.ResponseBetaPlastic[1][0], 
                                                    df_ICRP116_Beta_WholeBody['Energy (MeV)'].values*1E3, 
                                                    df_ICRP116_Beta_WholeBody['ISO (pSv cm2)'].values*coeffScalingFactor),
                                      self.ResponseBetaPlastic[1][0]])

        self.coeffBetaSkin = np.array([logInterpCoeff(self.ResponseBetaPlastic[1][0], 
                                                    df_ICRP116_Beta_Skin['Energy (MeV)'].values*1E3, 
                                                    df_ICRP116_Beta_Skin['ISO (pGy cm2)'].values*coeffScalingFactor),
                                      self.ResponseBetaPlastic[1][0]])
        
        self.coeffBetaEye = np.array([logInterpCoeff(self.ResponseBetaPlastic[1][0], 
                                                     df_ICRP116_Beta_EyeLens['Energy (MeV)'].values*1E3, 
                                                     df_ICRP116_Beta_EyeLens['ISO (pGy cm2)'].values*coeffScalingFactor),
                                      self.ResponseBetaPlastic[1][0]])

    def plotUnfoldedFluenceSpectrum(self, fName='UnfoldedFluenceSpectrum.pdf'):
        '''
        Function to plot the reconstructed fluence spectrum after performing multidimensional Bayesian unfolding
        NOTE: This function is to be used when multiple response matrices are used in the unfolding.
        '''

        # Calculate and plot the 95% Bayesian credible regions for the unfolded spectrum
        unfoldedBCIBeta = pm.stats.hpd(self.trace['PhiBeta'], alpha=0.05)
        unfoldedBCIGamma = pm.stats.hpd(self.trace['PhiGamma'], alpha=0.05)

        binRecoVal = np.array([unfoldedBCIBeta[:,0],                        # Beta 2.5% HPD
                               unfoldedBCIGamma[:,0],                       # Gamma 2.5% HPD
                               np.mean(self.trace['PhiBeta'],0),            # Beta Mean
                               np.mean(self.trace['PhiGamma'],0),           # Gamma Mean
                               unfoldedBCIBeta[:,1],                        # Beta 97.5% HPD
                               unfoldedBCIGamma[:,1]])                      # Gamma 97.5% HPD

        # Create a figure to plot the spectrum
        figFluence, axFluence = plt.subplots(2,2, figsize=(fig_size[0]*1.5,fig_size[1]*1.5))
        
        # Plot the data spectrum
        axFluence[0][0].plot(sorted(np.concatenate((self.DataPlastic[1][:-1],self.DataPlastic[1][1:]))), 
                    np.repeat(self.DataPlastic[0], 2),
                    lw=1.25, 
                    color='black', 
                    linestyle="-",
                    drawstyle='steps')
        
        axFluence[0][1].plot(sorted(np.concatenate((self.DataLaBr3[1][:-1],self.DataLaBr3[1][1:]))),  
                    np.repeat(self.DataLaBr3[0], 2),
                    lw=1.25, 
                    color='black', 
                    linestyle="-",
                    drawstyle='steps')

        minY = 1.
        maxY = np.maximum(np.power(10, np.ceil(np.log10(np.max(self.DataPlastic[0])))),
                          np.power(10, np.ceil(np.log10(np.max(self.DataLaBr3[0])))))

        axFluence[0][0].set_title('Measured Spectrum from Eljen M550-20x8-1 Plastic Detector')
        axFluence[0][0].set_xlabel('Measured Energy (keV)')
        axFluence[0][0].set_ylabel('Count Rate (cps)')
        axFluence[0][0].set_xlim(min(self.DataPlastic[1]),max(self.DataPlastic[1]))
        axFluence[0][0].set_ylim(minY, maxY)
        axFluence[0][0].set_xscale('log')
        axFluence[0][0].set_yscale('log')

        axFluence[0][1].set_title('Measured Spectrum from Saint Gobain B380 LaBr3 Detector')
        axFluence[0][1].set_xlabel('Measured Energy (keV)')
        axFluence[0][1].set_ylabel('Count Rate (cps)')
        axFluence[0][1].set_xlim(min(self.DataLaBr3[1]),max(self.DataLaBr3[1]))
        axFluence[0][1].set_ylim(minY, maxY)
        axFluence[0][1].set_xscale('log')
        axFluence[0][1].set_yscale('log')

        # Plot the true fluence spectrum, if available.
        if self.TruthBeta is not None:
            pTruthBeta, = axFluence[1][0].plot(sorted(np.concatenate((self.TruthBeta[1][1:],self.TruthBeta[1][:-1]))), 
                                            np.repeat(self.TruthBeta[0], 2),
                                            lw=1.25, 
                                            color='black', 
                                            linestyle="-", 
                                            drawstyle='steps')

        if self.TruthGamma is not None:
            pTruthGamma, = axFluence[1][1].plot(sorted(np.concatenate((self.TruthGamma[1][1:],self.TruthGamma[1][:-1]))), 
                                            np.repeat(self.TruthGamma[0], 2),
                                            lw=1.25, 
                                            color='black', 
                                            linestyle="-", 
                                            drawstyle='steps')

        # Plot the unfolded spectrum
        pBCIBeta = axFluence[1][0].fill_between(sorted(np.concatenate((self.ResponseBetaPlastic[1][0][1:],self.ResponseBetaPlastic[1][0][:-1]))), 
                                            np.repeat(binRecoVal[0], 2), 
                                            np.repeat(binRecoVal[4], 2),
                                            color='red',
                                            alpha=0.4)
    
        pBCIGamma = axFluence[1][1].fill_between(sorted(np.concatenate((self.ResponseGammaLaBr3[1][0][1:],self.ResponseGammaLaBr3[1][0][:-1]))), 
                                            np.repeat(binRecoVal[1], 2), 
                                            np.repeat(binRecoVal[5], 2),
                                            color='red',
                                            alpha=0.4)

        pMeanBeta, = axFluence[1][0].plot(sorted(np.concatenate((self.ResponseBetaPlastic[1][0][1:],self.ResponseBetaPlastic[1][0][:-1]))), 
                                        np.repeat(binRecoVal[2], 2),
                                        lw=1.25, 
                                        color='red', 
                                        linestyle="-", 
                                        drawstyle='steps')
       
        pMeanGamma, = axFluence[1][1].plot(sorted(np.concatenate((self.ResponseGammaLaBr3[1][0][1:],self.ResponseGammaLaBr3[1][0][:-1]))), 
                                        np.repeat(binRecoVal[3], 2),
                                        lw=1.25, 
                                        color='red', 
                                        linestyle="-", 
                                        drawstyle='steps')

        minY = np.min(binRecoVal[binRecoVal >= 1E-2])
        maxY = np.max(binRecoVal[np.isfinite(binRecoVal)])

        axFluence[1][0].set_xlabel('True Energy (keV)')
        axFluence[1][0].set_title('Reconstructed Beta-ray Fluence Spectrum')
        axFluence[1][0].set_ylabel('Fluence Rate (cm$^{-2}$ s$^{-1}$)')
        axFluence[1][0].set_xscale('log')
        axFluence[1][0].set_yscale('log')
        axFluence[1][0].set_xlim(min(self.ResponseBetaPlastic[1][0]),max(self.ResponseBetaPlastic[1][0]))
        axFluence[1][0].set_ylim(np.power(10, np.floor(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        if self.TruthBeta is not None:
            axFluence[1][0].legend([pTruthBeta, (pBCIBeta, pMeanBeta)], ['Truth','Reconstructed (95% BCI)'], loc='best')
        else:
            axFluence[1][0].legend([(pBCIBeta, pMeanBeta)], ['Reconstructed (95% BCI)'], loc='best')

        axFluence[1][1].set_xlabel('True Energy (keV)')
        axFluence[1][1].set_title('Reconstructed Gamma-ray Fluence Spectrum')
        axFluence[1][1].set_ylabel('Fluence Rate (cm$^{-2}$ s$^{-1}$)')
        axFluence[1][1].set_xscale('log')
        axFluence[1][1].set_yscale('log')
        axFluence[1][1].set_xlim(min(self.ResponseGammaLaBr3[1][0]),max(self.ResponseGammaLaBr3[1][0]))
        axFluence[1][1].set_ylim(np.power(10, np.floor(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        if self.TruthGamma is not None:
            axFluence[1][1].legend([pTruthGamma, (pBCIGamma, pMeanGamma)], ['Truth','Reconstructed (95% BCI)'], loc='best')
        else:
            axFluence[1][1].legend([(pBCIGamma, pMeanGamma)], ['Reconstructed (95% BCI)'], loc='best')
        
        # Fine-tune figure 
        figFluence.tight_layout()

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Unfolded plot saved to: ' + fName

        # Show the figure
        plt.close(figFluence)

    def calcSignificance(self, expected, observed):
        '''
        Function to calculate the statistical significance between two spectra using the algorithm described in Ref [1].

        References:
        [1] Choudalakis, Georgios, and Diego Casadei. "Plotting the differences between data and expectation." The European Physical Journal Plus 127.2 (2012): 25.
        '''
        pvalue = np.zeros(observed.size)
        zscore = np.zeros(observed.size)

        # Calculate the significance
        for i in range(observed.size):
            # Calculate the p-value
            if observed[i] > expected[i]:
                # Excess
                pvalue[i] = ROOT.Math.inc_gamma_c(observed[i], expected[i])
            else:
                # Deficit
                pvalue[i] = ROOT.Math.inc_gamma_c(observed[i] + 1, expected[i])

            # Calculate the significance
            if observed[i] > expected[i]:
                # Excess
                zscore[i] = ROOT.Math.normal_quantile(1-pvalue[i],1)
            else:
                # Deficit
                zscore[i] = ROOT.Math.normal_quantile(pvalue[i],1)
                

            # If there is a deficit of events, and its statistically significant (ie. pvalue < 0.5)
            # then take the negative of the z-value.
            if observed[i] < expected[i] and pvalue[i] < 0.5:
                zscore[i] *= -1.

        # Return signed z-score only if p-value < 0.5
        # See: https://arxiv.org/pdf/1111.2062.pdf
        zscore[pvalue >= 1.] = 0.

        return zscore

    def plotFoldedMeasuredSpectrum(self, fName='FoldedMeasuredSpectrum.pdf'):
        '''
        Function to plot the measured spectrum after folding the reconstructed fluence spectrum with 
        the response function for each detector.
        '''

        # Calculate and plot the 95% Bayesian credible regions for the unfolded spectrum
        unfoldedBCIBeta = pm.stats.hpd(self.trace['PhiBeta'], alpha=0.05)
        unfoldedBCIGamma = pm.stats.hpd(self.trace['PhiGamma'], alpha=0.05)

        binRecoVal = np.array([unfoldedBCIBeta[:,0],                        # Beta 2.5% HPD
                               unfoldedBCIGamma[:,0],                       # Gamma 2.5% HPD
                               np.mean(self.trace['PhiBeta'],0),            # Beta Mean
                               np.mean(self.trace['PhiGamma'],0),           # Gamma Mean
                               unfoldedBCIBeta[:,1],                        # Beta 97.5% HPD
                               unfoldedBCIGamma[:,1]])                      # Gamma 97.5% HPD

        # Define the forward model
        binFoldedVal = np.array([np.dot(self.ResponseBetaPlastic[0].T, binRecoVal[0]) + np.dot(self.ResponseGammaPlastic[0].T, binRecoVal[1]),
                                np.dot(self.ResponseBetaLaBr3[0].T, binRecoVal[0]) + np.dot(self.ResponseGammaLaBr3[0].T, binRecoVal[1]),
                                np.dot(self.ResponseBetaPlastic[0].T, binRecoVal[2]) + np.dot(self.ResponseGammaPlastic[0].T, binRecoVal[3]),
                                np.dot(self.ResponseBetaLaBr3[0].T, binRecoVal[2]) + np.dot(self.ResponseGammaLaBr3[0].T, binRecoVal[3]),
                                np.dot(self.ResponseBetaPlastic[0].T, binRecoVal[4]) + np.dot(self.ResponseGammaPlastic[0].T, binRecoVal[5]),
                                np.dot(self.ResponseBetaLaBr3[0].T, binRecoVal[4]) + np.dot(self.ResponseGammaLaBr3[0].T, binRecoVal[5])])
        
        # Create a figure to plot the spectrum
        figFolded = plt.figure(1, figsize=(fig_size[0]*1.3,fig_size[1]*1.3))

        axFolded = Grid(figFolded, 111,
                        nrows_ncols=(2, 2),
                        axes_pad=(0.35, 0.),
                        add_all=True,
                        label_mode = 'L')
        
        # Plot the data spectrum
        pTruthPlastic, = axFolded[0].plot(sorted(np.concatenate((self.DataPlastic[1][:-1],self.DataPlastic[1][1:]))), 
                            np.repeat(self.DataPlastic[0], 2),
                            lw=1.25, 
                            color='black', 
                            linestyle="-",
                            drawstyle='steps')
        
        pTruthLaBr3, = axFolded[1].plot(sorted(np.concatenate((self.DataLaBr3[1][:-1],self.DataLaBr3[1][1:]))),  
                            np.repeat(self.DataLaBr3[0], 2),
                            lw=1.25, 
                            color='black', 
                            linestyle="-",
                            drawstyle='steps')

        # Plot the folded spectrum
        pBCIPlastic = axFolded[0].fill_between(sorted(np.concatenate((self.DataPlastic[1][:-1],self.DataPlastic[1][1:]))), 
                                np.repeat(binFoldedVal[0], 2), 
                                np.repeat(binFoldedVal[4], 2),
                                color='red',
                                alpha=0.4)

        pBCILaBr3 = axFolded[1].fill_between(sorted(np.concatenate((self.DataLaBr3[1][:-1],self.DataLaBr3[1][1:]))), 
                                np.repeat(binFoldedVal[1], 2), 
                                np.repeat(binFoldedVal[5], 2),
                                color='red',
                                alpha=0.4)

        pMeanPlastic, = axFolded[0].plot(sorted(np.concatenate((self.DataPlastic[1][:-1],self.DataPlastic[1][1:]))), 
                            np.repeat(binFoldedVal[2], 2),
                            lw=1.25, 
                            color='red', 
                            linestyle="-",
                            drawstyle='steps')

        pMeanLaBr3, = axFolded[1].plot(sorted(np.concatenate((self.DataLaBr3[1][:-1],self.DataLaBr3[1][1:]))), 
                            np.repeat(binFoldedVal[3], 2),
                            lw=1.25, 
                            color='red', 
                            linestyle="-",
                            drawstyle='steps')

        minY = 1.
        maxY = np.maximum(np.power(10, np.ceil(np.log10(np.max(self.DataPlastic[0])))),
                          np.power(10, np.ceil(np.log10(np.max(self.DataLaBr3[0])))))
        
        axFolded[0].set_title('Measured Spectrum from Eljen Plastic Detector')
        axFolded[0].set_xlabel('Measured Energy (keV)')
        axFolded[0].set_ylabel('Count Rate (cps)')
        axFolded[0].set_xlim(min(self.DataPlastic[1]),max(self.DataPlastic[1]))
        axFolded[0].set_ylim(minY, maxY)
        axFolded[0].set_xscale('log')
        axFolded[0].set_yscale('log')
        axFolded[0].legend([pTruthPlastic, (pBCIPlastic, pMeanPlastic)], ['Measured','Folded (95% BCI)'], loc='best')

        axFolded[1].set_title('Measured Spectrum from Saint Gobain LaBr3 Detector')
        axFolded[1].set_xlabel('Measured Energy (keV)')
        axFolded[1].set_ylabel('Count Rate (cps)')
        axFolded[1].set_xlim(min(self.DataLaBr3[1]),max(self.DataLaBr3[1]))
        axFolded[1].set_ylim(minY, maxY)
        axFolded[1].set_xscale('log')
        axFolded[1].set_yscale('log')
        axFolded[1].legend([pTruthLaBr3, (pBCILaBr3, pMeanLaBr3)], ['Measured','Folded (95% BCI)'], loc='best')

        # Plot the significance between measured and reconstructed spectrum
        axFolded[2].fill_between(sorted(np.concatenate((self.DataPlastic[1][:-1],self.DataPlastic[1][1:]))),
                                0, 
                                np.repeat(self.calcSignificance(binFoldedVal[2], self.DataPlastic[0]), 2), 
                                where = np.repeat(self.calcSignificance(binFoldedVal[2], self.DataPlastic[0]), 2) > 0,
                                color='red', 
                                alpha=0.2)
        
        axFolded[2].fill_between(sorted(np.concatenate((self.DataPlastic[1][:-1],self.DataPlastic[1][1:]))),
                                0, 
                                np.repeat(self.calcSignificance(binFoldedVal[2], self.DataPlastic[0]), 2), 
                                where = np.repeat(self.calcSignificance(binFoldedVal[2], self.DataPlastic[0]), 2) < 0,
                                color='blue', 
                                alpha=0.2)
        
        axFolded[2].plot(sorted(np.concatenate((self.DataPlastic[1][:-1],self.DataPlastic[1][1:]))),
                        np.repeat(self.calcSignificance(binFoldedVal[2], self.DataPlastic[0]), 2), 
                        lw=1.25, 
                        color='black', 
                        linestyle="-", 
                        drawstyle='steps')

        axFolded[3].fill_between(sorted(np.concatenate((self.DataLaBr3[1][:-1],self.DataLaBr3[1][1:]))),
                                0, 
                                np.repeat(self.calcSignificance(binFoldedVal[3], self.DataLaBr3[0]), 2), 
                                where = np.repeat(self.calcSignificance(binFoldedVal[3], self.DataLaBr3[0]), 2) > 0,
                                color='red', 
                                alpha=0.2)
        
        axFolded[3].fill_between(sorted(np.concatenate((self.DataLaBr3[1][:-1],self.DataLaBr3[1][1:]))),
                                0, 
                                np.repeat(self.calcSignificance(binFoldedVal[3], self.DataLaBr3[0]), 2), 
                                where = np.repeat(self.calcSignificance(binFoldedVal[3], self.DataLaBr3[0]), 2) < 0,
                                color='blue', 
                                alpha=0.2)
        
        axFolded[3].plot(sorted(np.concatenate((self.DataLaBr3[1][:-1],self.DataLaBr3[1][1:]))),
                        np.repeat(self.calcSignificance(binFoldedVal[3], self.DataLaBr3[0]), 2), 
                        lw=1.25, 
                        color='black', 
                        linestyle="-", 
                        drawstyle='steps')

        axFolded[2].set_ylabel('Significance')  
        axFolded[2].set_xlabel('Measured Energy (keV)')
        axFolded[2].set_xlim(min(self.DataPlastic[1]),max(self.DataPlastic[1]))
        axFolded[2].set_ylim(-5,5)
        axFolded[2].set_xscale('log', nonposy='clip')

        axFolded[3].set_ylabel('Significance')  
        axFolded[3].set_xlabel('Measured Energy (keV)')
        axFolded[3].set_xlim(min(self.DataLaBr3[1]),max(self.DataLaBr3[1]))
        axFolded[3].set_ylim(-5,5)
        axFolded[3].set_xscale('log', nonposy='clip')
        
        # Fine-tune figure 
        figFolded.tight_layout()

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Folded plot saved to: ' + fName

        # Show the figure
        plt.close(figFolded)

        
    def plotUnfoldedDoseSpectrum(self, fName='UnfoldedDoseSpectrum.pdf'):
        '''
        Function to plot the reconstructed dose spectrum after performing multidimensional Bayesian unfolding
        NOTE: This function is to be used when multiple response matrices are used in the unfolding.
        '''

        # Calculate the true dose 
        # NOTE: The 0.36 is added to convert from nRem/s to mRem/hr
        binTruthDoseVal = np.array([(self.TruthBeta[0] if self.TruthBeta is not None else 0.)*self.coeffBetaWB[0]*0.36,                  # Beta WB Dose Mean
                                    (self.TruthBeta[0] if self.TruthBeta is not None else 0.)*self.coeffBetaSkin[0]*0.36,                 # Beta Skin Dose Mean 
                                    (self.TruthBeta[0] if self.TruthBeta is not None else 0.)*self.coeffBetaEye[0]*0.36,                  # Beta Eye Dose Mean
                                    (self.TruthGamma[0] if self.TruthGamma is not None else 0.)*self.coeffGammaWB[0]*0.36,                 # Gamma WB Dose Mean
                                    (self.TruthGamma[0] if self.TruthGamma is not None else 0.)*self.coeffGammaSkin[0]*0.36,               # Gamma Skin Dose Mean
                                    (self.TruthGamma[0] if self.TruthGamma is not None else 0.)*self.coeffGammaEye[0]*0.36])               # Gamma Eye Dose Mean

        # Calculate and plot the 95% Bayesian credible regions for the unfolded spectrum
        # NOTE: The 0.36 is added to convert from nRem/s to mRem/hr
        unfoldedBCIBeta = pm.stats.hpd(self.trace['PhiBeta'], alpha=0.05)
        unfoldedBCIGamma = pm.stats.hpd(self.trace['PhiGamma'], alpha=0.05)

        binRecoFluenceVal = np.array([unfoldedBCIBeta[:,0],                                 # Beta 2.5% HPD
                               unfoldedBCIGamma[:,0],                                       # Gamma 2.5% HPD
                               np.mean(self.trace['PhiBeta'],0),                            # Beta Mean
                               np.mean(self.trace['PhiGamma'],0),                           # Gamma Mean
                               unfoldedBCIBeta[:,1],                                        # Beta 97.5% HPD
                               unfoldedBCIGamma[:,1]])                                      # Gamma 97.5% HPD
        
        binRecoDoseVal = np.array([unfoldedBCIBeta[:,0]*self.coeffBetaWB[0]*0.36,                # Beta WB Dose 2.5% HPD (in mRem/hr)
                                   unfoldedBCIBeta[:,0]*self.coeffBetaSkin[0]*0.36,              # Beta Skin Dose 2.5% HPD (in mRad/hr)
                                   unfoldedBCIBeta[:,0]*self.coeffBetaEye[0]*0.36,               # Beta Eye Dose 2.5% HPD (in mRad/hr)
                                   unfoldedBCIGamma[:,0]*self.coeffGammaWB[0]*0.36,              # Gamma WB Dose 2.5% HPD (in mRem/hr)
                                   unfoldedBCIGamma[:,0]*self.coeffGammaSkin[0]*0.36,            # Gamma Skin Dose 2.5% HPD (in mRad/hr)
                                   unfoldedBCIGamma[:,0]*self.coeffGammaEye[0]*0.36,             # Gamma Eye Dose 2.5% HPD (in mRad/hr)
                                   np.mean(self.trace['PhiBeta'],0)*self.coeffBetaWB[0]*0.36,    # Beta WB Dose Mean (in mRem/hr)
                                   np.mean(self.trace['PhiBeta'],0)*self.coeffBetaSkin[0]*0.36,  # Beta Skin Dose Mean (in mRad/hr)
                                   np.mean(self.trace['PhiBeta'],0)*self.coeffBetaEye[0]*0.36,   # Beta Eye Dose Mean (in mRad/hr)
                                   np.mean(self.trace['PhiGamma'],0)*self.coeffGammaWB[0]*0.36,  # Gamma WB Dose Mean (in mRem/hr)
                                   np.mean(self.trace['PhiGamma'],0)*self.coeffGammaSkin[0]*0.36,# Gamma Skin Dose Mean (in mRad/hr)
                                   np.mean(self.trace['PhiGamma'],0)*self.coeffGammaEye[0]*0.36, # Gamma Eye Dose Mean (in mRad/hr)
                                   unfoldedBCIBeta[:,1]*self.coeffBetaWB[0]*0.36,                # Beta WB Dose 97.5% HPD (in mRem/hr)
                                   unfoldedBCIBeta[:,1]*self.coeffBetaSkin[0]*0.36,              # Beta Skin Dose 97.5% HPD (in mRad/hr)
                                   unfoldedBCIBeta[:,1]*self.coeffBetaEye[0]*0.36,               # Beta Eye Dose 97.5% HPD (in mRad/hr)
                                   unfoldedBCIGamma[:,1]*self.coeffGammaWB[0]*0.36,              # Gamma WB Dose 97.5% HPD (in mRem/hr)
                                   unfoldedBCIGamma[:,1]*self.coeffGammaSkin[0]*0.36,            # Gamma Skin Dose 97.5% HPD (in mRad/hr)
                                   unfoldedBCIGamma[:,1]*self.coeffGammaEye[0]*0.36])            # Gamma Eye Dose 97.5% HPD (in mRad/hr)

        # Create a figure to plot the spectrum
        figDose, axDose = plt.subplots(3,2, figsize=(fig_size[0]*2,fig_size[1]*2))

        # Plot the data spectrum
        axDose[0][0].plot(sorted(np.concatenate((self.DataPlastic[1][:-1],self.DataPlastic[1][1:]))), 
                    np.repeat(self.DataPlastic[0], 2),
                    lw=1.25, 
                    color='black', 
                    linestyle="-",
                    drawstyle='steps')
        
        axDose[0][1].plot(sorted(np.concatenate((self.DataLaBr3[1][:-1],self.DataLaBr3[1][1:]))),  
                    np.repeat(self.DataLaBr3[0], 2),
                    lw=1.25, 
                    color='black', 
                    linestyle="-",
                    drawstyle='steps')

        minY = 0.1
        maxY = np.maximum(np.power(10, np.ceil(np.log10(np.max(self.DataPlastic[0])))),
                          np.power(10, np.ceil(np.log10(np.max(self.DataLaBr3[0])))))

        axDose[0][0].set_title('Measured Spectrum from Eljen M550-20x8-1 Plastic Detector')
        axDose[0][0].set_xlabel('Measured Energy (keV)')
        axDose[0][0].set_ylabel('Count Rate (cps)')
        axDose[0][0].set_xlim(min(self.DataPlastic[1]),max(self.DataPlastic[1]))
        axDose[0][0].set_ylim(minY, maxY)
        axDose[0][0].set_xscale('log')
        axDose[0][0].set_yscale('log')

        axDose[0][1].set_title('Measured Spectrum from Saint Gobain B380 LaBr3 Detector')
        axDose[0][1].set_xlabel('Measured Energy (keV)')
        axDose[0][1].set_ylabel('Count Rate (cps)')
        axDose[0][1].set_xlim(min(self.DataLaBr3[1]),max(self.DataLaBr3[1]))
        axDose[0][1].set_ylim(minY, maxY)
        axDose[0][1].set_xscale('log')
        axDose[0][1].set_yscale('log')

        # Plot the true fluence spectrum, if available.
        if self.TruthBeta is not None:
            pTruthBeta, = axDose[1][0].plot(sorted(np.concatenate((self.ResponseBetaPlastic[1][0][1:],self.ResponseBetaPlastic[1][0][:-1]))), 
                                            np.repeat(self.TruthBeta[0], 2),
                                            lw=2., 
                                            color='black', 
                                            linestyle="-", 
                                            drawstyle='steps')
        
        if self.TruthGamma is not None:
            pTruthGamma, = axDose[1][1].plot(sorted(np.concatenate((self.ResponseGammaLaBr3[1][0][1:],self.ResponseGammaLaBr3[1][0][:-1]))), 
                                            np.repeat(self.TruthGamma[0], 2),
                                            lw=2., 
                                            color='black', 
                                            linestyle="-", 
                                            drawstyle='steps')

        # Plot the unfolded spectrum
        pBCIBeta = axDose[1][0].fill_between(sorted(np.concatenate((self.ResponseBetaPlastic[1][0][1:],self.ResponseBetaPlastic[1][0][:-1]))), 
                                            np.repeat(binRecoFluenceVal[0], 2), 
                                            np.repeat(binRecoFluenceVal[4], 2),
                                            color='red',
                                            alpha=0.4)
    
        pBCIGamma = axDose[1][1].fill_between(sorted(np.concatenate((self.ResponseGammaLaBr3[1][0][1:],self.ResponseGammaLaBr3[1][0][:-1]))), 
                                            np.repeat(binRecoFluenceVal[1], 2), 
                                            np.repeat(binRecoFluenceVal[5], 2),
                                            color='red',
                                            alpha=0.4)

        pMeanBeta, = axDose[1][0].plot(sorted(np.concatenate((self.ResponseBetaPlastic[1][0][1:],self.ResponseBetaPlastic[1][0][:-1]))), 
                                        np.repeat(binRecoFluenceVal[2], 2),
                                        lw=1.25, 
                                        color='red', 
                                        linestyle="-", 
                                        drawstyle='steps')
       

        pMeanGamma, = axDose[1][1].plot(sorted(np.concatenate((self.ResponseGammaLaBr3[1][1][1:],self.ResponseGammaLaBr3[1][1][:-1]))), 
                                        np.repeat(binRecoFluenceVal[3], 2),
                                        lw=1.25, 
                                        color='red', 
                                        linestyle="-", 
                                        drawstyle='steps')

        minY = np.min(binRecoFluenceVal[binRecoFluenceVal >= 1E-3])
        maxY = np.max(binRecoFluenceVal[np.isfinite(binRecoFluenceVal)])

        axDose[1][0].set_title('Reconstructed Beta-ray Fluence Spectrum')
        axDose[1][0].set_xlabel('True Energy (keV)')
        axDose[1][0].set_ylabel('Fluence Rate (cm$^{-2}$ s$^{-1}$)')
        axDose[1][0].set_xscale('log')
        axDose[1][0].set_yscale('log')
        axDose[1][0].set_xlim(min(self.ResponseBetaPlastic[1][0]),max(self.ResponseBetaPlastic[1][0]))
        axDose[1][0].set_ylim(np.power(10, np.floor(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        if self.TruthBeta is not None:
            axDose[1][0].legend([pTruthBeta, (pBCIBeta, pMeanBeta)], ['Truth','Reconstructed (95% BCI)'], loc='best')
        else:
            axDose[1][0].legend([(pBCIBeta, pMeanBeta)], ['Reconstructed (95% BCI)'], loc='best')

        axDose[1][1].set_title('Reconstructed Gamma-ray Fluence Spectrum')
        axDose[1][1].set_xlabel('True Energy (keV)')
        axDose[1][1].set_ylabel('Fluence Rate (cm$^{-2}$ s$^{-1}$)')
        axDose[1][1].set_xscale('log')
        axDose[1][1].set_yscale('log')
        axDose[1][1].set_xlim(min(self.ResponseGammaLaBr3[1][0]),max(self.ResponseGammaLaBr3[1][0]))
        axDose[1][1].set_ylim(np.power(10, np.floor(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        if self.TruthGamma is not None:
            axDose[1][1].legend([pTruthGamma, (pBCIGamma, pMeanGamma)], ['Truth','Reconstructed (95% BCI)'], loc='best')
        else:
            axDose[1][1].legend([(pBCIGamma, pMeanGamma)], ['Reconstructed (95% BCI)'], loc='best')

        # Plot the unfolded dose spectrum
        pBCIBetaDoseWB = axDose[2][0].fill_between(sorted(np.concatenate((self.ResponseBetaPlastic[1][0][1:],self.ResponseBetaPlastic[1][0][:-1]))), 
                                            np.repeat(binRecoDoseVal[0], 2), 
                                            np.repeat(binRecoDoseVal[12], 2),
                                            color='blue',
                                            alpha=0.5)

        pBCIBetaDoseSkin = axDose[2][0].fill_between(sorted(np.concatenate((self.ResponseBetaPlastic[1][0][1:],self.ResponseBetaPlastic[1][0][:-1]))), 
                                            np.repeat(binRecoDoseVal[1], 2), 
                                            np.repeat(binRecoDoseVal[13], 2),
                                            color='green',
                                            alpha=0.3)
        
        pBCIBetaDoseEye = axDose[2][0].fill_between(sorted(np.concatenate((self.ResponseBetaPlastic[1][0][1:],self.ResponseBetaPlastic[1][0][:-1]))), 
                                            np.repeat(binRecoDoseVal[2], 2), 
                                            np.repeat(binRecoDoseVal[14], 2),
                                            color='orange',
                                            alpha=0.3)
    
        pBCIGammaDoseWB = axDose[2][1].fill_between(sorted(np.concatenate((self.ResponseGammaLaBr3[1][0][1:],self.ResponseGammaLaBr3[1][0][:-1]))), 
                                            np.repeat(binRecoDoseVal[3], 2), 
                                            np.repeat(binRecoDoseVal[15], 2),
                                            color='blue',
                                            alpha=0.5)

        pBCIGammaDoseSkin = axDose[2][1].fill_between(sorted(np.concatenate((self.ResponseGammaLaBr3[1][0][1:],self.ResponseGammaLaBr3[1][0][:-1]))), 
                                            np.repeat(binRecoDoseVal[4], 2), 
                                            np.repeat(binRecoDoseVal[16], 2),
                                            color='green',
                                            alpha=0.4)
        
        pBCIGammaDoseEye = axDose[2][1].fill_between(sorted(np.concatenate((self.ResponseGammaLaBr3[1][0][1:],self.ResponseGammaLaBr3[1][0][:-1]))), 
                                            np.repeat(binRecoDoseVal[5], 2), 
                                            np.repeat(binRecoDoseVal[17], 2),
                                            color='orange',
                                            alpha=0.3)

        pMeanBetaDoseWB, = axDose[2][0].plot(sorted(np.concatenate((self.ResponseBetaPlastic[1][0][1:],self.ResponseBetaPlastic[1][0][:-1]))), 
                                        np.repeat(binRecoDoseVal[6], 2),
                                        lw=1.25, 
                                        color='blue', 
                                        linestyle="-", 
                                        drawstyle='steps')
        
        pMeanBetaDoseSkin, = axDose[2][0].plot(sorted(np.concatenate((self.ResponseBetaPlastic[1][0][1:],self.ResponseBetaPlastic[1][0][:-1]))), 
                                        np.repeat(binRecoDoseVal[7], 2),
                                        lw=1.25, 
                                        color='green', 
                                        linestyle="-", 
                                        drawstyle='steps')
        
        pMeanBetaDoseEye, = axDose[2][0].plot(sorted(np.concatenate((self.ResponseBetaPlastic[1][0][1:],self.ResponseBetaPlastic[1][0][:-1]))), 
                                        np.repeat(binRecoDoseVal[8], 2),
                                        lw=1.25, 
                                        color='orange', 
                                        linestyle="-", 
                                        drawstyle='steps')

        pMeanGammaDoseWB, = axDose[2][1].plot(sorted(np.concatenate((self.ResponseGammaLaBr3[1][1][1:],self.ResponseGammaLaBr3[1][1][:-1]))), 
                                        np.repeat(binRecoDoseVal[9], 2),
                                        lw=1.25, 
                                        color='blue', 
                                        linestyle="-", 
                                        drawstyle='steps')
        
        pMeanGammaDoseSkin, = axDose[2][1].plot(sorted(np.concatenate((self.ResponseGammaLaBr3[1][1][1:],self.ResponseGammaLaBr3[1][1][:-1]))), 
                                        np.repeat(binRecoDoseVal[10], 2),
                                        lw=1.25, 
                                        color='green', 
                                        linestyle="-", 
                                        drawstyle='steps')
        
        pMeanGammaDoseEye, = axDose[2][1].plot(sorted(np.concatenate((self.ResponseGammaLaBr3[1][1][1:],self.ResponseGammaLaBr3[1][1][:-1]))), 
                                        np.repeat(binRecoDoseVal[11], 2),
                                        lw=1.25, 
                                        color='orange', 
                                        linestyle="-", 
                                        drawstyle='steps')

        # Plot Statistics
        tblStats1 = axDose[2][0].table( cellText = (
                                        ('Body',
                                        '{:0.2E} mRem/hr'.format(np.sum(binTruthDoseVal[0])) if self.TruthBeta is not None else 'Unknown',
                                        '{:0.2E} ({:0.2E} to {:0.2E}) mRem/hr'.format(np.sum(binRecoDoseVal[6]), np.sum(binRecoDoseVal[0]), np.sum(binRecoDoseVal[12]))),
                                        ('Skin',
                                        '{:0.2E} mRad/hr'.format(np.sum(binTruthDoseVal[1])) if self.TruthGamma is not None else 'Unknown',
                                        '{:0.2E} ({:0.2E} to {:0.2E}) mRad/hr'.format(np.sum(binRecoDoseVal[7]), np.sum(binRecoDoseVal[1]), np.sum(binRecoDoseVal[13]))),
                                        ('Eye',
                                        '{:0.2E} mRad/hr'.format(np.sum(binTruthDoseVal[2])) if self.TruthBeta is not None else 'Unknown',
                                        '{:0.2E} ({:0.2E} to {:0.2E}) mRad/hr'.format(np.sum(binRecoDoseVal[8]), np.sum(binRecoDoseVal[2]), np.sum(binRecoDoseVal[14])))),
                                        cellLoc = 'center',
                                        colLabels = ['Organ', 'True Dose', 'Estimated Dose (95% BCI)'],
                                        colLoc = 'center',
                                        loc = 'bottom',
                                        bbox = [0, -0.57, 1, .35])  
        
        tblStats2 = axDose[2][1].table( cellText = (
                                        ('Body',
                                        '{:0.2E} mRem/hr'.format(np.sum(binTruthDoseVal[3])) if self.TruthGamma is not None else 'Unknown',
                                        '{:0.2E} ({:0.2E} to {:0.2E}) mRem/hr'.format(np.sum(binRecoDoseVal[9]), np.sum(binRecoDoseVal[3]), np.sum(binRecoDoseVal[15]))),
                                        ('Skin',
                                        '{:0.2E} mRad/hr'.format(np.sum(binTruthDoseVal[4])) if self.TruthGamma is not None else 'Unknown',
                                        '{:0.2E} ({:0.2E} to {:0.2E}) mRad/hr'.format(np.sum(binRecoDoseVal[10]), np.sum(binRecoDoseVal[4]), np.sum(binRecoDoseVal[16]))),
                                        ('Eye',
                                        '{:0.2E} mRad/hr'.format(np.sum(binTruthDoseVal[5])) if self.TruthGamma is not None else 'Unknown',
                                        '{:0.2E} ({:0.2E} to {:0.2E}) mRad/hr'.format(np.sum(binRecoDoseVal[11]), np.sum(binRecoDoseVal[5]), np.sum(binRecoDoseVal[17])))),
                                        cellLoc = 'center',
                                        colLabels = ['Organ', 'True Dose', 'Estimated Dose (95% BCI)'],
                                        colLoc = 'center',
                                        loc = 'bottom',
                                        bbox = [0, -0.57, 1, .35])
        
        # Figure Properties
        dnrFluence = maxY/minY      # Limit the dynamic range of the dose spectrum to the same as the fluence spectrum
        maxY = np.max(binRecoDoseVal[np.isfinite(binRecoDoseVal)])
        minY = np.min(binRecoDoseVal[binRecoDoseVal >= maxY/dnrFluence])

        axDose[2][0].set_title('Reconstructed Beta-ray Dose Spectrum')
        axDose[2][0].set_xlabel('True Energy (keV)')
        axDose[2][0].set_ylabel('Dose Rate (mRem/hr or mRad/hr)')
        axDose[2][0].set_xscale('log')
        axDose[2][0].set_yscale('log')
        axDose[2][0].set_xlim(min(self.ResponseBetaLaBr3[1][0]),max(self.ResponseBetaLaBr3[1][0]))
        axDose[2][0].set_ylim(np.power(10, np.ceil(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        axDose[2][0].legend([(pBCIBetaDoseWB, pMeanBetaDoseWB), (pBCIBetaDoseSkin, pMeanBetaDoseSkin), (pBCIBetaDoseEye, pMeanBetaDoseEye)],
                                ['Body (95% BCI)', 'Skin (95% BCI)', 'Eye (95% BCI)'],
                                loc='best')

        axDose[2][1].set_title('Reconstructed Gamma-ray Dose Spectrum')
        axDose[2][1].set_xlabel('True Energy (keV)')
        axDose[2][1].set_ylabel('Dose Rate (mRem/hr or mRad/hr)')
        axDose[2][1].set_xscale('log')
        axDose[2][1].set_yscale('log')
        axDose[2][1].set_xlim(min(self.ResponseGammaLaBr3[1][0]),max(self.ResponseGammaLaBr3[1][0]))
        axDose[2][1].set_ylim(np.power(10, np.ceil(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        axDose[2][1].legend([(pBCIGammaDoseWB, pMeanGammaDoseWB), (pBCIGammaDoseSkin, pMeanGammaDoseSkin), (pBCIGammaDoseEye, pMeanGammaDoseEye)],
                                ['Body (95% BCI)', 'Skin (95% BCI)','Eye (95% BCI)'],
                                loc='best')
        
         # Fine-tune figure 
        figDose.tight_layout()

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Unfolded plot saved to: ' + fName

        # Show the figure
        plt.close(figDose)

    def asMat(self, x):
        '''
        Transform an array of doubles into a Theano-type array so that it can be used in the model
        '''
        return np.asarray(x,dtype=theano.config.floatX)

    def buildModel(self, DataPlastic=None, DataLaBr3=None, TruthBeta=None, TruthGamma=None, BackgroundLaBr3=None):
        '''
        Build a multidimensional inference model
        '''
        # Check the input instance type for each data file
        if DataPlastic:
            if isinstance(DataPlastic, ROOT.TH1): 
                self.DataPlastic = np.asarray(hist2array(DataPlastic, include_overflow=False, copy=True, return_edges=True))
                self.DataPlastic[1] = np.round(self.DataPlastic[1], 5)
                self.DataPlastic[0] = np.delete(self.DataPlastic[0], np.where(self.DataPlastic[1][0] < self.ThresholdEnergy))
                self.DataPlastic[1] = np.delete(self.DataPlastic[1], np.where(self.DataPlastic[1][0] < self.ThresholdEnergy))
            else:
                raise TypeError("Data histogram from the Plastic detector must be of type ROOT.TH1")
        else:
            raise TypeError("Missing data histogram from the Plastic detector")
        
        if DataLaBr3:
            if isinstance(DataLaBr3, ROOT.TH1): 
                self.DataLaBr3 = np.asarray(hist2array(DataLaBr3, include_overflow=False, copy=True, return_edges=True))
                self.DataLaBr3[1] = np.round(self.DataLaBr3[1], 5)
                self.DataLaBr3[0] = np.delete(self.DataLaBr3[0], np.where(self.DataLaBr3[1][0] < self.ThresholdEnergy))
                self.DataLaBr3[1] = np.delete(self.DataLaBr3[1], np.where(self.DataLaBr3[1][0] < self.ThresholdEnergy))
            else:
                raise TypeError("Data histogram from the LaBr3 detector must be of type ROOT.TH1")
        else:
            raise TypeError("Missing data histogram from the LaBr3 detector")

        if TruthBeta:
            if isinstance(TruthBeta, ROOT.TH1): 
                self.TruthBeta = np.asarray(hist2array(TruthBeta, include_overflow=False, copy=True, return_edges=True))
                self.TruthBeta[1] = np.round(self.TruthBeta[1], 5)
                self.TruthBeta[0] = np.delete(self.TruthBeta[0], np.where(self.TruthBeta[1][0] < self.ThresholdEnergy))
                self.TruthBeta[1] = np.delete(self.TruthBeta[1], np.where(self.TruthBeta[1][0] < self.ThresholdEnergy))
            else:
                raise TypeError("Truth histogram for the Beta spectrum must be of type ROOT.TH1")
        else:
            self.TruthBeta = None 

        if TruthGamma:
            if isinstance(TruthGamma, ROOT.TH1): 
                self.TruthGamma = np.asarray(hist2array(TruthGamma, include_overflow=False, copy=True, return_edges=True))
                self.TruthGamma[1] = np.round(self.TruthGamma[1], 5)
                self.TruthGamma[0] = np.delete(self.TruthGamma[0], np.where(self.TruthGamma[1][0] < self.ThresholdEnergy))
                self.TruthGamma[1] = np.delete(self.TruthGamma[1], np.where(self.TruthGamma[1][0] < self.ThresholdEnergy))
            else:
                raise TTypeError("Truth histogram for the Gamma spectrum must be of type ROOT.TH1")
        else:
            self.TruthGamma = None 

        # Check if background is available and subtract from data
        if BackgroundLaBr3:
            if isinstance(BackgroundLaBr3, ROOT.TH1): 
                self.BackgroundLaBr3 = np.asarray(hist2array(BackgroundLaBr3, include_overflow=False, copy=True, return_edges=True))
                self.BackgroundLaBr3[1] = np.round(self.BackgroundLaBr3[1], 5)
                self.BackgroundLaBr3[0] = np.delete(self.BackgroundLaBr3[0], np.where(self.BackgroundLaBr3[1][0] < self.ThresholdEnergy))
                self.BackgroundLaBr3[1] = np.delete(self.BackgroundLaBr3[1], np.where(self.BackgroundLaBr3[1][0] < self.ThresholdEnergy))

                self.DataLaBr3[0] -= self.BackgroundLaBr3[0]
                self.DataLaBr3[0][self.DataLaBr3[0] < 0] = 0.
            else:
                raise TypeError("Background histogram from the LaBr3 detector must be of type ROOT.TH1")

        # Build the model
        with pm.Model() as self.model:

            # Define the upper and lower bounds for the priors
            GFBetaPlastic = np.sum(self.ResponseBetaPlastic[0], axis=1)
            GFGammaPlastic = np.sum(self.ResponseGammaPlastic[0], axis=1)
            GFBetaLaBr3 = np.sum(self.ResponseBetaLaBr3[0], axis=1)
            GFGammaLaBr3 = np.sum(self.ResponseGammaLaBr3[0], axis=1)

            SFBeta = GFBetaPlastic/np.power(GFBetaPlastic + GFGammaPlastic, 2)
            SFBeta[np.isclose(SFBeta, 0)] = np.finfo(np.float64).eps
            SFGamma = GFGammaLaBr3/np.power(GFBetaLaBr3 + GFGammaLaBr3, 2)
            SFGamma[np.isclose(SFGamma, 0)] = np.finfo(np.float64).eps

            nCountsPlastic = self.DataPlastic[0]
            nCountsLaBr3 = self.DataLaBr3[0]

            lbPhiBeta = np.zeros(self.ResponseBetaPlastic[1][0].size-1)
            ubPhiBeta = 10*np.ones(self.ResponseBetaPlastic[1][0].size-1)*np.sum(nCountsPlastic)*SFBeta
            lbPhiGamma = np.zeros(self.ResponseGammaLaBr3[1][0].size-1)
            ubPhiGamma = 10*np.ones(self.ResponseGammaLaBr3[1][0].size-1)*np.sum(nCountsLaBr3)*SFGamma

            ubPhiBeta[np.isclose(ubPhiBeta, 0)] = 1E-15
            ubPhiGamma[np.isclose(ubPhiGamma, 0)] = 1E-15
            
            # Define the alpha
            self.var_alpha = theano.shared(value = 10., borrow = False)

            # Define the prior
            self.prior = 'Uniform'

            # Define the prior probability densities
            if self.prior == 'Uniform':
                self.PhiBeta = pm.Uniform('PhiBeta', lower = lbPhiBeta, upper = ubPhiBeta, shape = (self.ResponseBetaPlastic[1][0].size-1))
                self.PhiGamma = pm.Uniform('PhiGamma', lower = lbPhiGamma, upper = ubPhiGamma, shape = (self.ResponseGammaPlastic[1][0].size-1))
            elif self.prior == 'Gaussian':
                self.PhiBeta = pm.DensityDist('PhiBeta', logp = lambda val: -self.var_alpha*0.5*theano.tensor.sqr((val - ubPhiBeta)/ubPhiBeta).sum(), shape = (self.ResponseBetaPlastic[1][0].size-1))            
                self.PhiGamma = pm.DensityDist('PhiGamma', logp = lambda val: -self.var_alpha*0.5*theano.tensor.sqr((val - ubPhiGamma)/ubPhiGamma).sum(), shape = (self.ResponseGammaPlastic[1][0].size-1))

            # Define the generative models
            self.MPlastic = theano.tensor.dot(theano.shared(self.asMat(self.ResponseBetaPlastic[0].T)), self.PhiBeta) + theano.tensor.dot(theano.shared(self.asMat(self.ResponseGammaPlastic[0].T)), self.PhiGamma)
            self.MLaBr3 = theano.tensor.dot(theano.shared(self.asMat(self.ResponseBetaLaBr3[0].T)), self.PhiBeta) + theano.tensor.dot(theano.shared(self.asMat(self.ResponseGammaLaBr3[0].T)), self.PhiGamma)
            
            # Define the posterior probability function (PPF)
            self.PPFPlastic = pm.Poisson('PPF_Plastic', 
                                    mu = self.MPlastic,
                                    observed = theano.shared(self.DataPlastic[0], borrow = False), 
                                    shape = (self.DataPlastic[0].size, 1))

            self.PPFLaBr3 = pm.Poisson('PPF_LaBr3', 
                                    mu = self.MLaBr3,
                                    observed = theano.shared(self.DataLaBr3[0], borrow = False), 
                                    shape = (self.DataLaBr3[0].size, 1))
    
    def plotCornerPDF(self, fName="CornerPosteriorPDF.pdf"):
        import corner 
        figCornerPDF = corner.corner(self.trace['PhiBeta'][:,46:50], show_titles=True, title_kwargs={"fontsize": 12})
        
        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Corner plot saved to: ' + fName

        # Show the figure
        plt.close(figCornerPDF)

    def sampleMH(self, N = 100000, B = 100000):
        '''
        Function to sample the posterior distribution using a Markov Chain Monte Carlo (MCMC) and the
        Metropolis Hastings algorithm in PyMC3.
        ''' 
        self.Samples = N
        self.Burn = B
        with self.model:
            # Select the Posterior sampling algorithm
            print 'Sampling the posterior using Metropolis ...'
            step = pm.Metropolis()
            start = pm.find_MAP(model = self.model)
            
            self.trace = pm.sample(self.Samples,
                                   tune = self.Burn,
                                   start = start,
                                   step=step)
            

            # Print a summary of the MCMC trace      
            pm.summary(self.trace)
    
    def sampleADVI(self, iterations = 1000000, samples = 100000):
        '''
        Function to sample the posterior distribution using the ADVI variational inference algorithm.
        The outputs from this algorithm can be used to update the prior estimate before a more general
        MCMC algorithm is used.

        PARAMETERS:
        ----------
        iterations: int
            The maximum number of variational inference iterations to run. 

        '''
        self.Iterations = iterations
        self.Samples = samples
        
        with self.model:
            from pymc3.variational.callbacks import CheckParametersConvergence

            print '\nInfering the posterior distribution using ADVI'
            self.approxADVI = pm.fit(n=self.Iterations, method='advi', callbacks=[CheckParametersConvergence(every=1000, diff='absolute', tolerance = 5E-2)])

            # Draw sample from ADVI fit
            self.trace = self.approxADVI.sample(draws=self.Samples)

            # Print a summary of the MCMC trace      
            pm.summary(self.trace)

    def sampleNUTS(self, N = 100000, B = 20000):
        '''
        Function to sample the posterior distribution using a Markov Chain Monte Carlo (MCMC) and the
        No-U-Turn Sampling (NUTS) algorithm in PyMC3.
        ''' 
        self.Samples = N
        self.Burn = B
        with self.model:
            # Sample
            self.trace = pm.sample(self.Samples, tune = self.Burn)

            # Print a summary of the MCMC trace      
            pm.summary(self.trace)
    
    def sampleHMC(self, N = 10000, B = 10000):
        '''
        Function to sample the posterior distribution using a Markov Chain Monte Carlo (MCMC) and the
        Hamiltonian Monte Carlo (HMC) sampling algorithm in PyMC3.
        ''' 
        self.Samples = N
        self.Burn = B
        with self.model:
            mu, sds, elbo = pm.variational.advi(n=2000000)

            # Select the Posterior sampling algorithm
            print 'Sampling the posterior using HMC ...'
            step = pm.HamiltonianMC(scaling=np.power(self.model.dict_to_array(sds), 2), is_cov=True)

            # Sample
            self.trace = pm.sample(self.Samples,
                                   tune = self.Burn,
                                   start = mu,
                                   step=step)
            
            # Print a summary of the MCMC trace      
            pm.summary(self.trace)

    def sampleSMC(self, N = 10000, n_chains = 100, cores = 1):
        '''
        Function to sample the posterior distribution using a Markov Chain Monte Carlo (MCMC) and the
        Sequential Monte Carlo (SMC) sampling algorithm in PyMC3.

        URL: https://github.com/pymc-devs/pymc3/blob/master/docs/source/notebooks/SMC2_gaussians.ipynb
        
        Description:
        Sampling from n-dimensional distributions with multiple peaks with a standard Metropolis-Hastings algorithm can be difficult, 
        if not impossible, as the Markov chain often gets stuck in either of the minima. SMC is a way to overcome this problem, 
        or at least to ameliorate it. 
        ''' 
        with self.model:
            print 'Sampling the posterior using Sequential Monte Carlo (SMC)'
            test_folder = mkdtemp(prefix='SMC_TEST')   
            #start = pm.find_MAP(model = self.model)
            self.trace = pm.smc.sample_smc(samples=100000,
                                            n_chains=10000,
                                            n_jobs=1,
                                            progressbar=True,
                                            #stage = 22,
                                            #start=start,
                                            #step=pm.SMC(),
                                            model=self.model, 
                                            homepath=test_folder)

            # Print a summary of the MCMC trace  
            pm.summary(self.trace) 
            #pm.traceplot(self.trace)

# ROOT file context manager
class ROOTFile(object):

    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.file = ROOT.TFile.Open(self.filename, 'read')
        return self.file

    def __exit__(self, exception_type, exception_value, traceback):
        self.file.Close()

# Command Line Parser
parser = argparse.ArgumentParser()
parser.add_argument('ResponsePlastic', type=ROOTFile, help='ROOT file containing the response matrix for the Plastic detector')
parser.add_argument('ResponseLaBr3', type=ROOTFile, help='ROOT file containing the response matrix for the LaBr3 detector')
parser.add_argument('SpectrumPlastic', type=ROOTFile, help='ROOT file containing the measured spectrum from the Plastic detector')
parser.add_argument('SpectrumLaBr3', type=ROOTFile, help='ROOT file containing the measured spectrum from the LaBr3 detector')
parser.add_argument('--BackgroundLaBr3', type=ROOTFile, help='ROOT file containing the measured background spectrum from the LaBr3 detector')
parser.add_argument('--OutputFilename', help='Filename to be used for output')
parser.add_argument('--ThresholdEnergy', help='Threshold energy, in keV, above which channels will be included in the unfolding')
parser.add_argument('--SamplingAlgorithm', 
                    choices=['MH', 'ADVI', 'NUTS', 'HMC', 'SMC'], 
                    default='ADVI', 
                    help='Threshold energy, in keV, above which channels will be included in the unfolding')
args = parser.parse_args()

# Detector Response Matrices
fResponsePlastic = args.ResponsePlastic.__enter__()
fResponseLaBr3 = args.ResponseLaBr3.__enter__()

# Measured Spectrum Files
fDataPlastic = args.SpectrumPlastic.__enter__()
fDataLaBr3 = args.SpectrumLaBr3.__enter__()
fBackgroundLaBr3 = args.BackgroundLaBr3.__enter__() if args.BackgroundLaBr3 else None

# Output filenames
fOutputFilename = args.OutputFilename if args.OutputFilename else ''

# Dose Coefficients
fDoseCoeffGamma  = './Dose Coefficients/ICRP116_Photon_DoseConversionCoefficients.xlsx'
fDoseCoeffBeta  = './Dose Coefficients/ICRP116_Electron_DoseConversionCoefficients.xlsx'

# Initiate the class
myMBSD = PyMBSD(MigBetaPlastic = fResponsePlastic.Get('Energy Migration Matrix (Electron)'),
                MigGammaPlastic = fResponsePlastic.Get('Energy Migration Matrix (Gamma)'), 
                MigBetaLaBr3 = fResponseLaBr3.Get('Energy Migration Matrix (Electron)'),
                MigGammaLaBr3 = fResponseLaBr3.Get('Energy Migration Matrix (Gamma)'),
                SourceBetaPlastic = fResponsePlastic.Get('Source Spectrum (Electron)'),
                SourceGammaPlastic = fResponsePlastic.Get('Source Spectrum (Gamma)'),
                SourceBetaLaBr3 = fResponseLaBr3.Get('Source Spectrum (Electron)'),
                SourceGammaLaBr3 = fResponseLaBr3.Get('Source Spectrum (Gamma)'),
                ThresholdEnergy = args.ThresholdEnergy if args.ThresholdEnergy else 0.)

# Plot the response matrices
myMBSD.plotResponse(fName = 'ResponseMatrix.jpg')

# Load the dose coefficients (NOTE: Using ICRP 116)
myMBSD.loadDoseCoeffGamma(fName = fDoseCoeffGamma)
myMBSD.loadDoseCoeffBeta(fName = fDoseCoeffBeta)

# Build the model
myMBSD.buildModel(DataPlastic = fDataPlastic.Get('Detector Measured Spectrum'), 
                  DataLaBr3 = fDataLaBr3.Get('Detector Measured Spectrum'),
                  TruthBeta = fDataPlastic.Get('Source Spectrum (Electron)'),
                  TruthGamma = fDataPlastic.Get('Source Spectrum (Gamma)'),
                  BackgroundLaBr3 = fBackgroundLaBr3.Get('Detector Measured Spectrum') if fBackgroundLaBr3 else None)

# Run Inference
if args.SamplingAlgorithm == 'MH':
    # Run Metropolis Hastings sampling algorithm
    myMBSD.sampleMH(N=100000,B=100000)
    myMBSD.plotUnfoldedFluenceSpectrum(fName = fOutputFilename + '_Fluence_MH.pdf')
    myMBSD.plotUnfoldedDoseSpectrum(fName = fOutputFilename + '_Dose_MH.pdf')
    myMBSD.plotFoldedMeasuredSpectrum(fName = fOutputFilename + '_Folded_MH.pdf')
elif args.SamplingAlgorithm == 'ADVI':
    # Run Variational Inference sampling algorithm
    myMBSD.sampleADVI()
    myMBSD.plotUnfoldedFluenceSpectrum(fName = fOutputFilename + '_Fluence_ADVI.pdf')
    myMBSD.plotUnfoldedDoseSpectrum(fName = fOutputFilename + '_Dose_ADVI.pdf')
    myMBSD.plotFoldedMeasuredSpectrum(fName = fOutputFilename + '_Folded_ADVI.pdf')
elif args.SamplingAlgorithm == 'NUTS':
    # Run the No-U-Turn sampling algorithm
    myMBSD.sampleNUTS(100000,100000)
    myMBSD.plotUnfoldedFluenceSpectrum(fName = fOutputFilename + '_Fluence_NUTS.pdf')
    myMBSD.plotUnfoldedDoseSpectrum(fName = fOutputFilename + '_Dose_NUTS.pdf')
    myMBSD.plotFoldedMeasuredSpectrum(fName = fOutputFilename + '_Folded_NUTS.pdf')
elif args.SamplingAlgorithm == 'HMC':
    # Run the Hamiltonian Monte Carlo sampling algorithm
    myMBSD.sampleHMC()
    myMBSD.plotUnfoldedFluenceSpectrum(fName = fOutputFilename + '_Fluence_HMC.pdf')
    myMBSD.plotUnfoldedDoseSpectrum(fName = fOutputFilename + '_Dose_HMC.pdf')
elif args.SamplingAlgorithm == 'SMC':
    # Run the Sequential Monte Carlo sampling algorithm
    myMBSD.sampleSMC()
    myMBSD.plotUnfoldedFluenceSpectrum(fName = fOutputFilename + '_Fluence_SMC.pdf')
    myMBSD.plotUnfoldedDoseSpectrum(fName = fOutputFilename + '_Dose_SMC.pdf')