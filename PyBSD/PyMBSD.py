import numpy as np
import pymc3 as pm
import ROOT 

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
from mpl_toolkits.axes_grid1 import AxesGrid


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
                MigBetaPIPS, MigGammaPIPS, MigBetaLaBr3, MigGammaLaBr3,
                SourceBetaPIPS, SourceGammaPIPS, SourceBetaLaBr3,SourceGammaLaBr3):
        '''
        Object initialization function
        '''

        # Check the input instance type as follows:
        if not isinstance(MigBetaPIPS, ROOT.TH2): raise TypeError("Beta migration matrix for PIPS detector must be of type ROOT.TH2")
        if not isinstance(MigGammaPIPS, ROOT.TH2): raise TypeError("Gamma migration matrix for PIPS detector must be of type ROOT.TH2")
        if not isinstance(MigBetaLaBr3, ROOT.TH2): raise TypeError("Beta migration matrix for LaBr3 detector must be of type ROOT.TH2")
        if not isinstance(MigGammaLaBr3, ROOT.TH2): raise TypeError("Gamma migration matrix for LaBr3 detector must be of type ROOT.TH2")
        if not isinstance(SourceBetaPIPS, ROOT.TH1): raise TypeError("Beta source spectrum for the PIPS detector must be of type ROOT.TH1")
        if not isinstance(SourceGammaPIPS, ROOT.TH1): raise TypeError("Gamma source spectrum for the PIPS detector must be of type ROOT.TH1")
        if not isinstance(SourceBetaLaBr3, ROOT.TH1): raise TypeError("Beta source spectrum for the LaBr3 detector must be of type ROOT.TH1")
        if not isinstance(SourceGammaLaBr3, ROOT.TH1): raise TypeError("Gamma source spectrum for the LaBr3 detector must be of type ROOT.TH1")

        # Copy the inputs to the object
        self.MigBetaPIPS = hist2array(MigBetaPIPS, include_overflow=False, copy=True, return_edges=True)
        self.MigGammaPIPS = hist2array(MigGammaPIPS, include_overflow=False, copy=True, return_edges=True)
        self.MigBetaLaBr3 = hist2array(MigBetaLaBr3, include_overflow=False, copy=True, return_edges=True)
        self.MigGammaLaBr3 = hist2array(MigGammaLaBr3, include_overflow=False, copy=True, return_edges=True)
        self.SourceBetaPIPS = hist2array(SourceBetaPIPS, include_overflow=False, copy=True, return_edges=True)
        self.SourceGammaPIPS = hist2array(SourceGammaPIPS, include_overflow=False, copy=True, return_edges=True)
        self.SourceBetaLaBr3 = hist2array(SourceBetaLaBr3, include_overflow=False, copy=True, return_edges=True)
        self.SourceGammaLaBr3 = hist2array(SourceGammaLaBr3, include_overflow=False, copy=True, return_edges=True)

        # Calculate the response matrix (aka. conditional probability) using Eq. 5 from the Choudalakis paper
        # Response[i,j] = P(d = j|t = i) = P(t = i, d = j)/P(t = i)
        # Response[j|i] = M[d = j, t = i] / Truth[i]
        self.ResponseBetaPIPS = copy.deepcopy(self.MigBetaPIPS)
        for i in np.arange(self.ResponseBetaPIPS[1][0].size - 1):
            for j in np.arange(self.ResponseBetaPIPS[1][1].size - 1):
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Normalize with Truth
                    self.ResponseBetaPIPS[0][i,j]=(self.ResponseBetaPIPS[0][i,j]/self.SourceBetaPIPS[0][i] if np.nonzero(self.SourceBetaPIPS[0][i]) else 0.)
        
        self.ResponseGammaPIPS = copy.deepcopy(self.MigGammaPIPS)
        for i in np.arange(self.ResponseGammaPIPS[1][0].size - 1):
            for j in np.arange(self.ResponseGammaPIPS[1][1].size -1):
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Normalize with Truth
                    self.ResponseGammaPIPS[0][i,j]=(self.ResponseGammaPIPS[0][i,j]/self.SourceGammaPIPS[0][i] if np.nonzero(self.SourceGammaPIPS[0][i]) else 0.)

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
        rLimLow = 1E-3
        rLimUp = 1E2

        # Plot the response matrices
        X, Y = np.meshgrid(self.ResponseBetaPIPS[1][0], self.ResponseBetaPIPS[1][1])
        H0 = axResp[0].pcolormesh(X, Y, self.ResponseBetaPIPS[0].T, norm = colors.LogNorm(), cmap = cmap, linewidth = 0, rasterized = True) 

        X, Y = np.meshgrid(self.ResponseGammaPIPS[1][0], self.ResponseGammaPIPS[1][1])
        H1 = axResp[1].pcolormesh(X, Y, self.ResponseGammaPIPS[0].T, norm = colors.LogNorm(), cmap = cmap, linewidth = 0, rasterized = True) 

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
        axResp[0].text(0.03,0.95, 'Canberra PD450-15-500AM PIPS', transform=axResp[0].transAxes, verticalalignment='top', color = 'white')

        axResp[1].set_xscale('log')
        axResp[1].set_yscale('log')
        axResp[1].text(0.03,0.95, 'Canberra PD450-15-500AM PIPS', transform=axResp[1].transAxes, verticalalignment='top', color = 'white')

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
        df_ICRP116_Photon_WholeBody = pandas.read_excel(fName, sheetname = 'Effective Dose (Whole Body)')
        df_ICRP116_Photon_FemaleSkin = pandas.read_excel(fName, sheetname = 'Absorbed Dose (Female Skin)')
        df_ICRP116_Photon_MaleSkin = pandas.read_excel(fName, sheetname = 'Absorbed Dose (Male Skin)')
        df_ICRP116_Photon_EyeLens = pandas.read_excel(fName, sheetname = 'Absorbed Dose (Lens of Eye)')

        # Interpolate the coefficients into the true log energy bins
        def logInterpCoeff(coeffBins, coeffX, coeffY):
            midBin = [np.log10(coeffBins[i]*coeffBins[i + 1])/2 for i in range(0, len(coeffBins)-1)]
            return np.nan_to_num(np.power(10, interpolate.interp1d(np.log10(coeffX), np.log10(coeffY), kind='linear')(midBin)))

        # Scaling factor to convert the magnitude of the dose coefficients
        coeffScalingFactor = 1E-12*1E9 # pSv/pGy to nSv/nGy

        # Load the dose coefficients into self object
        # NOTE: Energy scaling from MeV to keV
        self.coeffGammaWB = np.array([logInterpCoeff(self.ResponseBetaPIPS[1][0], 
                                                     df_ICRP116_Photon_WholeBody['Energy (MeV)'].values*1E3, 
                                                     df_ICRP116_Photon_WholeBody['ISO (pSv cm2)'].values*coeffScalingFactor),
                                      self.ResponseBetaPIPS[1][0]])
        
        self.coeffGammaSkin = np.array([logInterpCoeff(self.ResponseBetaPIPS[1][0], 
                                                     df_ICRP116_Photon_MaleSkin['Energy (MeV)'].values*1E3, 
                                                     df_ICRP116_Photon_MaleSkin['ISO (pGy cm2)'].values*coeffScalingFactor),
                                      self.ResponseBetaPIPS[1][0]])
        
        self.coeffGammaEye = np.array([logInterpCoeff(self.ResponseBetaPIPS[1][0], 
                                                     df_ICRP116_Photon_EyeLens['Energy (MeV)'].values*1E3, 
                                                     df_ICRP116_Photon_EyeLens['ISO (pGy cm2)'].values*coeffScalingFactor),
                                      self.ResponseBetaPIPS[1][0]])
    
    def loadDoseCoeffBeta(self, fName=''):
        '''
        Function to load the ICRP 116 beta-ray dose coefficients for the whole body, skin, and lens of the eye.
        '''

        # Import coefficients from files
        df_ICRP116_Beta_WholeBody = pandas.read_excel(fName, sheetname = 'Effective Dose (Whole Body)')
        df_ICRP116_Beta_EyeLens = pandas.read_excel(fName, sheetname = 'Absorbed Dose (Lens of Eye)')

        # Interpolate the coefficients into the true log energy bins
        def logInterpCoeff(coeffBins, coeffX, coeffY):
            midBin = [np.log10(coeffBins[i]*coeffBins[i + 1])/2 for i in range(0, len(coeffBins)-1)]
            return np.nan_to_num(np.power(10, interpolate.interp1d(np.log10(coeffX), np.log10(coeffY), kind='linear')(midBin)))

        # Scaling factor to convert the magnitude of the dose coefficients
        coeffScalingFactor = 1E-12*1E9 # pSv/pGy to nSv/nGy

        # Load the dose coefficients into self object
        # NOTE: Energy scaling from MeV to keV
        self.coeffBetaWB = np.array([logInterpCoeff(self.ResponseBetaPIPS[1][0], 
                                                    df_ICRP116_Beta_WholeBody['Energy (MeV)'].values*1E3, 
                                                    df_ICRP116_Beta_WholeBody['ISO (pSv cm2)'].values*coeffScalingFactor),
                                      self.ResponseBetaPIPS[1][0]])
        
        self.coeffBetaEye = np.array([logInterpCoeff(self.ResponseBetaPIPS[1][0], 
                                                     df_ICRP116_Beta_EyeLens['Energy (MeV)'].values*1E3, 
                                                     df_ICRP116_Beta_EyeLens['ISO (pGy cm2)'].values*coeffScalingFactor),
                                      self.ResponseBetaPIPS[1][0]])

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
        figFluence, axFluence = plt.subplots(2,2, figsize=(fig_size[0]*2,fig_size[1]*1.5))
        
        # Plot the data spectrum
        axFluence[0][0].plot(sorted(np.concatenate((self.DataPIPS[1][0][:-1],self.DataPIPS[1][0][1:]))), 
                    np.repeat(self.DataPIPS[0], 2),
                    lw=1.25, 
                    color='black', 
                    linestyle="-",
                    drawstyle='steps')
        
        axFluence[0][1].plot(sorted(np.concatenate((self.DataLaBr3[1][0][:-1],self.DataLaBr3[1][0][1:]))),  
                    np.repeat(self.DataLaBr3[0], 2),
                    lw=1.25, 
                    color='black', 
                    linestyle="-",
                    drawstyle='steps')

        axFluence[0][0].set_title('Measured Spectrum from Canberra PD450-15-500AM PIPS')
        axFluence[0][0].set_xlabel('Measured Energy (keV)')
        axFluence[0][0].set_ylabel('Counts')
        axFluence[0][0].set_xlim(min(self.DataPIPS[1][0]),max(self.DataPIPS[1][0]))
        axFluence[0][0].set_ylim(1., np.power(10, np.ceil(np.log10(np.max(self.DataPIPS[0])))))
        axFluence[0][0].set_xscale('log')
        axFluence[0][0].set_yscale('log')

        axFluence[0][1].set_title('Measured Spectrum from Saint Gobain B380 LaBr3')
        axFluence[0][1].set_xlabel('Measured Energy (keV)')
        axFluence[0][1].set_ylabel('Counts')
        axFluence[0][1].set_xlim(min(self.DataLaBr3[1][0]),max(self.DataLaBr3[1][0]))
        axFluence[0][1].set_ylim(1., np.power(10, np.ceil(np.log10(np.max(self.DataLaBr3[0])))))
        axFluence[0][1].set_xscale('log')
        axFluence[0][1].set_yscale('log')

        # Plot the true fluence spectrum, if available.
        pTruthBeta, = axFluence[1][0].plot(sorted(np.concatenate((self.TruthBeta[1][0][1:],self.TruthBeta[1][0][:-1]))), 
                                        np.repeat(self.TruthBeta[0], 2),
                                        lw=1.25, 
                                        color='black', 
                                        linestyle="-", 
                                        drawstyle='steps')

        pTruthGamma, = axFluence[1][1].plot(sorted(np.concatenate((self.TruthGamma[1][0][1:],self.TruthGamma[1][0][:-1]))), 
                                        np.repeat(self.TruthGamma[0], 2),
                                        lw=1.25, 
                                        color='black', 
                                        linestyle="-", 
                                        drawstyle='steps')

        # Plot the unfolded spectrum
        pBCIBeta = axFluence[1][0].fill_between(sorted(np.concatenate((self.ResponseBetaPIPS[1][0][1:],self.ResponseBetaPIPS[1][0][:-1]))), 
                                            np.repeat(binRecoVal[0], 2), 
                                            np.repeat(binRecoVal[4], 2),
                                            color='red',
                                            alpha=0.4)
    
        pBCIGamma = axFluence[1][1].fill_between(sorted(np.concatenate((self.ResponseGammaLaBr3[1][0][1:],self.ResponseGammaLaBr3[1][0][:-1]))), 
                                            np.repeat(binRecoVal[1], 2), 
                                            np.repeat(binRecoVal[5], 2),
                                            color='red',
                                            alpha=0.4)

        pMeanBeta, = axFluence[1][0].plot(sorted(np.concatenate((self.ResponseBetaPIPS[1][0][1:],self.ResponseBetaPIPS[1][0][:-1]))), 
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
        axFluence[1][0].set_ylabel('Fluence (cm$^{-2}$)')
        axFluence[1][0].set_xscale('log')
        axFluence[1][0].set_yscale('log')
        axFluence[1][0].set_xlim(min(self.ResponseBetaPIPS[1][0]),max(self.ResponseBetaPIPS[1][0]))
        axFluence[1][0].set_ylim(np.power(10, np.floor(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        axFluence[1][0].legend([pTruthBeta, (pBCIBeta, pMeanBeta)], ['Truth','Reconstructed (95% BCI)'], loc='best')

        axFluence[1][1].set_xlabel('True Energy (keV)')
        axFluence[1][1].set_title('Reconstructed Gamma-ray Fluence Spectrum')
        axFluence[1][1].set_ylabel('Fluence (cm$^{-2}$)')
        axFluence[1][1].set_xscale('log')
        axFluence[1][1].set_yscale('log')
        axFluence[1][1].set_xlim(min(self.ResponseGammaLaBr3[1][0]),max(self.ResponseGammaLaBr3[1][0]))
        axFluence[1][1].set_ylim(np.power(10, np.floor(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        axFluence[1][1].legend([pTruthGamma, (pBCIGamma, pMeanGamma)], ['Truth','Reconstructed (95% BCI)'], loc='best')
        
        # Fine-tune figure 
        figFluence.tight_layout()

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Unfolded plot saved to: ' + fName

        # Show the figure
        plt.close(figFluence)

    def plotUnfoldedDoseSpectrum(self, fName='UnfoldedDoseSpectrum.pdf', plotTruth = False):
        '''
        Function to plot the reconstructed dose spectrum after performing multidimensional Bayesian unfolding
        NOTE: This function is to be used when multiple response matrices are used in the unfolding.
        '''

        # Calculate the true dose values
        binTruthDoseVal = np.array([self.TruthBeta[0]*self.coeffBetaWB[0],                  # Beta WB Dose Mean 
                                   self.TruthBeta[0]*self.coeffBetaEye[0],                  # Beta Eye Dose Mean
                                   self.TruthGamma[0]*self.coeffGammaWB[0],                 # Gamma WB Dose Mean
                                   self.TruthGamma[0]*self.coeffGammaSkin[0],               # Gamma Skin Dose Mean
                                   self.TruthGamma[0]*self.coeffGammaEye[0]])               # Gamma Eye Dose Mean

        # Calculate and plot the 95% Bayesian credible regions for the unfolded spectrum
        unfoldedBCIBeta = pm.stats.hpd(self.trace['PhiBeta'], alpha=0.05)
        unfoldedBCIGamma = pm.stats.hpd(self.trace['PhiGamma'], alpha=0.05)

        binRecoFluenceVal = np.array([unfoldedBCIBeta[:,0],                                 # Beta 2.5% HPD
                               unfoldedBCIGamma[:,0],                                       # Gamma 2.5% HPD
                               np.mean(self.trace['PhiBeta'],0),                            # Beta Mean
                               np.mean(self.trace['PhiGamma'],0),                           # Gamma Mean
                               unfoldedBCIBeta[:,1],                                        # Beta 97.5% HPD
                               unfoldedBCIGamma[:,1]])                                      # Gamma 97.5% HPD
        
        binRecoDoseVal = np.array([unfoldedBCIBeta[:,0]*self.coeffBetaWB[0],                # Beta WB Dose 2.5% HPD
                                   unfoldedBCIBeta[:,0]*self.coeffBetaEye[0],               # Beta Eye Dose 2.5% HPD
                                   unfoldedBCIGamma[:,0]*self.coeffGammaWB[0],              # Gamma WB Dose 2.5% HPD
                                   unfoldedBCIGamma[:,0]*self.coeffGammaSkin[0],            # Gamma Skin Dose 2.5% HPD
                                   unfoldedBCIGamma[:,0]*self.coeffGammaEye[0],             # Gamma Eye Dose 2.5% HPD
                                   np.mean(self.trace['PhiBeta'],0)*self.coeffBetaWB[0],    # Beta WB Dose Mean 
                                   np.mean(self.trace['PhiBeta'],0)*self.coeffBetaEye[0],   # Beta Eye Dose Mean
                                   np.mean(self.trace['PhiGamma'],0)*self.coeffGammaWB[0],  # Gamma WB Dose Mean
                                   np.mean(self.trace['PhiGamma'],0)*self.coeffGammaSkin[0],# Gamma Skin Dose Mean
                                   np.mean(self.trace['PhiGamma'],0)*self.coeffGammaEye[0], # Gamma Eye Dose Mean
                                   unfoldedBCIBeta[:,1]*self.coeffBetaWB[0],                # Beta WB Dose 97.5% HPD
                                   unfoldedBCIBeta[:,1]*self.coeffBetaEye[0],               # Beta Eye Dose 97.5% HPD
                                   unfoldedBCIGamma[:,1]*self.coeffGammaWB[0],              # Gamma WB Dose 97.5% HPD
                                   unfoldedBCIGamma[:,1]*self.coeffGammaSkin[0],            # Gamma Skin Dose 97.5% HPD
                                   unfoldedBCIGamma[:,1]*self.coeffGammaEye[0]])            # Gamma Eye Dose 97.5% HPD

        # Create a figure to plot the spectrum
        figDose, axDose = plt.subplots(2,2, figsize=(fig_size[0]*2,fig_size[1]*1.5))

         # Plot the true fluence spectrum, if available.
        pTruthBeta, = axDose[0][0].plot(sorted(np.concatenate((self.ResponseBetaPIPS[1][0][1:],self.ResponseBetaPIPS[1][0][:-1]))), 
                                        np.repeat(self.TruthBeta[0], 2),
                                        lw=1.25, 
                                        color='black', 
                                        linestyle="-", 
                                        drawstyle='steps')

        pTruthGamma, = axDose[0][1].plot(sorted(np.concatenate((self.ResponseGammaLaBr3[1][0][1:],self.ResponseGammaLaBr3[1][0][:-1]))), 
                                        np.repeat(self.TruthGamma[0], 2),
                                        lw=1.25, 
                                        color='black', 
                                        linestyle="-", 
                                        drawstyle='steps')

        # Plot the unfolded spectrum
        pBCIBeta = axDose[0][0].fill_between(sorted(np.concatenate((self.ResponseBetaPIPS[1][0][1:],self.ResponseBetaPIPS[1][0][:-1]))), 
                                            np.repeat(binRecoFluenceVal[0], 2), 
                                            np.repeat(binRecoFluenceVal[4], 2),
                                            color='red',
                                            alpha=0.4)
    
        pBCIGamma = axDose[0][1].fill_between(sorted(np.concatenate((self.ResponseGammaLaBr3[1][0][1:],self.ResponseGammaLaBr3[1][0][:-1]))), 
                                            np.repeat(binRecoFluenceVal[1], 2), 
                                            np.repeat(binRecoFluenceVal[5], 2),
                                            color='red',
                                            alpha=0.4)

        pMeanBeta, = axDose[0][0].plot(sorted(np.concatenate((self.ResponseBetaPIPS[1][0][1:],self.ResponseBetaPIPS[1][0][:-1]))), 
                                        np.repeat(binRecoFluenceVal[2], 2),
                                        lw=1.25, 
                                        color='red', 
                                        linestyle="-", 
                                        drawstyle='steps')
       

        pMeanGamma, = axDose[0][1].plot(sorted(np.concatenate((self.ResponseGammaLaBr3[1][1][1:],self.ResponseGammaLaBr3[1][1][:-1]))), 
                                        np.repeat(binRecoFluenceVal[3], 2),
                                        lw=1.25, 
                                        color='red', 
                                        linestyle="-", 
                                        drawstyle='steps')

        minY = np.min(binRecoFluenceVal[binRecoFluenceVal >= 1E-2])
        maxY = np.max(binRecoFluenceVal[np.isfinite(binRecoFluenceVal)])

        axDose[0][0].set_title('Reconstructed Beta-ray Fluence Spectrum')
        axDose[0][0].set_xlabel('True Energy (keV)')
        axDose[0][0].set_ylabel('Fluence (cm$^{-2}$)')
        axDose[0][0].set_xscale('log')
        axDose[0][0].set_yscale('log')
        axDose[0][0].set_xlim(min(self.ResponseBetaPIPS[1][0]),max(self.ResponseBetaPIPS[1][0]))
        axDose[0][0].set_ylim(np.power(10, np.floor(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        if plotTruth:
            axDose[0][0].legend([pTruthBeta, (pBCIBeta, pMeanBeta)], ['Truth','Reconstructed (95% BCI)'], loc='best')
        else:
            axDose[0][0].legend([(pBCIBeta, pMeanBeta)], ['Reconstructed (95% BCI)'], loc='best')

        axDose[0][1].set_title('Reconstructed Gamma-ray Fluence Spectrum')
        axDose[0][1].set_xlabel('True Energy (keV)')
        axDose[0][1].set_ylabel('Fluence (cm$^{-2}$)')
        axDose[0][1].set_xscale('log')
        axDose[0][1].set_yscale('log')
        axDose[0][1].set_xlim(min(self.ResponseGammaLaBr3[1][0]),max(self.ResponseGammaLaBr3[1][0]))
        axDose[0][1].set_ylim(np.power(10, np.floor(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        if plotTruth:
            axDose[0][1].legend([pTruthGamma, (pBCIGamma, pMeanGamma)], ['Truth','Reconstructed (95% BCI)'], loc='best')
        else:
            axDose[0][1].legend([(pBCIGamma, pMeanGamma)], ['Reconstructed (95% BCI)'], loc='best')

        # Plot the unfolded dose spectrum
        pBCIBetaDoseWB = axDose[1][0].fill_between(sorted(np.concatenate((self.ResponseBetaPIPS[1][0][1:],self.ResponseBetaPIPS[1][0][:-1]))), 
                                            np.repeat(binRecoDoseVal[0], 2), 
                                            np.repeat(binRecoDoseVal[10], 2),
                                            color='blue',
                                            alpha=0.5)
        
        pBCIBetaDoseEye = axDose[1][0].fill_between(sorted(np.concatenate((self.ResponseBetaPIPS[1][0][1:],self.ResponseBetaPIPS[1][0][:-1]))), 
                                            np.repeat(binRecoDoseVal[1], 2), 
                                            np.repeat(binRecoDoseVal[11], 2),
                                            color='orange',
                                            alpha=0.3)
    
        pBCIGammaDoseWB = axDose[1][1].fill_between(sorted(np.concatenate((self.ResponseGammaLaBr3[1][0][1:],self.ResponseGammaLaBr3[1][0][:-1]))), 
                                            np.repeat(binRecoDoseVal[2], 2), 
                                            np.repeat(binRecoDoseVal[12], 2),
                                            color='blue',
                                            alpha=0.5)

        pBCIGammaDoseSkin = axDose[1][1].fill_between(sorted(np.concatenate((self.ResponseGammaLaBr3[1][0][1:],self.ResponseGammaLaBr3[1][0][:-1]))), 
                                            np.repeat(binRecoDoseVal[3], 2), 
                                            np.repeat(binRecoDoseVal[13], 2),
                                            color='green',
                                            alpha=0.4)
        
        pBCIGammaDoseEye = axDose[1][1].fill_between(sorted(np.concatenate((self.ResponseGammaLaBr3[1][0][1:],self.ResponseGammaLaBr3[1][0][:-1]))), 
                                            np.repeat(binRecoDoseVal[4], 2), 
                                            np.repeat(binRecoDoseVal[14], 2),
                                            color='orange',
                                            alpha=0.3)

        pMeanBetaDoseWB, = axDose[1][0].plot(sorted(np.concatenate((self.ResponseBetaPIPS[1][0][1:],self.ResponseBetaPIPS[1][0][:-1]))), 
                                        np.repeat(binRecoDoseVal[5], 2),
                                        lw=1.25, 
                                        color='blue', 
                                        linestyle="-", 
                                        drawstyle='steps')
        
        pMeanBetaDoseEye, = axDose[1][0].plot(sorted(np.concatenate((self.ResponseBetaPIPS[1][0][1:],self.ResponseBetaPIPS[1][0][:-1]))), 
                                        np.repeat(binRecoDoseVal[6], 2),
                                        lw=1.25, 
                                        color='orange', 
                                        linestyle="-", 
                                        drawstyle='steps')

        pMeanGammaDoseWB, = axDose[1][1].plot(sorted(np.concatenate((self.ResponseGammaLaBr3[1][1][1:],self.ResponseGammaLaBr3[1][1][:-1]))), 
                                        np.repeat(binRecoDoseVal[7], 2),
                                        lw=1.25, 
                                        color='blue', 
                                        linestyle="-", 
                                        drawstyle='steps')
        
        pMeanGammaDoseSkin, = axDose[1][1].plot(sorted(np.concatenate((self.ResponseGammaLaBr3[1][1][1:],self.ResponseGammaLaBr3[1][1][:-1]))), 
                                        np.repeat(binRecoDoseVal[8], 2),
                                        lw=1.25, 
                                        color='green', 
                                        linestyle="-", 
                                        drawstyle='steps')
        
        pMeanGammaDoseEye, = axDose[1][1].plot(sorted(np.concatenate((self.ResponseGammaLaBr3[1][1][1:],self.ResponseGammaLaBr3[1][1][:-1]))), 
                                        np.repeat(binRecoDoseVal[9], 2),
                                        lw=1.25, 
                                        color='orange', 
                                        linestyle="-", 
                                        drawstyle='steps')

        # Plot Statistics
        tblStats1 = axDose[1][0].table( cellText = (
                                        ('Eye',
                                        '{:0.0f} nGy'.format(np.sum(binTruthDoseVal[1])),
                                        '{:0.0f} ({:0.0f}-{:0.0f}) nGy'.format(np.sum(binRecoDoseVal[6]), np.sum(binRecoDoseVal[1]), np.sum(binRecoDoseVal[11]))),
                                        ('Whole Body',
                                        '{:0.0f} nsV'.format(np.sum(binTruthDoseVal[0])),
                                        '{:0.0f} ({:0.0f}-{:0.0f}) nSv'.format(np.sum(binRecoDoseVal[5]), np.sum(binRecoDoseVal[0]), np.sum(binRecoDoseVal[10])))),
                            cellLoc = 'center',
                            colLabels = ['Organ', 'True Dose', 'Estimated Dose (95% BCI)'],
                            colLoc = 'center',
                            loc = 'upper left')             
        tblStats1.auto_set_column_width(0)
        tblStats1.auto_set_column_width(1)
        tblStats1.auto_set_column_width(2)
        for key, cell in tblStats1.get_celld().items():
            cell.set_linewidth(0)
        
        tblStats2 = axDose[1][1].table( cellText = (
                                        ('Skin',
                                        '{:0.0f} nGy'.format(np.sum(binTruthDoseVal[3])),
                                        '{:0.0f} ({:0.0f}-{:0.0f}) nGy'.format(np.sum(binRecoDoseVal[8]), np.sum(binRecoDoseVal[3]), np.sum(binRecoDoseVal[13]))),
                                        ('Eye',
                                        '{:0.0f} nGy'.format(np.sum(binTruthDoseVal[4])),
                                        '{:0.0f} ({:0.0f}-{:0.0f}) nGy'.format(np.sum(binRecoDoseVal[9]), np.sum(binRecoDoseVal[4]), np.sum(binRecoDoseVal[14]))),
                                        ('Whole Body',
                                        '{:0.0f} nsV'.format(np.sum(binTruthDoseVal[2])),
                                        '{:0.0f} ({:0.0f}-{:0.0f}) nSv'.format(np.sum(binRecoDoseVal[7]), np.sum(binRecoDoseVal[2]), np.sum(binRecoDoseVal[12])))),
                            cellLoc = 'center',
                            colLabels = ['Organ', 'True Dose', 'Estimated Dose (95% BCI)'],
                            colLoc = 'center',
                            loc = 'upper left')
        tblStats2.auto_set_column_width(0)
        tblStats2.auto_set_column_width(1)
        tblStats2.auto_set_column_width(2)
        for key, cell in tblStats2.get_celld().items():
            cell.set_linewidth(0)
        

        # Figure Properties
        dnrFluence = maxY/minY      # Limit the dynamic range of the dose spectrum to the same as the fluence spectrum
        maxY = np.max(binRecoDoseVal[np.isfinite(binRecoDoseVal)])
        minY = np.min(binRecoDoseVal[binRecoDoseVal >= maxY/dnrFluence])

        axDose[1][0].set_title('A2 - Reconstructed Beta-ray Dose Spectrum')
        axDose[1][0].set_xlabel('True Energy (keV)')
        axDose[1][0].set_ylabel('Dose (nSv or nGy)')
        axDose[1][0].set_xscale('log')
        axDose[1][0].set_yscale('log')
        axDose[1][0].set_xlim(min(self.ResponseBetaLaBr3[1][0]),max(self.ResponseBetaLaBr3[1][0]))
        axDose[1][0].set_ylim(np.power(10, np.ceil(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        axDose[1][0].legend([(pBCIBetaDoseEye, pMeanBetaDoseEye), (pBCIBetaDoseWB, pMeanBetaDoseWB)],
                                ['Lens of Eye (95% BCI)', 'Whole Body (95% BCI)'],
                                bbox_to_anchor=(0., 1.02, 1., .102), ncol=2, loc=3, mode="expand", borderaxespad=0.)

        axDose[1][1].set_title('B2 - Reconstructed Gamma-ray Dose Spectrum')
        axDose[1][1].set_xlabel('True Energy (keV)')
        axDose[1][1].set_ylabel('Dose (nSv or nGy)')
        axDose[1][1].set_xscale('log')
        axDose[1][1].set_yscale('log')
        axDose[1][1].set_xlim(min(self.ResponseGammaLaBr3[1][0]),max(self.ResponseGammaLaBr3[1][0]))
        axDose[1][1].set_ylim(np.power(10, np.ceil(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        axDose[1][1].legend([(pBCIGammaDoseSkin, pMeanGammaDoseSkin),(pBCIGammaDoseEye, pMeanGammaDoseEye),(pBCIGammaDoseWB, pMeanGammaDoseWB)],
                                ['Skin (95% BCI)','Lens of Eye (95% BCI)', 'Whole Body (95% BCI)'],
                                bbox_to_anchor=(0., 1.02, 1., .102), ncol=3, loc=3, mode="expand", borderaxespad=0.)
        
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

    def buildModel(self, DataPIPS, DataLaBr3, TruthBeta, TruthGamma):
        '''
        Build a multidimensional inference model
        '''
        # Check the input instance type as follows: 
        if not isinstance(DataPIPS, ROOT.TH1): raise TypeError("Data histogram from the PIPS detector must be of type ROOT.TH1")
        if not isinstance(DataLaBr3, ROOT.TH1): raise TypeError("Data histogram from the LaBr3 detector must be of type ROOT.TH1")
        if not isinstance(TruthBeta, ROOT.TH1): raise TypeError("Truth histogram for the Beta spectrum must be of type ROOT.TH1")
        if not isinstance(TruthGamma, ROOT.TH1): raise TypeError("Truth histogram for the Gamma spectrum must be of type ROOT.TH1")

        # Copy the inputs to the object
        self.DataPIPS = hist2array(DataPIPS, include_overflow=False, copy=True, return_edges=True)
        self.DataLaBr3 = hist2array(DataLaBr3, include_overflow=False, copy=True, return_edges=True)
        self.TruthBeta = hist2array(TruthBeta, include_overflow=False, copy=True, return_edges=True)
        self.TruthGamma = hist2array(TruthGamma, include_overflow=False, copy=True, return_edges=True)

        # Build the model
        with pm.Model() as self.model:

            # Define the upper and lower bounds for the priors
            GFBetaPIPS = np.sum(self.ResponseBetaPIPS[0], axis=1)
            GFGammaPIPS = np.sum(self.ResponseGammaPIPS[0], axis=1)
            GFBetaLaBr3 = np.sum(self.ResponseBetaLaBr3[0], axis=1)
            GFGammaLaBr3 = np.sum(self.ResponseGammaLaBr3[0], axis=1)

            SFBeta = GFBetaPIPS/np.power(GFBetaPIPS + GFGammaPIPS, 2)
            SFBeta[np.isclose(SFBeta, 0)] = np.finfo(np.float64).eps
            SFGamma = GFGammaLaBr3/np.power(GFBetaLaBr3 + GFGammaLaBr3, 2)
            SFGamma[np.isclose(SFGamma, 0)] = np.finfo(np.float64).eps

            nCountsPIPS = np.sum(self.DataPIPS[0])
            nCountsLaBr3 = np.sum(self.DataLaBr3[0])

            lbPhiBeta = np.zeros(self.ResponseBetaPIPS[1][0].size-1)
            ubPhiBeta = np.ones(self.ResponseBetaPIPS[1][0].size-1)*nCountsPIPS*SFBeta
            lbPhiGamma = np.zeros(self.ResponseGammaLaBr3[1][0].size-1)
            ubPhiGamma = np.ones(self.ResponseGammaLaBr3[1][0].size-1)*nCountsLaBr3*SFGamma

            ubPhiBeta[np.isclose(ubPhiBeta, 0)] = 1E-15
            ubPhiGamma[np.isclose(ubPhiGamma, 0)] = 1E-15

            # Define the prior probability densities 
            self.PhiBeta = pm.Uniform('PhiBeta', 
                                lower = lbPhiBeta,
                                upper = ubPhiBeta, 
                                shape = (self.ResponseBetaPIPS[1][0].size-1))
            
            self.PhiGamma = pm.Uniform('PhiGamma', 
                                lower = lbPhiGamma,
                                upper = ubPhiGamma, 
                                shape = (self.ResponseGammaPIPS[1][0].size-1))

            # Define the models
            self.MPIPS = theano.tensor.dot(theano.shared(self.asMat(self.ResponseBetaPIPS[0].T)), self.PhiBeta) + theano.tensor.dot(theano.shared(self.asMat(self.ResponseGammaPIPS[0].T)), self.PhiGamma)
            self.MLaBr3 = theano.tensor.dot(theano.shared(self.asMat(self.ResponseBetaLaBr3[0].T)), self.PhiBeta) + theano.tensor.dot(theano.shared(self.asMat(self.ResponseGammaLaBr3[0].T)), self.PhiGamma)
            
            # Define the likelihood
            self.LPIPS = pm.Poisson('Likelihood_PIPS', 
                                    mu = self.MPIPS,
                                    observed = theano.shared(self.DataPIPS[0], borrow = False), 
                                    shape = (self.DataPIPS[0].size, 1))

            self.LLaBr3 = pm.Poisson('Likelihood_LaBr3', 
                                    mu = self.MLaBr3,
                                    observed = theano.shared(self.DataLaBr3[0], borrow = False), 
                                    shape = (self.DataLaBr3[0].size, 1))

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

    def sampleNUTS(self, N = 50000, B = 20000):
        '''
        Function to sample the posterior distribution using a Markov Chain Monte Carlo (MCMC) and the
        No-U-Turn Sampling (NUTS) algorithm in PyMC3.
        ''' 
        self.Samples = N
        self.Burn = B
        with self.model:
            # Initialize with ADVI to speedup NUTS
            # https://tdhopper.com/blog/speeding-up-pymc3-nuts-sampler/
            #print 'Initialize the sampling using ADVI ...'
            mu, sds, elbo = pm.variational.advi(n=200000)

            # Initialization using MAP or some other algorithm
            #print 'Finding a good starting point ...'
            #mu, sds, elbo = pm.variational.advi(n=200000)
            #start = pm.find_MAP(model = self.model)

            # Select the Posterior sampling algorithm
            print 'Sampling the posterior using NUTS ...'
            step = pm.NUTS(scaling=np.power(self.model.dict_to_array(sds), 2), is_cov=True)

            # Sample
            self.trace = pm.sample(self.Samples,
                                   tune = self.Burn,
                                   start = mu,
                                   step=step)
            

            # Print a summary of the MCMC trace      
            pm.summary(self.trace)

# ROOT file context manager
class ROOTFile(object):

    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.file = ROOT.TFile.Open(self.filename, 'read')
        return self.file

    def __exit__(self, exception_type, exception_value, traceback):
        self.file.Close()

# Settings
ResponseFilePIPS = './../TestData/Canberra PD450-15-500AM/Response Matrix/Canberra PD450-15-500AM PIPS.root'
ResponseFileLaBr3 = './../TestData/Saint Gobain B380 LaBr3/Response Matrix/Saint Gobain B380 LaBr3.root'

DataFilePIPS = './../TestData/Canberra PD450-15-500AM/Cs137/Cs137_R_35_cm_Nr_200000000_ISO.root'
DataFileLaBr3 = './../TestData/Saint Gobain B380 LaBr3/Cs137/Cs137_R_35_cm_Nr_200000000_ISO.root'

DoseCoeffFolder = './../../Dose Coefficients/'
fDoseCoeffGamma  = 'ICRP116_Photon_DoseConversionCoefficients.xlsx'
fDoseCoeffBeta  = 'ICRP116_Electron_DoseConversionCoefficients.xlsx'

with ROOTFile(ResponseFilePIPS) as fResponsePIPS:
    with ROOTFile(ResponseFileLaBr3) as fResponseLaBr3:
        with ROOTFile(DataFilePIPS) as fDataPIPS:
            with ROOTFile(DataFileLaBr3) as fDataLaBr3:
                # Initiate the class
                myMBSD = PyMBSD(MigBetaPIPS = fResponsePIPS.Get('Energy Migration Matrix (Electron)'),
                                MigGammaPIPS = fResponsePIPS.Get('Energy Migration Matrix (Gamma)'), 
                                MigBetaLaBr3 = fResponseLaBr3.Get('Energy Migration Matrix (Electron)'),
                                MigGammaLaBr3 = fResponseLaBr3.Get('Energy Migration Matrix (Gamma)'),
                                SourceBetaPIPS = fResponsePIPS.Get('Source Spectrum (Electron)'),
                                SourceGammaPIPS = fResponsePIPS.Get('Source Spectrum (Gamma)'),
                                SourceBetaLaBr3 = fResponseLaBr3.Get('Source Spectrum (Electron)'),
                                SourceGammaLaBr3 = fResponseLaBr3.Get('Source Spectrum (Gamma)'))

                # Plot the response matrices
                myMBSD.plotResponse(fName = 'ResponseMatrix.jpg')

                # Load the dose coefficients (NOTE: Using ICRP 116)
                myMBSD.loadDoseCoeffGamma(fName = DoseCoeffFolder + fDoseCoeffGamma)
                myMBSD.loadDoseCoeffBeta(fName = DoseCoeffFolder + fDoseCoeffBeta)

                # Build the model
                myMBSD.buildModel(DataPIPS = fDataPIPS.Get('Detector Measured Spectrum'),
                                  DataLaBr3 = fDataLaBr3.Get('Detector Measured Spectrum'),
                                  TruthBeta = fDataPIPS.Get('Source Spectrum (Electron)'),
                                  TruthGamma = fDataPIPS.Get('Source Spectrum (Gamma)'))

                # Run Variational Inference
                myMBSD.sampleADVI()
                
                myMBSD.plotUnfoldedFluenceSpectrum(fName = DataFilePIPS.split('.')[-2].split('/')[-1] + '_Fluence_ADVI.pdf')
                myMBSD.plotUnfoldedDoseSpectrum(fName = DataFilePIPS.split('.')[-2].split('/')[-1] + '_Dose_ADVI.pdf', plotTruth = True)

                # Run MCMC Inference
                #myMBSD.sampleNUTS(20000,20000)
                #myMBSD.plotUnfoldedFluenceSpectrum(fName = 'Unfolded_Fluence_NUTS.pdf')

