import numpy as np
import pymc3 as pm
import ROOT 

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
        'legend.fontsize': 10,
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
#        'text.usetex': True,
        'figure.figsize': fig_size}

# Update rcParams
rcParams.update(params)

class PyBSD(object):
    '''
    Object initialization function

    PARAMETERS:

    migration: A ROOT TH2 histogram whose elements are the joint probability P(t,d)
               of an event that was produced in the truth energy bin t and measured
               in the detector energy bin d. Other names for this matrix include
               energy migration matrix or smearing matrix.

               NOTE: The x-axis is the truth energy bins
                     The y-axis is the detected energy bins

    truth:    A ROOT TH1 histogram which contains the true spectrum from which the
              measured energy originated before smearing
    '''
    def __init__(self, migration=ROOT.TH2D(), sourcetruth=ROOT.TH1D(), migration2=ROOT.TH2D(), sourcetruth2=ROOT.TH1D()):

        # Check the input instance type as follows:
        # migration = ROOT.TH2
        # truth = ROOT.TH1
        if not isinstance(migration, ROOT.TH2): raise TypeError("Migration matrix must be of type ROOT.TH2")
        if not isinstance(sourcetruth, ROOT.TH1): raise TypeError("Truth histogram for the source spectrum must be of type ROOT.TH1")
        if not isinstance(migration2, ROOT.TH2): raise TypeError("Second migration matrix must be of type ROOT.TH2")
        if not isinstance(sourcetruth2, ROOT.TH1): raise TypeError("Second truth histogram for the source spectrum must be of type ROOT.TH1")

        # Copy the inputs to the object
        self.migration = copy.deepcopy(migration)
        self.sourcetruth = copy.deepcopy(sourcetruth)
        self.migration2 = copy.deepcopy(migration2)
        self.sourcetruth2 = copy.deepcopy(sourcetruth2)

        # Calculate the response matrix (aka. conditional probability) using Eq. 5 from the Choudalakis paper
        # Response[i,j] = P(d = j|t = i) = P(t = i, d = j)/P(t = i)
        # Response[j|i] = M[d = j, t = i] / Truth[i]
        self.response = copy.deepcopy(migration)
        #tSum = np.sum([[self.response.GetBinContent(i+1,j+1) for i in range(0, self.response.GetNbinsX())] for j in range(0, self.response.GetNbinsY())])
        for i in range(0, self.response.GetNbinsX()):
            tSum = np.sum([self.response.GetBinContent(i+1,j+1) for j in range(0, self.response.GetNbinsY())])
            for j in range(0, self.response.GetNbinsY()):
                #self.response.SetBinContent(i+1, j+1, self.response.GetBinContent(i+1,j+1)/self.sourcetruth.GetBinContent(i+1))
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Normalize with Truth
                    self.response.SetBinContent(i+1,
                                                j+1,
                                                (self.response.GetBinContent(i+1,j+1)/self.sourcetruth.GetBinContent(i+1) if np.isfinite(self.response.GetBinContent(i+1,j+1)/self.sourcetruth.GetBinContent(i+1)) else 0.))

        # Calculate the second response matrix (aka. conditional probability) using Eq. 5 from the Choudalakis paper
        # Response[i,j] = P(d = j|t = i) = P(t = i, d = j)/P(t = i)
        # Response[j|i] = M[d = j, t = i] / Truth[i]
        self.response2 = copy.deepcopy(migration2)
        for i in range(0, self.response2.GetNbinsX()):
            tSum = np.sum([self.response2.GetBinContent(i+1,j+1) for j in range(0, self.response2.GetNbinsY())])
            for j in range(0, self.response2.GetNbinsY()):
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Normalize with Truth
                    self.response2.SetBinContent(i+1,
                                                j+1,
                                                (self.response2.GetBinContent(i+1,j+1)/self.sourcetruth2.GetBinContent(i+1) if np.isfinite(self.response2.GetBinContent(i+1,j+1)/self.sourcetruth2.GetBinContent(i+1)) else 0.))

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
        self.coeffGammaWB = np.array([logInterpCoeff(np.array([self.response2.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsY() + 1)]), 
                                                     df_ICRP116_Photon_WholeBody['Energy (MeV)'].values*1E3, 
                                                     df_ICRP116_Photon_WholeBody['ISO (pSv cm2)'].values*coeffScalingFactor),
                                      np.array([self.response2.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsY() + 1)])])
        
        self.coeffGammaSkin = np.array([logInterpCoeff(np.array([self.response2.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsY() + 1)]), 
                                                     df_ICRP116_Photon_MaleSkin['Energy (MeV)'].values*1E3, 
                                                     df_ICRP116_Photon_MaleSkin['ISO (pGy cm2)'].values*coeffScalingFactor),
                                      np.array([self.response2.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsY() + 1)])])
        
        self.coeffGammaEye = np.array([logInterpCoeff(np.array([self.response2.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsY() + 1)]), 
                                                     df_ICRP116_Photon_EyeLens['Energy (MeV)'].values*1E3, 
                                                     df_ICRP116_Photon_EyeLens['ISO (pGy cm2)'].values*coeffScalingFactor),
                                      np.array([self.response2.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsY() + 1)])])
    
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
        self.coeffBetaWB = np.array([logInterpCoeff(np.array([self.response2.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsY() + 1)]), 
                                                     df_ICRP116_Beta_WholeBody['Energy (MeV)'].values*1E3, 
                                                     df_ICRP116_Beta_WholeBody['ISO (pSv cm2)'].values*coeffScalingFactor),
                                      np.array([self.response2.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsY() + 1)])])
        
        self.coeffBetaEye = np.array([logInterpCoeff(np.array([self.response2.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsY() + 1)]), 
                                                     df_ICRP116_Beta_EyeLens['Energy (MeV)'].values*1E3, 
                                                     df_ICRP116_Beta_EyeLens['ISO (pGy cm2)'].values*coeffScalingFactor),
                                      np.array([self.response2.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsY() + 1)])])
    
    def plotData(self,  fName='DataHistogram.jpg'):
        '''
        Function to plot the measure data histogram

        IMPORTANT NOTE: It is customary to plot data (D) with error bars equal to sqrt(D) since it is
                        assumed they come from an underlying Poisson distribution. This is a frequentist
                        approach and assumes the data has uncertainty. However, there is usually no uncertainy
                        in the number of events we counted, assuming we know how to count properly. The 
                        uncertainty is in the parameters of the underlying probability distribution function.

                        This is an important aspect of the Bayesian approach. Measured data has no uncertainty!
        '''

        # Get bin values, errors, and edges
        binVal = np.array([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())])
        binEdge = np.array([self.data.GetBinLowEdge(i+1) for i in range(0, self.data.GetNbinsX() + 1)])

        # Create a figure
        figData, axData = plt.subplots()

        # Plot the data
        axData.plot(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), np.repeat(binVal, 2), lw=1.25, color='black', linestyle="-", drawstyle='steps')

        # Figure properties
        axData.set_xlabel('Measured Energy (keV)' if not self.data.GetXaxis().GetTitle() else self.data.GetXaxis().GetTitle())
        axData.set_ylabel('Counts' if not self.data.GetYaxis().GetTitle() else self.data.GetYaxis().GetTitle())
        axData.set_xlim(min(binEdge),max(binEdge))
        axData.set_ylim(np.power(10, np.floor(np.log10(np.min(binVal[binVal > 0])))), np.power(10, np.ceil(np.log10(np.max(binVal)))))
        axData.set_xscale('log')
        axData.set_yscale('log')
        axData.legend(loc='upper right')

        # Fine-tune figure 
        figData.tight_layout()

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Data plot saved to: ' + fName

        # Show the figure
        plt.close(figData)
    
    # Function to plot the migration matrix
    def plotMigration(self,  fName='MigrationMatrix.jpg'):

        # Get bin values, errors, and edges
        binVal = np.array([[self.migration.GetBinContent(i+1,j+1) for i in range(0, self.migration.GetNbinsX())] for j in range(0, self.migration.GetNbinsY())])
        binEdge = np.array([[self.migration.GetXaxis().GetBinLowEdge(i+1) for i in range(0, self.migration.GetNbinsX() + 1)],
                            [self.migration.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.migration.GetNbinsY() + 1)]])

        # Create a figure
        figMigration, axMigration = plt.subplots()

        # Color map
        cmap = palettable.matplotlib.Viridis_20.mpl_colormap
        cmap.set_bad(cmap(0.))
        cmap.set_over(cmap(1.)) 

        # Plot the migration matrix
        X, Y = np.meshgrid(binEdge[0], binEdge[1])
        H = axMigration.pcolormesh(X, Y, binVal, norm = colors.LogNorm(), cmap = cmap, linewidth = 0, rasterized = True)

        # Set color limits for the plot
        H.set_clim(np.power(10, np.floor(np.log10(np.min(binVal[binVal>0])))), np.power(10, np.ceil(np.log10(np.max(binVal[binVal>0])))))

        # Add a colorbar
        cbar = figMigration.colorbar(H, ax=axMigration, pad = 0.01, aspect = 20., extend = 'both', spacing = 'uniform')
        cbar.set_label('# of Events' if not self.migration.GetZaxis().GetTitle() else self.migration.GetZaxis().GetTitle())  

        # Figure properties
        axMigration.set_xlabel('True Energy (keV)' if not self.migration.GetXaxis().GetTitle() else self.migration.GetXaxis().GetTitle())
        axMigration.set_ylabel('Measured Energy (keV)' if not self.migration.GetYaxis().GetTitle() else self.migration.GetYaxis().GetTitle())
        axMigration.set_xlim(min(binEdge[0]),max(binEdge[0]))
        axMigration.set_ylim(min(binEdge[1]),max(binEdge[1]))
        axMigration.set_xscale('log')
        axMigration.set_yscale('log')

        # Fine-tune figure 
        figMigration.tight_layout()

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Migration matrix plot saved to: ' + fName

        # Show the figure
        plt.close(figMigration)
    
    # Function to plot the second migration matrix
    def plotMigration2(self,  fName='MigrationMatrix2.jpg'):

        # Get bin values, errors, and edges
        binVal = np.array([[self.migration2.GetBinContent(i+1,j+1) for i in range(0, self.migration2.GetNbinsX())] for j in range(0, self.migration2.GetNbinsY())])
        binEdge = np.array([[self.migration2.GetXaxis().GetBinLowEdge(i+1) for i in range(0, self.migration2.GetNbinsX() + 1)],
                            [self.migration2.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.migration2.GetNbinsY() + 1)]])

        # Create a figure
        figMigration, axMigration = plt.subplots()

        # Color map
        cmap = palettable.matplotlib.Viridis_20.mpl_colormap
        cmap.set_bad(cmap(0.))
        cmap.set_over(cmap(1.)) 

        # Plot the migration matrix
        X, Y = np.meshgrid(binEdge[0], binEdge[1])
        H = axMigration.pcolormesh(X, Y, binVal, norm = colors.LogNorm(), cmap = cmap, linewidth = 0, rasterized = True)

        # Set color limits for the plot
        H.set_clim(np.power(10, np.floor(np.log10(np.min(binVal[binVal>0])))), np.power(10, np.ceil(np.log10(np.max(binVal[binVal>0])))))

        # Add a colorbar
        cbar = figMigration.colorbar(H, ax=axMigration, pad = 0.01, aspect = 20., extend = 'both', spacing = 'uniform')
        cbar.set_label('# of Events' if not self.migration2.GetZaxis().GetTitle() else self.migration2.GetZaxis().GetTitle())  

        # Figure properties
        axMigration.set_xlabel('True Energy (keV)' if not self.migration2.GetXaxis().GetTitle() else self.migration2.GetXaxis().GetTitle())
        axMigration.set_ylabel('Measured Energy (keV)' if not self.migration2.GetYaxis().GetTitle() else self.migration2.GetYaxis().GetTitle())
        axMigration.set_xlim(min(binEdge[0]),max(binEdge[0]))
        axMigration.set_ylim(min(binEdge[1]),max(binEdge[1]))
        axMigration.set_xscale('log')
        axMigration.set_yscale('log')

        # Fine-tune figure 
        figMigration.tight_layout()

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Migration matrix plot saved to: ' + fName

        # Show the figure
        plt.close(figMigration)

    def plotResponse(self,  fName='ResponseMatrix.jpg'):
        '''
        Function to plot the response matrix
        '''

        # Get bin values, errors, and edges
        binVal = np.array([[[self.response.GetBinContent(i+1,j+1) for i in range(0, self.response.GetNbinsX())] for j in range(0, self.response.GetNbinsY())],
                           [[self.response2.GetBinContent(i+1,j+1) for i in range(0, self.response2.GetNbinsX())] for j in range(0, self.response2.GetNbinsY())]])

        binTruthVal = np.array([[self.sourcetruth.GetBinContent(i+1) for i in range(0, self.sourcetruth.GetNbinsX())],
                                [self.sourcetruth2.GetBinContent(i+1) for i in range(0, self.sourcetruth2.GetNbinsX())]])

        binEdge = np.array([[self.response.GetXaxis().GetBinLowEdge(i+1) for i in range(0, self.response.GetNbinsX() + 1)],
                            [self.response.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.response.GetNbinsY() + 1)]])
        binEdge2 = np.array([[self.response2.GetXaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsX() + 1)],
                            [self.response2.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsY() + 1)]])
        

        figResponse = plt.figure()
        gs = gridspec.GridSpec(2, 2) 

        # Color map
        cmap = palettable.matplotlib.Viridis_20.mpl_colormap
        cmap.set_bad(cmap(0.))
        cmap.set_over(cmap(1.))

        # Response Limits
        rLimLow = 1E-3
        rLimUp = 1E2

        # Plot the response matrices
        X, Y = np.meshgrid(binEdge[0], binEdge[1])
        axResponseBeta = plt.subplot(gs[0,0])
        axResponseBetaIns = inset_axes(axResponseBeta, width="35%", height="5%", loc = 2, borderpad=2)
        H = axResponseBeta.pcolormesh(X, Y, binVal[0], norm = colors.LogNorm(), cmap = cmap, linewidth = 0, rasterized = True) 

        X, Y = np.meshgrid(binEdge2[0], binEdge2[1])
        axResponseGamma = plt.subplot(gs[0,1])
        axResponseGammaIns = inset_axes(axResponseGamma, width="35%", height="5%", loc = 2, borderpad=2)
        H2 = axResponseGamma.pcolormesh(X, Y, binVal[1], norm = colors.LogNorm(), cmap = cmap, linewidth = 0, rasterized = True) 

        # Color limits for the plot
        H.set_clim(rLimLow, rLimUp)
        H2.set_clim(rLimLow, rLimUp)

        # Figure Properties
        axResponseBeta.set_title('Beta-ray Response Matrix')
        axResponseBeta.set_ylabel('Measured Energy (keV)')
        axResponseBeta.set_yscale('log')
        axResponseBeta.set_xscale('log')
                    
        axResponseGamma.set_title('Gamma-ray Response Matrix')
        axResponseGamma.set_yscale('log')
        axResponseGamma.set_xscale('log') 

        # Add a colorbar
        cbar = figResponse.colorbar(H, cax=axResponseBetaIns, orientation='horizontal', extend = 'both', spacing = 'uniform')
        cbar.outline.set_edgecolor('white')
        axResponseBetaIns.set_title('Response (cm$^2$)', fontdict={'family' : 'monospace'})
        axResponseBetaIns.title.set_color('white')
        axResponseBetaIns.xaxis.label.set_color('white')
        axResponseBetaIns.tick_params(axis='x', colors='white')

        cbar2 = figResponse.colorbar(H2, cax = axResponseGammaIns, orientation='horizontal', extend = 'both', spacing = 'uniform')
        cbar2.outline.set_edgecolor('white')
        axResponseGammaIns.set_title('Response (cm$^2$)', fontdict={'family' : 'monospace'})
        axResponseGammaIns.title.set_color('white')
        axResponseGammaIns.xaxis.label.set_color('white')
        axResponseGammaIns.tick_params(axis='x', colors='white')

        # Plot the geometric factor spectrum
        axGFBeta = plt.subplot(gs[1,0], sharex = axResponseBeta)
        axGFBeta.plot(sorted(np.concatenate((binEdge[0][:-1],binEdge[1][1:]))), 
                            np.repeat(np.sum(binVal[0], axis=0), 2),
                            lw=1.25,
                            color='black',
                            linestyle="-",
                            drawstyle='steps')

        axGFGamma = plt.subplot(gs[1,1], sharex = axResponseGamma)
        axGFGamma.plot(sorted(np.concatenate((binEdge2[0][:-1],binEdge2[1][1:]))), 
                            np.repeat(np.sum(binVal[1], axis=0), 2),
                            lw=1.25,
                            color='black',
                            linestyle="-",
                            drawstyle='steps')

        axGFBeta.text(0.03,0.95,
        #axGFBeta.text(0.03,0.25, 
                    'Max: {:02.2f} cm$^2$\
                    \nAvg: {:02.2f} cm$^2$\
                    \nMin: {:02.2f} cm$^2$'
                    .format(np.max(np.sum(binVal[0], axis=0)),
                            np.mean(np.sum(binVal[0], axis=0)),
                            np.min(np.sum(binVal[0], axis=0))), 
                    transform=axGFBeta.transAxes, 
                    verticalalignment='top', 
                    fontdict={'family' : 'monospace'})

        #axGFGamma.text(0.75,0.95, 
        axGFGamma.text(0.03,0.25, 
        #axGFGamma.text(0.72,0.3, 
                    'Max: {:02.2f} cm$^2$\
                    \nAvg: {:02.2f} cm$^2$\
                    \nMin: {:02.2f} cm$^2$'
                    .format(np.max(np.sum(binVal[1], axis=0)),
                            np.mean(np.sum(binVal[1], axis=0)),
                            np.min(np.sum(binVal[1], axis=0))), 
                    transform=axGFGamma.transAxes, 
                    verticalalignment='top', 
                    fontdict={'family' : 'monospace'})

        # Figure Properties
        axGFBeta.set_ylabel('Omnidirectional Response (cm$^2$)')
        axGFBeta.set_xlabel('True Energy (keV)') 
        axGFBeta.set_ylim(rLimLow,rLimUp)
        axGFBeta.set_yscale('log')
                    
        #axGFGamma.set_ylabel('Geometric Factor (cm$^2$)')  
        axGFGamma.set_xlabel('True Energy (keV)')
        axGFGamma.set_ylim(rLimLow,rLimUp)
        axGFGamma.set_yscale('log')

        plt.setp(axResponseBeta.get_xticklabels(), visible=False)
        plt.setp(axResponseGamma.get_xticklabels(), visible=False)
        plt.setp(axResponseGamma.get_yticklabels(), visible=False)
        plt.setp(axGFGamma.get_yticklabels(), visible=False)

        # Fine-tune figure 
        figResponse.tight_layout()
        figResponse.subplots_adjust(hspace=0.06, wspace = 0.06)

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Response matrix plot saved to: ' + fName

        # Show the figure
        plt.close(figResponse)

    # Function to plot the truth histogram
    def plotTruth(self, fName='TruthHistogram.jpg'):
        # Get bin values, errors, and edges
        binVal = np.array([self.sourcetruth.GetBinContent(i+1) for i in range(0, self.sourcetruth.GetNbinsX())])
        binEdge = np.array([self.sourcetruth.GetBinLowEdge(i+1) for i in range(0, self.sourcetruth.GetNbinsX() + 1)])
        binCenter = np.array([self.sourcetruth.GetBinCenter(i+1) for i in range(0, self.sourcetruth.GetNbinsX())])

        # Create a figure
        figTruth, axTruth = plt.subplots()

        # Plot the truth
        axTruth.plot(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), np.repeat(binVal, 2), lw=1.25, color='black', linestyle="-", drawstyle='steps')

        # Figure properties
        axTruth.set_xlabel('True Energy (keV)' if not self.sourcetruth.GetXaxis().GetTitle() else self.sourcetruth.GetXaxis().GetTitle())
        axTruth.set_ylabel('# of Events' if not self.sourcetruth.GetYaxis().GetTitle() else self.sourcetruth.GetYaxis().GetTitle())
        axTruth.set_xlim(min(binEdge),max(binEdge))
        axTruth.set_ylim(np.power(10, np.floor(np.log10(np.min(binVal[binVal > 0])))), np.power(10, np.ceil(np.log10(np.max(binVal)))))
        axTruth.set_xscale('log')
        axTruth.set_yscale('log')

        # Fine-tune figure 
        figTruth.tight_layout()

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Truth plot saved to: ' + fName

        # Show the figure
        plt.close(figTruth)
    
    def calcSignificance(self, expected, observed):
        '''
        Function to calculate the statistical significance between two spectra using the algorithm described in Ref [1].

        References:
        [1] Choudalakis, Georgios, and Diego Casadei. "Plotting the differences between data and expectation." The European Physical Journal Plus 127.2 (2012): 25.
        '''
        pvalue = np.zeros(observed.size)
        zscore = np.zeros(observed.size)
        for i in range(observed.size):
            # Calculate the p-value
            if observed[i] > expected[i]:
                pvalue[i] = 1 - ROOT.Math.inc_gamma_c(observed[i], expected[i])
            else:
                pvalue[i] = ROOT.Math.inc_gamma_c(observed[i] + 1, expected[i])
            
            if (np.isclose(observed[i],0.) and np.isclose(expected[i],0.)): pvalue[i] = 1.

            # Calculate the z-score using the p-value
            if pvalue[i] > 1E-17:
                zscore[i] = ROOT.TMath.ErfInverse(1.0 - 2.0*pvalue[i])*ROOT.TMath.Sqrt(2.0)
            else:
                # Inversion is not possible, try something else
                u = -2. * np.log( pvalue[i] * np.sqrt(2.*np.pi))
                if np.isfinite(u):
                    zscore[i] = np.sqrt(u - np.log(u))
                else:
                    zscore[i] = 40.

                #u = -2.0 * ROOT.TMath.Log( pvalue[i]*ROOT.TMath.Sqrt( 2.*ROOT.TMath.Pi()))
                #zscore[i] = np.nan_to_num(ROOT.TMath.Sqrt(u - ROOT.TMath.Log(u)))

            # If there is a deficit of events, and its statistically significant (ie. pvalue < 0.5)
            # then take the negative of the z-value.
            if observed[i] < expected[i] and pvalue[i] < 0.5:
                zscore[i] *= -1.

        # Return signed z-score only if p-value < 0.5
        # See: https://arxiv.org/pdf/1111.2062.pdf
        zscore[pvalue >= 0.5] = 0.

        return zscore

    def plotUnfoldedFluenceSpectrum(self, fName='UnfoldedFluenceSpectrum.pdf', plotTruth = False):
        '''
        Function to plot the reconstructed fluence spectrum after performing multidimensional Bayesian unfolding
        NOTE: This function is to be used when multiple response matrices are used in the unfolding.
        '''

        binDataVal = np.array([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())])
        binDataEdge = np.array([self.data.GetBinLowEdge(i+1) for i in range(0, self.data.GetNbinsX() + 1)])

        binTruthVal = np.array([[self.datatruth.GetBinContent(i+1) for i in range(0, self.datatruth.GetNbinsX())],
                                [self.datatruth2.GetBinContent(i+1) for i in range(0, self.datatruth2.GetNbinsX())]])
        binTruthEdge = np.array([[self.datatruth.GetBinLowEdge(i+1) for i in range(0, self.datatruth.GetNbinsX() + 1)],
                                 [self.datatruth2.GetBinLowEdge(i+1) for i in range(0, self.datatruth2.GetNbinsX() + 1)]])

        # Calculate and plot the 95% Bayesian credible regions for the unfolded spectrum
        unfoldedBCI = pm.stats.hpd(self.trace.Truth, alpha=0.05)
        binRecoVal = np.array([unfoldedBCI[:,0,0],                      # Beta 2.5% HPD
                                unfoldedBCI[:,1,0],                     # Gamma 2.5% HPD
                                np.mean(self.trace.Truth,0)[:,0],       # Beta Mean
                                np.mean(self.trace.Truth,0)[:,1],       # Gamma Mean
                                unfoldedBCI[:,0,1],                     # Beta 97.5% HPD
                                unfoldedBCI[:,1,1]])                    # Gamma 97.5% HPD
        binRecoEdge = np.array([[self.response.GetXaxis().GetBinLowEdge(i+1) for i in range(0, self.response.GetNbinsX() + 1)],
                                [self.response2.GetXaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsX() + 1)]])

        # Create a figure to plot the spectrum
        figFluence, axFluence = plt.subplots(2,2, figsize=(fig_size[0]*2,fig_size[1]*1.5))
        
        # Plot the data spectrum
        axFluence[0][0].plot(sorted(np.concatenate((binDataEdge[1:],binDataEdge[:-1]))), 
                    np.repeat(binDataVal, 2),
                    lw=1.25, 
                    color='black', 
                    linestyle="-",
                    drawstyle='steps')
        
        axFluence[0][1].plot(sorted(np.concatenate((binDataEdge[1:],binDataEdge[:-1]))), 
                    np.repeat(binDataVal, 2),
                    lw=1.25, 
                    color='black', 
                    linestyle="-",
                    drawstyle='steps')

        axFluence[0][0].set_title('Measured Spectrum')
        axFluence[0][0].set_xlabel('Measured Energy (keV)' if not self.data.GetXaxis().GetTitle() else self.data.GetXaxis().GetTitle())
        axFluence[0][0].set_ylabel('Counts' if not self.data.GetYaxis().GetTitle() else self.data.GetYaxis().GetTitle())
        axFluence[0][0].set_xlim(min(binDataEdge),max(binDataEdge))
        axFluence[0][0].set_ylim(1., np.power(10, np.ceil(np.log10(np.max(binDataVal)))))
        axFluence[0][0].set_xscale('log')
        axFluence[0][0].set_yscale('log')

        axFluence[0][1].set_title('Measured Spectrum')
        axFluence[0][1].set_xlabel('Measured Energy (keV)' if not self.data.GetXaxis().GetTitle() else self.data.GetXaxis().GetTitle())
        axFluence[0][1].set_ylabel('Counts' if not self.data.GetYaxis().GetTitle() else self.data.GetYaxis().GetTitle())
        axFluence[0][1].set_xlim(min(binDataEdge),max(binDataEdge))
        axFluence[0][1].set_ylim(1., np.power(10, np.ceil(np.log10(np.max(binDataVal)))))
        axFluence[0][1].set_xscale('log')
        axFluence[0][1].set_yscale('log')

        # Plot the true fluence spectrum, if available.
        pTruthBeta, = axFluence[1][0].plot(sorted(np.concatenate((binTruthEdge[0][1:],binTruthEdge[0][:-1]))), 
                                        np.repeat(binTruthVal[0], 2),
                                        lw=1.25, 
                                        color='black', 
                                        linestyle="-", 
                                        drawstyle='steps')

        pTruthGamma, = axFluence[1][1].plot(sorted(np.concatenate((binTruthEdge[0][1:],binTruthEdge[0][:-1]))), 
                                        np.repeat(binTruthVal[1], 2),
                                        lw=1.25, 
                                        color='black', 
                                        linestyle="-", 
                                        drawstyle='steps')


        # Plot the unfolded spectrum
        pBCIBeta = axFluence[1][0].fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                            np.repeat(binRecoVal[0], 2), 
                                            np.repeat(binRecoVal[4], 2),
                                            color='red',
                                            alpha=0.4)
    
        pBCIGamma = axFluence[1][1].fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                            np.repeat(binRecoVal[1], 2), 
                                            np.repeat(binRecoVal[5], 2),
                                            color='red',
                                            alpha=0.4)

        pMeanBeta, = axFluence[1][0].plot(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                        np.repeat(binRecoVal[2], 2),
                                        lw=1.25, 
                                        color='red', 
                                        linestyle="-", 
                                        drawstyle='steps')
       

        pMeanGamma, = axFluence[1][1].plot(sorted(np.concatenate((binRecoEdge[1][1:],binRecoEdge[1][:-1]))), 
                                        np.repeat(binRecoVal[3], 2),
                                        lw=1.25, 
                                        color='red', 
                                        linestyle="-", 
                                        drawstyle='steps')

        minY = np.min(binRecoVal[binRecoVal >= 1E-2])
        maxY = np.maximum(np.max(binRecoVal[np.isfinite(binRecoVal)]), 
                          np.max(binTruthVal[np.isfinite(binTruthVal)]))

        axFluence[1][0].set_xlabel('True Energy (keV)')
        axFluence[1][0].set_title('Reconstructed Beta-ray Fluence Spectrum')
        axFluence[1][0].set_ylabel('Fluence (cm$^{-2}$)')
        axFluence[1][0].set_xscale('log')
        axFluence[1][0].set_yscale('log')
        axFluence[1][0].set_xlim(min(binRecoEdge[0]),max(binRecoEdge[0]))
        axFluence[1][0].set_ylim(np.power(10, np.floor(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        if plotTruth:
            axFluence[1][0].legend([pTruthBeta, (pBCIBeta, pMeanBeta)], ['Truth','Reconstructed (95% BCI)'], loc='best')
        else:
            axFluence[1][0].legend([(pBCIBeta, pMeanBeta)], ['Reconstructed (95% BCI)'], loc='best')

        axFluence[1][1].set_xlabel('True Energy (keV)')
        axFluence[1][1].set_title('Reconstructed Gamma-ray Fluence Spectrum')
        axFluence[1][1].set_ylabel('Fluence (cm$^{-2}$)')
        axFluence[1][1].set_xscale('log')
        axFluence[1][1].set_yscale('log')
        axFluence[1][1].set_xlim(min(binRecoEdge[1]),max(binRecoEdge[1]))
        axFluence[1][1].set_ylim(np.power(10, np.floor(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        if plotTruth:
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
    
    def plotUnfoldedDoseSpectrum(self, fName='UnfoldedDoseSpectrum.pdf', plotTruth = False):
        '''
        Function to plot the reconstructed dose spectrum after performing multidimensional Bayesian unfolding
        NOTE: This function is to be used when multiple response matrices are used in the unfolding.
        '''

        binTruthVal = np.array([[self.datatruth.GetBinContent(i+1) for i in range(0, self.datatruth.GetNbinsX())],
                                [self.datatruth2.GetBinContent(i+1) for i in range(0, self.datatruth2.GetNbinsX())]])
        binTruthEdge = np.array([[self.datatruth.GetBinLowEdge(i+1) for i in range(0, self.datatruth.GetNbinsX() + 1)],
                                 [self.datatruth2.GetBinLowEdge(i+1) for i in range(0, self.datatruth2.GetNbinsX() + 1)]])

        # Calculate the true dose values
        binTruthDoseVal = np.array([binTruthVal[0]*self.coeffBetaWB[0],        # Beta WB Dose Mean 
                                   binTruthVal[0]*self.coeffBetaEye[0],      # Beta Eye Dose Mean
                                   binTruthVal[1]*self.coeffGammaWB[0],       # Gamma WB Dose Mean
                                   binTruthVal[1]*self.coeffGammaSkin[0],     # Gamma Skin Dose Mean
                                   binTruthVal[1]*self.coeffGammaEye[0]])      # Gamma Eye Dose Mean

        print np.sum(binTruthDoseVal, axis=1)

        # Calculate and plot the 95% Bayesian credible regions for the unfolded spectrum
        unfoldedBCI = pm.stats.hpd(self.trace.Truth, alpha=0.05)

        binRecoFluenceVal = np.array([unfoldedBCI[:,0,0],                   # Beta 2.5% HPD
                                      unfoldedBCI[:,1,0],                   # Gamma 2.5% HPD
                                      np.mean(self.trace.Truth,0)[:,0],     # Beta Mean
                                      np.mean(self.trace.Truth,0)[:,1],     # Gamma Mean
                                      unfoldedBCI[:,0,1],                   # Beta 97.5% HPD
                                      unfoldedBCI[:,1,1]])                  # Gamma 97.5% HPD
        
        binRecoDoseVal = np.array([unfoldedBCI[:,0,0]*self.coeffBetaWB[0],                      # Beta WB Dose 2.5% HPD
                                   unfoldedBCI[:,0,0]*self.coeffBetaEye[0],                    # Beta Eye Dose 2.5% HPD
                                   unfoldedBCI[:,1,0]*self.coeffGammaWB[0],                     # Gamma WB Dose 2.5% HPD
                                   unfoldedBCI[:,1,0]*self.coeffGammaSkin[0],                   # Gamma Skin Dose 2.5% HPD
                                   unfoldedBCI[:,1,0]*self.coeffGammaEye[0],                    # Gamma Eye Dose 2.5% HPD
                                   np.mean(self.trace.Truth,0)[:,0]*self.coeffBetaWB[0],        # Beta WB Dose Mean 
                                   np.mean(self.trace.Truth,0)[:,0]*self.coeffBetaEye[0],      # Beta Eye Dose Mean
                                   np.mean(self.trace.Truth,0)[:,1]*self.coeffGammaWB[0],       # Gamma WB Dose Mean
                                   np.mean(self.trace.Truth,0)[:,1]*self.coeffGammaSkin[0],     # Gamma Skin Dose Mean
                                   np.mean(self.trace.Truth,0)[:,1]*self.coeffGammaEye[0],      # Gamma Eye Dose Mean
                                   unfoldedBCI[:,0,1]*self.coeffBetaWB[0],                      # Beta WB Dose 97.5% HPD
                                   unfoldedBCI[:,0,1]*self.coeffBetaEye[0],                    # Beta Eye Dose 97.5% HPD
                                   unfoldedBCI[:,1,1]*self.coeffGammaWB[0],                     # Gamma WB Dose 97.5% HPD
                                   unfoldedBCI[:,1,1]*self.coeffGammaSkin[0],                   # Gamma Skin Dose 97.5% HPD
                                   unfoldedBCI[:,1,1]*self.coeffGammaEye[0]])                   # Gamma Eye Dose 97.5% HPD

        binRecoEdge = np.array([[self.response.GetXaxis().GetBinLowEdge(i+1) for i in range(0, self.response.GetNbinsX() + 1)],
                                [self.response2.GetXaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsX() + 1)]])

        # Create a figure to plot the spectrum
        figDose, axDose = plt.subplots(2,2, figsize=(fig_size[0]*2,fig_size[1]*1.5))

         # Plot the true fluence spectrum, if available.
        pTruthBeta, = axDose[0][0].plot(sorted(np.concatenate((binTruthEdge[0][1:],binTruthEdge[0][:-1]))), 
                                        np.repeat(binTruthVal[0], 2),
                                        lw=1.25, 
                                        color='black', 
                                        linestyle="-", 
                                        drawstyle='steps')

        pTruthGamma, = axDose[0][1].plot(sorted(np.concatenate((binTruthEdge[0][1:],binTruthEdge[0][:-1]))), 
                                        np.repeat(binTruthVal[1], 2),
                                        lw=1.25, 
                                        color='black', 
                                        linestyle="-", 
                                        drawstyle='steps')

        # Plot the unfolded spectrum
        pBCIBeta = axDose[0][0].fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                            np.repeat(binRecoFluenceVal[0], 2), 
                                            np.repeat(binRecoFluenceVal[4], 2),
                                            color='red',
                                            alpha=0.4)
    
        pBCIGamma = axDose[0][1].fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                            np.repeat(binRecoFluenceVal[1], 2), 
                                            np.repeat(binRecoFluenceVal[5], 2),
                                            color='red',
                                            alpha=0.4)

        pMeanBeta, = axDose[0][0].plot(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                        np.repeat(binRecoFluenceVal[2], 2),
                                        lw=1.25, 
                                        color='red', 
                                        linestyle="-", 
                                        drawstyle='steps')
       

        pMeanGamma, = axDose[0][1].plot(sorted(np.concatenate((binRecoEdge[1][1:],binRecoEdge[1][:-1]))), 
                                        np.repeat(binRecoFluenceVal[3], 2),
                                        lw=1.25, 
                                        color='red', 
                                        linestyle="-", 
                                        drawstyle='steps')

        minY = np.min(binRecoFluenceVal[binRecoFluenceVal >= 1E-2])
        maxY = np.maximum(np.max(binRecoFluenceVal[np.isfinite(binRecoFluenceVal)]), 
                          np.max(binTruthVal[np.isfinite(binTruthVal)]))

        axDose[0][0].set_title('Reconstructed Beta-ray Fluence Spectrum')
        axDose[0][0].set_xlabel('True Energy (keV)')
        axDose[0][0].set_ylabel('Fluence (cm$^{-2}$)')
        axDose[0][0].set_xscale('log')
        axDose[0][0].set_yscale('log')
        axDose[0][0].set_xlim(min(binRecoEdge[0]),max(binRecoEdge[0]))
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
        axDose[0][1].set_xlim(min(binRecoEdge[1]),max(binRecoEdge[1]))
        axDose[0][1].set_ylim(np.power(10, np.floor(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        if plotTruth:
            axDose[0][1].legend([pTruthGamma, (pBCIGamma, pMeanGamma)], ['Truth','Reconstructed (95% BCI)'], loc='best')
        else:
            axDose[0][1].legend([(pBCIGamma, pMeanGamma)], ['Reconstructed (95% BCI)'], loc='best')

        # Plot the unfolded dose spectrum
        pBCIBetaDoseWB = axDose[1][0].fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                            np.repeat(binRecoDoseVal[0], 2), 
                                            np.repeat(binRecoDoseVal[10], 2),
                                            color='blue',
                                            alpha=0.5)
        
        pBCIBetaDoseEye = axDose[1][0].fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                            np.repeat(binRecoDoseVal[1], 2), 
                                            np.repeat(binRecoDoseVal[11], 2),
                                            color='orange',
                                            alpha=0.3)
    
        pBCIGammaDoseWB = axDose[1][1].fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                            np.repeat(binRecoDoseVal[2], 2), 
                                            np.repeat(binRecoDoseVal[12], 2),
                                            color='blue',
                                            alpha=0.5)

        pBCIGammaDoseSkin = axDose[1][1].fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                            np.repeat(binRecoDoseVal[3], 2), 
                                            np.repeat(binRecoDoseVal[13], 2),
                                            color='green',
                                            alpha=0.4)
        
        pBCIGammaDoseEye = axDose[1][1].fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                            np.repeat(binRecoDoseVal[4], 2), 
                                            np.repeat(binRecoDoseVal[14], 2),
                                            color='orange',
                                            alpha=0.3)

        pMeanBetaDoseWB, = axDose[1][0].plot(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                        np.repeat(binRecoDoseVal[5], 2),
                                        lw=1.25, 
                                        color='blue', 
                                        linestyle="-", 
                                        drawstyle='steps')
        
        pMeanBetaDoseEye, = axDose[1][0].plot(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                        np.repeat(binRecoDoseVal[6], 2),
                                        lw=1.25, 
                                        color='orange', 
                                        linestyle="-", 
                                        drawstyle='steps')

        pMeanGammaDoseWB, = axDose[1][1].plot(sorted(np.concatenate((binRecoEdge[1][1:],binRecoEdge[1][:-1]))), 
                                        np.repeat(binRecoDoseVal[7], 2),
                                        lw=1.25, 
                                        color='blue', 
                                        linestyle="-", 
                                        drawstyle='steps')
        
        pMeanGammaDoseSkin, = axDose[1][1].plot(sorted(np.concatenate((binRecoEdge[1][1:],binRecoEdge[1][:-1]))), 
                                        np.repeat(binRecoDoseVal[8], 2),
                                        lw=1.25, 
                                        color='green', 
                                        linestyle="-", 
                                        drawstyle='steps')
        
        pMeanGammaDoseEye, = axDose[1][1].plot(sorted(np.concatenate((binRecoEdge[1][1:],binRecoEdge[1][:-1]))), 
                                        np.repeat(binRecoDoseVal[9], 2),
                                        lw=1.25, 
                                        color='orange', 
                                        linestyle="-", 
                                        drawstyle='steps')

        # Plot Statistics
        tblStats1 = axDose[1][0].table( cellText = (('Whole Body',
                                        '{:0.0f} nsV'.format(np.sum(binTruthDoseVal[0])),
                                        '{:0.0f} ({:0.0f}-{:0.0f}) nSv'.format(np.sum(binRecoDoseVal[5]), np.sum(binRecoDoseVal[0]), np.sum(binRecoDoseVal[10]))),
                                        ('Eye',
                                        '{:0.0f} nGy'.format(np.sum(binTruthDoseVal[1])),
                                        '{:0.0f} ({:0.0f}-{:0.0f}) nGy'.format(np.sum(binRecoDoseVal[6]), np.sum(binRecoDoseVal[1]), np.sum(binRecoDoseVal[11])))),
                            cellLoc = 'center',
                            colLabels = ['Organ', 'True Dose', 'Estimated Dose (95% BCI)'],
                            colLoc = 'center',
                            loc = 'upper left')             
        tblStats1.auto_set_column_width(0)
        tblStats1.auto_set_column_width(1)
        tblStats1.auto_set_column_width(2)
        for key, cell in tblStats1.get_celld().items():
            cell.set_linewidth(0)
        
        tblStats2 = axDose[1][1].table( cellText = (('Whole Body',
                                        '{:0.0f} nsV'.format(np.sum(binTruthDoseVal[2])),
                                        '{:0.0f} ({:0.0f}-{:0.0f}) nSv'.format(np.sum(binRecoDoseVal[7]), np.sum(binRecoDoseVal[2]), np.sum(binRecoDoseVal[12]))),
                                        ('Skin',
                                        '{:0.0f} nGy'.format(np.sum(binTruthDoseVal[3])),
                                        '{:0.0f} ({:0.0f}-{:0.0f}) nGy'.format(np.sum(binRecoDoseVal[8]), np.sum(binRecoDoseVal[3]), np.sum(binRecoDoseVal[13]))),
                                        ('Eye',
                                        '{:0.0f} nGy'.format(np.sum(binTruthDoseVal[4])),
                                        '{:0.0f} ({:0.0f}-{:0.0f}) nGy'.format(np.sum(binRecoDoseVal[9]), np.sum(binRecoDoseVal[4]), np.sum(binRecoDoseVal[14])))),
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
        axDose[1][0].set_xlim(min(binRecoEdge[0]),max(binRecoEdge[0]))
        axDose[1][0].set_ylim(np.power(10, np.ceil(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        if plotTruth:
            axDose[1][0].legend([pTruthBeta, (pBCIBetaDose, pMeanBetaDose)], ['Truth','Reconstructed (95% BCI)'],
                                bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)
        else:
            axDose[1][0].legend([(pBCIBetaDoseWB, pMeanBetaDoseWB),(pBCIBetaDoseEye, pMeanBetaDoseEye)],
                                ['Whole Body (95% BCI)','Lens of Eye (95% BCI)'],
                                bbox_to_anchor=(0., 1.02, 1., .102), ncol=2, loc=3, mode="expand", borderaxespad=0.)

        axDose[1][1].set_title('B2 - Reconstructed Gamma-ray Dose Spectrum')
        axDose[1][1].set_xlabel('True Energy (keV)')
        axDose[1][1].set_ylabel('Dose (nSv or nGy)')
        axDose[1][1].set_xscale('log')
        axDose[1][1].set_yscale('log')
        axDose[1][1].set_xlim(min(binRecoEdge[1]),max(binRecoEdge[1]))
        axDose[1][1].set_ylim(np.power(10, np.ceil(np.log10(minY))), np.power(10, np.ceil(np.log10(maxY))))
        if plotTruth:
            axDose[1][1].legend([pTruthGamma, (pBCIGammaDose, pMeanGammaDose)], ['Truth','Reconstructed (95% BCI)'],
                                bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)
        else:
            axDose[1][1].legend([(pBCIGammaDoseWB, pMeanGammaDoseWB),(pBCIGammaDoseSkin, pMeanGammaDoseSkin),(pBCIGammaDoseEye, pMeanGammaDoseEye)],
                                ['Whole Body (95% BCI)','Skin (95% BCI)','Lens of Eye (95% BCI)'],
                                bbox_to_anchor=(0., 1.02, 1., .102), ncol=3, loc=3, mode="expand", borderaxespad=0.)
        
         # Fine-tune figure 
        figDose.tight_layout()

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Unfolded plot saved to: ' + fName

        # Show the figure
        plt.close(figDose)
   
    def plotUnfolded(self, fName='UnfoldedHistogram.pdf', plotTruth = True, plotSignificance = True, plotDose = True):
        '''
        Function to plot the reconstructed spectrum from a multidimensional Bayesian unfolding
        NOTE: This function is to be used when multiple response matrices are used in the unfolding.
        '''
        binDataVal = np.array([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())])
        binDataEdge = np.array([self.data.GetBinLowEdge(i+1) for i in range(0, self.data.GetNbinsX() + 1)])

        binTruthVal = np.array([[self.datatruth.GetBinContent(i+1) for i in range(0, self.datatruth.GetNbinsX())],
                                [self.datatruth2.GetBinContent(i+1) for i in range(0, self.datatruth2.GetNbinsX())]])
        binTruthEdge = np.array([[self.datatruth.GetBinLowEdge(i+1) for i in range(0, self.datatruth.GetNbinsX() + 1)],
                                 [self.datatruth2.GetBinLowEdge(i+1) for i in range(0, self.datatruth2.GetNbinsX() + 1)]])

        binRecoVal = np.array([np.mean(self.trace.Truth,0)[:,0],
                               np.mean(self.trace.Truth,0)[:,1]])
        binRecoEdge = np.array([[self.response.GetXaxis().GetBinLowEdge(i+1) for i in range(0, self.response.GetNbinsX() + 1)],
                                [self.response2.GetXaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsX() + 1)]])
   
        # Create a figure and a gridspec to plot the spectrum
        figUnfolded = plt.figure()

        outerGS = gridspec.GridSpec(2, 1, height_ratios = [10,60]) 
        #make nested gridspecs
        dataGS = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outerGS[0], hspace = 0.)
        recoGS = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec = outerGS[1], hspace = 0., wspace = 0.15)

        # Plot the data spectrum
        axData = plt.subplot(dataGS[0])
        axData.plot(sorted(np.concatenate((binDataEdge[1:],binDataEdge[:-1]))), np.repeat(binDataVal, 2), lw=1.25, color='black', linestyle="-", drawstyle='steps')

        axData.set_title('Measured Spectrum')
        axData.set_xlabel('Measured Energy (keV)' if not self.data.GetXaxis().GetTitle() else self.data.GetXaxis().GetTitle())
        axData.set_ylabel('Counts' if not self.data.GetYaxis().GetTitle() else self.data.GetYaxis().GetTitle())
        axData.set_xlim(min(binDataEdge),max(binDataEdge))
        axData.set_ylim(1., np.power(10, np.ceil(np.log10(np.max(binDataVal)))))
        axData.set_xscale('log')
        axData.set_yscale('log')

        # Plot the unfolded spectrum
        axUnfolded = plt.subplot(recoGS[0,0])
        axUnfolded2 = plt.subplot(recoGS[0,1], sharey = axUnfolded)

        # Calculate and plot the 95% Bayesian credible regions for the unfolded spectrum
        unfoldedCR = pm.stats.hpd(self.trace.Truth, alpha=0.05)
        pUnfoldedCR = axUnfolded.fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                            np.repeat(unfoldedCR[:,0,0], 2), 
                                            np.repeat(unfoldedCR[:,0,1], 2),
                                            color='red',
                                            alpha=0.6)
    
        pUnfoldedCR2 = axUnfolded2.fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                            np.repeat(unfoldedCR[:,1,0], 2), 
                                            np.repeat(unfoldedCR[:,1,1], 2),
                                            color='red',
                                            alpha=0.6)

        # Plot the mean of the Bayesian posterior PDF
        pUnfoldedMean, = axUnfolded.plot(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                        np.repeat(binRecoVal[0], 2),
                                        lw=1.25, 
                                        color='red', 
                                        linestyle="-", 
                                        drawstyle='steps')
       

        pUnfoldedMean2, = axUnfolded2.plot(sorted(np.concatenate((binRecoEdge[1][1:],binRecoEdge[1][:-1]))), 
                                        np.repeat(binRecoVal[1], 2),
                                        lw=1.25, 
                                        color='red', 
                                        linestyle="-", 
                                        drawstyle='steps')
        
        # Plot the truth spectrum, if it exists
        if plotTruth:
            pTruth, = axUnfolded.plot(sorted(np.concatenate((binTruthEdge[0][1:],binTruthEdge[0][:-1]))), np.repeat(binTruthVal[0], 2), lw=1.25, color='black', linestyle="-", drawstyle='steps')
            pTruth2, = axUnfolded2.plot(sorted(np.concatenate((binTruthEdge[1][1:],binTruthEdge[1][:-1]))), np.repeat(binTruthVal[1], 2), lw=1.25, color='black', linestyle="-", drawstyle='steps')

        if not plotSignificance: axUnfolded.set_xlabel('True Energy (keV)')
        axUnfolded.set_title('Reconstructed Beta-ray Spectrum')
        axUnfolded.set_ylabel('Fluence (cm$^{-2}$)')
        axUnfolded.set_xscale('log')
        axUnfolded.set_yscale('log')
        axUnfolded.set_xlim(min(binRecoEdge[0]),max(binRecoEdge[0]))
        if plotTruth:
            minY = np.min(np.append(binTruthVal[0][np.nonzero(binTruthVal[0])],
                                    binTruthVal[1][np.nonzero(binTruthVal[1])]))
            maxY = np.max(np.append(binTruthVal[0][np.isfinite(binTruthVal[0])],
                                    binTruthVal[1][np.isfinite(binTruthVal[1])]))
            
            axUnfolded.legend([pTruth, (pUnfoldedCR, pUnfoldedMean)], ['Truth','Reconstructed (95% BCI)'], loc='best')
        else:
            minY = np.min(np.append(binRecoVal[0][np.nonzero(binRecoVal[0])],
                                    binRecoVal[1][np.nonzero(binRecoVal[1])]))
            if np.isclose(minY, 0.): minY = 1E-2 
            maxY = np.max(np.append(binRecoVal[0][np.isfinite(binRecoVal[0])],
                                    binRecoVal[1][np.isfinite(binRecoVal[1])]))

            axUnfolded.legend([(pUnfoldedCR, pUnfoldedMean)], ['Reconstructed (95% BCI)'], loc='best')
        
        axUnfolded.set_ylim(np.power(10, np.ceil(np.log10(minY))),
                                np.power(10, np.ceil(np.log10(maxY))))

        if not plotSignificance: axUnfolded2.set_xlabel('True Energy (keV)')
        axUnfolded2.set_title('Reconstructed Gamma-ray Spectrum')
        axUnfolded2.set_ylabel('Fluence (cm$^{-2}$)')
        axUnfolded2.set_xscale('log')
        axUnfolded2.set_yscale('log')
        axUnfolded2.set_xlim(min(binRecoEdge[1]),max(binRecoEdge[1]))
        if plotTruth:
            axUnfolded2.legend([pTruth2, (pUnfoldedCR2, pUnfoldedMean2)], ['Truth','Reconstructed (95% BCI)'], loc='best')
        else:
            axUnfolded2.legend([(pUnfoldedCR2, pUnfoldedMean2)], ['Reconstructed (95% BCI)'], loc='best')
        
        # Plot the significance between the unfolded spectrum and the true spectrum
        if plotSignificance:

            # Calculate the statistical significance using the signed zscore method
            significance = self.calcSignificance(binTruthVal[0], binRecoVal[0])
            significance2 = self.calcSignificance(binTruthVal[1], binRecoVal[1])

            axSignificance = plt.subplot(recoGS[1,0], sharex = axUnfolded)
            axSignificance2 = plt.subplot(recoGS[1,1], sharex = axUnfolded2)

            axSignificance.fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 0, np.repeat(significance, 2), where = np.repeat(significance, 2) > 0, color='red', alpha=0.2)
            axSignificance.fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 0, np.repeat(significance, 2), where = np.repeat(significance, 2) < 0, color='blue', alpha=0.2)
            axSignificance.plot(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), np.repeat(significance, 2), lw=1.25, color='black', linestyle="-", drawstyle='steps')
            
            axSignificance2.fill_between(sorted(np.concatenate((binRecoEdge[1][1:],binRecoEdge[1][:-1]))), 0, np.repeat(significance2, 2), where = np.repeat(significance2, 2) > 0, color='red', alpha=0.2)
            axSignificance2.fill_between(sorted(np.concatenate((binRecoEdge[1][1:],binRecoEdge[1][:-1]))), 0, np.repeat(significance2, 2), where = np.repeat(significance2, 2) < 0, color='blue', alpha=0.2)
            axSignificance2.plot(sorted(np.concatenate((binRecoEdge[1][1:],binRecoEdge[1][:-1]))), np.repeat(significance2, 2), lw=1.25, color='black', linestyle="-", drawstyle='steps')
            
            # Compare True and Reco using significance
            '''
            axSignificance.text(0.02, 0.95, 
                                '$\sum$ Significance: {significance_tot:g} '
                                .expandtabs().format(significance_tot=np.sum(significance)),
                                transform=axSignificance.transAxes, 
                                verticalalignment='top', 
                                fontdict={'family' : 'monospace'})
            
            axSignificance2.text(0.02, 0.95, 
                                '$\sum$ Significance: {significance_tot:g} '
                                .expandtabs().format(significance_tot=np.sum(significance2)),
                                transform=axSignificance2.transAxes, 
                                verticalalignment='top', 
                                fontdict={'family' : 'monospace'})
            '''
        
            # Figure properties
            axSignificance.set_ylabel('Significance')  
            axSignificance.set_xlabel('True Energy (keV)')
            axSignificance.set_xlim(min(binRecoEdge[0]),max(binRecoEdge[0]))
            axSignificance.set_ylim(-5,5)
            axSignificance.set_xscale('log', nonposy='clip')

            axSignificance2.set_ylabel('Significance')  
            axSignificance2.set_xlabel('True Energy (keV)')
            axSignificance2.set_xlim(min(binRecoEdge[1]),max(binRecoEdge[1]))
            axSignificance2.set_ylim(-5,5)
            axSignificance2.set_xscale('log', nonposy='clip')

            plt.setp(axUnfolded.get_xticklabels(), visible=False)
            plt.setp(axUnfolded2.get_xticklabels(), visible=False)

        # Plot the dose spectrum, if selected
        if plotDose:
            if plotSignificance:
                axDose = plt.subplot(recoGS[2,0], sharex = axSignificance)
                axDose2 = plt.subplot(recoGS[2,1], sharex = axSignificance2)
            else: 
                axDose = plt.subplot(recoGS[1,0], sharex = axUnfolded)
                axDose2 = plt.subplot(recoGS[1,1], sharex = axUnfolded2)

            if plotTruth:
                axDose2.plot(sorted(np.concatenate((self.coeffGammaWB[1][1:],self.coeffGammaWB[1][:-1]))), 
                            np.repeat(binTruthVal[1]*self.coeffGammaWB[0], 2), 
                            lw=1.25, 
                            color='black', 
                            linestyle="-",
                            drawstyle='steps')

            # Plot the mean and the 95% BCI reconstructed whole body (WB) dose
            axDose2.plot(sorted(np.concatenate((self.coeffGammaWB[1][1:],self.coeffGammaWB[1][:-1]))), 
                                        np.repeat(binRecoVal[1]*self.coeffGammaWB[0], 2),
                                        lw=1.25, 
                                        color='red', 
                                        linestyle="-", 
                                        drawstyle='steps')

            axDose2.fill_between(sorted(np.concatenate((self.coeffGammaWB[1][1:],self.coeffGammaWB[1][:-1]))), 
                                            np.repeat(unfoldedCR[:,1,0]*self.coeffGammaWB[0], 2), 
                                            np.repeat(unfoldedCR[:,1,1]*self.coeffGammaWB[0], 2),
                                            color='red',
                                            alpha=0.3)

            # Plot the mean and the 95% BCI reconstructed skin dose
            axDose2.plot(sorted(np.concatenate((self.coeffGammaSkin[1][1:],self.coeffGammaSkin[1][:-1]))), 
                                        np.repeat(binRecoVal[1]*self.coeffGammaSkin[0], 2),
                                        lw=1.25, 
                                        color='blue', 
                                        linestyle="-", 
                                        drawstyle='steps')

            axDose2.fill_between(sorted(np.concatenate((self.coeffGammaSkin[1][1:],self.coeffGammaSkin[1][:-1]))), 
                                            np.repeat(unfoldedCR[:,1,0]*self.coeffGammaSkin[0], 2), 
                                            np.repeat(unfoldedCR[:,1,1]*self.coeffGammaSkin[0], 2),
                                            color='blue',
                                            alpha=0.3)

            # Plot the mean and the 95% BCI reconstructed eye dose
            axDose2.plot(sorted(np.concatenate((self.coeffGammaEye[1][1:],self.coeffGammaEye[1][:-1]))), 
                                        np.repeat(binRecoVal[1]*self.coeffGammaEye[0], 2),
                                        lw=1.25, 
                                        color='green', 
                                        linestyle="-", 
                                        drawstyle='steps')

            axDose2.fill_between(sorted(np.concatenate((self.coeffGammaEye[1][1:],self.coeffGammaEye[1][:-1]))), 
                                            np.repeat(unfoldedCR[:,1,0]*self.coeffGammaEye[0], 2), 
                                            np.repeat(unfoldedCR[:,1,1]*self.coeffGammaEye[0], 2),
                                            color='green',
                                            alpha=0.3)

            # Plot Statistics
            axDose2.text(0.02, 0.95, 
                                'Organ \t Mean Dose (95% BCI)\
                                \nWB \t {:02.3f} ({:02.3f}-{:02.3f}) $\mu$Sv\
                                \nSkin \t {:02.3f} ({:02.3f}-{:02.3f}) $\mu$Gy\
                                \nEye \t {:02.3f} ({:02.3f}-{:02.3f}) $\mu$Gy'
                                .expandtabs()
                                .format(np.sum(binRecoVal[1]*self.coeffGammaWB[0]),
                                        np.sum(unfoldedCR[:,1,0]*self.coeffGammaWB[0]),
                                        np.sum(unfoldedCR[:,1,1]*self.coeffGammaWB[0]),
                                        np.sum(binRecoVal[1]*self.coeffGammaSkin[0]),
                                        np.sum(unfoldedCR[:,1,0]*self.coeffGammaSkin[0]),
                                        np.sum(unfoldedCR[:,1,1]*self.coeffGammaSkin[0]),
                                        np.sum(binRecoVal[1]*self.coeffGammaEye[0]),
                                        np.sum(unfoldedCR[:,1,0]*self.coeffGammaEye[0]),
                                        np.sum(unfoldedCR[:,1,1]*self.coeffGammaEye[0])),
                                transform=axDose2.transAxes, 
                                verticalalignment='top', 
                                fontdict={'family' : 'monospace'})

            # Figure properties
            axDose2.set_ylabel('Absorbed or Effective Dose ($\mu$Gy or $\mu$Sv)')  
            axDose2.set_xlabel('True Energy (keV)')
            axDose2.set_xlim(min(self.coeffGammaWB[1]),max(self.coeffGammaWB[1]))
            axDose2.set_xscale('log', nonposy='clip')
            axDose2.set_yscale('log')

        # Fine-tune figure 
        figUnfolded.tight_layout()

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Unfolded plot saved to: ' + fName

        # Show the figure
        plt.close(figUnfolded)

    # Function to plot the posterior PDF in an unfolded bin
    def plotPosteriorPDF(self, fName='PosteriorPDF.jpg', confInt = 0.95):

        for i in range(0, self.sourcetruth.GetNbinsX()):
            figPosteriorPDF, axPosteriorPDF = plt.subplots()

            # Histogram the MCMC trace
            binPosteriorPDF, edgePosteriorPDF = np.histogram(self.trace.Truth[:,i], 
                                                             bins = 'doane',
                                                             normed = True)

            # Calculate the interquartile range
            iqrPosteriorPDF = pm.stats.hpd(self.trace.Truth[:,i], alpha=(1-confInt))

            # Plot the PDF histogram
            axPosteriorPDF.plot(sorted(np.concatenate((edgePosteriorPDF[1:],edgePosteriorPDF[:-1]))), 
                                np.repeat(binPosteriorPDF, 2), 
                                lw=1.25, 
                                color='red', 
                                linestyle="-", 
                                drawstyle='steps')
            '''
            # Plot the CDF
            axPosteriorPDF.plot(np.sort(self.trace.Truth[:,i]), np.array(range(len(self.trace.Truth[:,i])))/float(len(self.trace.Truth[:,i])),
                                lw=1.25, 
                                color='red', 
                                linestyle="-")

            '''
            axPosteriorPDF.fill_between(sorted(np.concatenate((edgePosteriorPDF[1:],edgePosteriorPDF[:-1]))),
                                        0,
                                        np.repeat(binPosteriorPDF, 2),
                                        interpolate=False,
                                        where=((sorted(np.concatenate((edgePosteriorPDF[1:],edgePosteriorPDF[:-1]))) >= iqrPosteriorPDF[0]) & 
                                        (sorted(np.concatenate((edgePosteriorPDF[1:],edgePosteriorPDF[:-1]))) <= iqrPosteriorPDF[1])),
                                        color='red', 
                                        alpha=0.2)

            axPosteriorPDF.axvline(mode(np.rint(self.trace.Truth[:,i]))[0], color='b', linestyle='-')

            axPosteriorPDF.set_xscale('log', nonposy='clip')

            # Fine-tune figure 
            figPosteriorPDF.tight_layout()

            # Save the figure 
            plt.savefig(fName.split('.')[0] + '_TruthBin_' + str(i) + '.' + fName.split('.')[1], bbox_inches="tight")
            print 'Posterior PDF plot saved to: ' + fName.split('.')[0] + '_TruthBin_' + str(i) + '.' + fName.split('.')[1]

            # Show the figure
            plt.close(figPosteriorPDF)

    # Function to 
    def plotCornerPDF(self, fName="CornerPosteriorPDF.pdf"):
        import corner 

        figCornerPDF = corner.corner(self.trace.Truth[:,0:12],
                       show_titles=True, title_kwargs={"fontsize": 12})
        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Corner plot saved to: ' + fName

        # Show the figure
        plt.close(figCornerPDF)

    def plotAutocorrelation(self, fName="AutocorrelationPDF.pdf"):

        # Create a figure
        figAutoCorr, axAutoCorr = plt.subplots()

        # Plot the autocorrelation of the Truth MCMC chain
        pm.autocorrplot(self.trace, varnames=['Truth'])

        # Fine-tune figure 
        figAutoCorr.tight_layout()

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Autocorrelation plot saved to: ' + fName

        # Show the figure
        plt.close(figAutoCorr)

    def plotELBO(self, fName="ELBO.pdf"):
        '''
        Function to plot the negative of evidence lower bound (ELBO) while running variational inference

        PARAMETERS:
        ----------
        fName:  
            A string that contains the name of the file to which the ELBO plot will be saved.

        '''

        # Create a figure
        figELBO, axELBO = plt.subplots()

        # Plot the ELBO
        axELBO.plot(self.approxADVI.hist)

        # Figure properties
        axELBO.set_xlabel('Iteration')
        axELBO.set_ylabel('Evidence of Lower Bound (ELBO)')
        axELBO.set_xscale('log', nonposy='clip')
        axELBO.set_yscale('log', nonposy='clip')
        axELBO.grid(linestyle='dotted', which="both")

        # Fine-tune figure 
        figELBO.tight_layout()
        figELBO.subplots_adjust(hspace=0, wspace = 0)

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'ELBO plot saved to: ' + fName

        # Show the figure
        plt.close(figELBO)

    # Transform an array of doubles into a Theano-type array so that it can be used in the model
    def asMat(self, x):
        return np.asarray(x,dtype=theano.config.floatX)

    # Build the inference model
    def buildModel(self, data=ROOT.TH1D(), datatruth=ROOT.TH1D(), datatruth2=ROOT.TH1D(), background=ROOT.TH1D()):
        # Check the input instance type as follows: 
        # data == ROOT.TH1
        # datatruth == ROOT.TH1
        # background == ROOT.TH1
        if not isinstance(data, ROOT.TH1): raise TypeError("Data histogram must be of type ROOT.TH1")
        if not isinstance(datatruth, ROOT.TH1): raise TypeError("Data truth histogram must be of type ROOT.TH1")
        if not isinstance(datatruth2, ROOT.TH1): raise TypeError("Second data truth histogram must be of type ROOT.TH1")
        if not isinstance(background, ROOT.TH1): raise TypeError("Background histogram must be of type ROOT.TH1")

        # Copy the inputs to the object
        self.data = copy.deepcopy(data)
        self.datatruth = copy.deepcopy(datatruth)
        self.datatruth2 = copy.deepcopy(datatruth2)
        self.background = copy.deepcopy(background)

        # Run Inference
        with pm.Model() as self.model:

            # Calculate the Geometric Factor from the normalize response matrix
            GeomFactResp1 = np.array([np.sum([self.response.GetBinContent(i+1,j+1) for j in range(0, self.response.GetNbinsY())]) for i in range(0, self.response.GetNbinsX())])
            GeomFactResp2 = np.array([np.sum([self.response2.GetBinContent(i+1,j+1) for j in range(0, self.response2.GetNbinsY())]) for i in range(0, self.response2.GetNbinsX())])

            # Calculate the scaling factor for upper bounds on the prior probabilities
            ScalFact1 = GeomFactResp1/(GeomFactResp1+GeomFactResp2)
            ScalFact1[np.isclose(ScalFact1, 0)] = np.finfo(np.float64).eps
            ScalFact2 = GeomFactResp2/(GeomFactResp1+GeomFactResp2)
            ScalFact2[np.isclose(ScalFact2, 0)] = np.finfo(np.float64).eps

            # Total number of data counts
            nCounts = np.array([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())])

            # Calculate Lower Bounds
            #lBound1 = np.zeros(self.sourcetruth.GetNbinsX())
            lBound1 = 0.1*np.array([self.datatruth.GetBinContent(i+1) for i in range(0, self.datatruth.GetNbinsX())])
            #lBound2 = np.zeros(self.sourcetruth2.GetNbinsX())
            lBound2 = 0.1*np.array([self.datatruth2.GetBinContent(i+1) for i in range(0, self.datatruth2.GetNbinsX())])

            # Calculate Upper Bounds
            #uBound1 = np.ones(self.sourcetruth.GetNbinsX())*1E-1
            uBound1 = 10*np.array([self.datatruth.GetBinContent(i+1) for i in range(0, self.datatruth.GetNbinsX())])
            #uBound1 = np.ones(self.sourcetruth.GetNbinsX())*np.sum(nCounts)/(GeomFactResp1+GeomFactResp2)*ScalFact1
            #uBound1 = 100*np.ones(self.sourcetruth.GetNbinsX())*np.sum(nCounts)
            #uBound1[np.isinf(uBound1)] = np.max(uBound1[np.isfinite(uBound1)])
            uBound1[np.isclose(uBound1, 0)] = 1E-15

            #uBound2 = np.ones(self.sourcetruth.GetNbinsX())*1E-1
            uBound2 = 10*np.array([self.datatruth2.GetBinContent(i+1) for i in range(0, self.datatruth2.GetNbinsX())])
            #uBound2 = np.ones(self.sourcetruth2.GetNbinsX())*np.sum(nCounts)/(GeomFactResp1+GeomFactResp2)*ScalFact2
            #uBound2 = np.ones(self.sourcetruth2.GetNbinsX())*np.sum(nCounts)/(GeomFactResp1+GeomFactResp2)*ScalFact2
            #uBound2 = np.ones(self.sourcetruth2.GetNbinsX())*nCounts*ScalFact2
            #uBound2[np.isinf(uBound2)] = np.max(uBound2[np.isfinite(uBound2)])
            uBound2[np.isclose(uBound2, 0)] = 1E-15

            # Define the prior probability density pi(T)
            self.T = pm.Uniform('Truth', 
                                lower = np.array([lBound1,lBound2]).T,
                                upper = np.array([uBound1,uBound2]).T, 
                                shape = (self.sourcetruth.GetNbinsX(),2))

            # Define Eq.8
            # TODO: Add background & multiple response matrices/priors
            self.var_response = np.dstack((self.asMat([[self.response.GetBinContent(i+1,j+1) for i in range(0, self.response.GetNbinsX())] for j in range(0, self.response.GetNbinsY())]),
                                       self.asMat([[self.response2.GetBinContent(i+1,j+1) for i in range(0, self.response2.GetNbinsX())] for j in range(0, self.response2.GetNbinsY())])))
            #print self.var_response, self.var_response.shape
            self.R = theano.tensor.tensordot(self.var_response, self.T, axes=2)
            
            # Define the Poisson likelihood, L(D|T) in Eq. 3, for the measured data  
            self.U = pm.Poisson('Likelihood', 
                                mu = self.R, 
                                observed = theano.shared(value = np.array([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())]), borrow = False), 
                                shape = (self.data.GetNbinsX(), 1))

    

    # Samples the posterior with N toy experiments
    # Saves the toys in self.trace, the unfolded distribution mean and mode in self.hunf and self.hunf_mode respectivel
    # the sqrt of the variance in self.hunf_err
    def sampleADVI(self, iterations = 1000000, samples = 50000):
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

    def sampleNUTS(self, N = 10000, B = 10000):
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
            #step = pm.NUTS()
            step = pm.NUTS(scaling=np.power(self.model.dict_to_array(sds), 2), is_cov=True)

            # Sample
            '''
            self.trace = pm.sample(self.Samples,
                                   tune = self.Burn,
                                   start = start,
                                   step = step,
                                   chains = 1, 
                                   njobs = 1)
            '''
            
            self.trace = pm.sample(self.Samples,
                                   tune = self.Burn,
                                   start = mu,
                                   step=step)
            

            # Print a summary of the MCMC trace      
            pm.summary(self.trace)
    
    def sampleMH(self, N = 10000, B = 10000):
        '''
        Function to sample the posterior distribution using a Markov Chain Monte Carlo (MCMC) and the
        Metropolis Hastings algorithm in PyMC3.
        ''' 
        self.Samples = N
        self.Burn = B
        with self.model:
            # Initialize with ADVI to speedup NUTS
            # https://tdhopper.com/blog/speeding-up-pymc3-nuts-sampler/
            #print 'Initialize the sampling using ADVI ...'
            #mu, sds, elbo = pm.variational.advi(n=200000)
            #print mu, sds

            # Initialization using MAP or some other algorithm
            #print 'Finding a good starting point ...'
            #mu, sds, elbo = pm.variational.advi(n=200000)
            #start = pm.find_MAP(model = self.model)

            # Select the Posterior sampling algorithm
            print 'Sampling the posterior using Metropolis ...'
            step = pm.Metropolis()

            # Sample
            '''
            self.trace = pm.sample(self.Samples,
                                   tune = self.Burn,
                                   start = start,
                                   step = step,
                                   chains = 1, 
                                   njobs = 1)
            '''
            
            self.trace = pm.sample(self.Samples,
                                   tune = self.Burn,
                                   #start = mu,
                                   step=step)
            

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

            from pymc3.variational.callbacks import CheckParametersConvergence
            print 'Initialize the sampling using ADVI ...'
            mu, sds, elbo = pm.variational.advi(n=2000000)
            #print mu, sds

            #start = pm.find_MAP(model = self.model)

            # Select the Posterior sampling algorithm
            print 'Sampling the posterior using HMC ...'
            step = pm.HamiltonianMC(scaling=np.power(self.model.dict_to_array(sds), 2), is_cov=True)
            #step = pm.HamiltonianMC()

            # Sample
            self.trace = pm.sample(self.Samples,
                                   tune = self.Burn,
                                   start = mu,
                                   #start = start,
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
ResponseFolder = './../TestData/Saint Gobain B380 LaBr3/Response Matrix/'
#DataFolder = './../../23JAN2018 BA Unit 1 Boiler 6 Measurements/'
DataFolder = './../TestData/Saint Gobain B380 LaBr3/'
DoseCoeffFolder = './../../Dose Coefficients/'
fDoseCoeffGamma  = 'ICRP116_Photon_DoseConversionCoefficients.xlsx'
fDoseCoeffBeta  = 'ICRP116_Electron_DoseConversionCoefficients.xlsx'

# Load the response and measured data from the ROOT file
#fDataName = 'Cs137_R_35_cm_Nr_2000000000_ISO.root'
#fDataName = 'gamma_Power_10_10000_keV_alpha_-2_electron_Gauss_100_100_keV_R_35_cm_Nr_200000000_ISO.root'
fDataName = 'gamma_Power_10_10000_keV_alpha_-3_electron_Gauss_3000_500_keV_R_35_cm_Nr_200000000_ISO.root'
#fDataName = '23JAN2018_LaBr3_BA_U1_B06_CL_20cm.root'
#fDataName = '23JAN2018_LaBr3_BA_U1_B06_CL_50cm.root'
#fDataName = '23JAN2018_LaBr3_BA_U1_B06_CL_72cm.root'
#fDataName = '23JAN2018_LaBr3_BA_U1_B06_CL_170cm.root'
#fDataName = '23JAN2018_LaBr3_BA_U1_B06_HL_150cm.root'
#fDataName = '20SEP2017_LaBr3_BB_FM_Snout_30cm.root'
#fDataName = '20SEP2017_LaBr3_BB_FM_Snout_50cm.root'
#fDataName = '20SEP2017_LaBr3_BB_FM_Snout_100cm.root'
#fDataName = '20SEP2017_LaBr3_BB_FM_TailStock_30cm.root'
#fDataName = '20SEP2017_LaBr3_BB_FM_TailStock_55cm.root'
#fDataName = '20SEP2017_LaBr3_BB_FM_TailStock_100cm.root'
with ROOTFile(ResponseFolder + 'Saint Gobain B380 LaBr3.root') as fResponse:
    with ROOTFile(DataFolder + fDataName) as fData:
        # Test the class
        myBSD = PyBSD(migration = fResponse.Get('Energy Migration Matrix (Electron)'),
                        sourcetruth = fResponse.Get('Source Spectrum (Electron)'),
                        migration2 = fResponse.Get('Energy Migration Matrix (Gamma)'),
                        sourcetruth2 = fResponse.Get('Source Spectrum (Gamma)'),)

        myBSD.plotResponse(fName = ResponseFolder + 'ResponseMatrix.pdf')

        # Load the dose coefficients (NOTE: Using ICRP 116)
        myBSD.loadDoseCoeffGamma(fName = DoseCoeffFolder + fDoseCoeffGamma)
        myBSD.loadDoseCoeffBeta(fName = DoseCoeffFolder + fDoseCoeffBeta)

        # Build the model
        #myBSD.buildModel(data = fData.Get('Logarithmic Energy Spectrum'))
        myBSD.buildModel(data = fData.Get('Detector Measured Spectrum'),
                        datatruth = fData.Get('Source Spectrum (Electron)'),
                        datatruth2 = fData.Get('Source Spectrum (Gamma)'))

        # Run Variational Inference
        myBSD.sampleADVI()
        #myBSD.plotELBO(fName = DataFolder + fDataName.split('.')[0] + '_ELBO.pdf')
        myBSD.plotUnfoldedFluenceSpectrum(fName = DataFolder + fDataName.split('.')[0] + '_Fluence_ADVI.pdf', plotTruth = False)
        myBSD.plotUnfoldedDoseSpectrum(fName = DataFolder + fDataName.split('.')[0] + '_Dose_ADVI.pdf', plotTruth = False)
        #myBSD.plotUnfolded(fName = DataFolder + fDataName.split('.')[0] + '_ADVI.pdf', plotTruth = False, plotSignificance = False, plotDose = True)
        #myBSD.plotUnfolded(fName = DataFolder + fDataName.split('.')[0] + '_ADVI.pdf')

        # Run MCMC Inference
        myBSD.sampleNUTS(20000,20000)
        myBSD.plotUnfoldedFluenceSpectrum(fName = DataFolder + fDataName.split('.')[0] + '_Fluence_NUTS.pdf')
        myBSD.plotUnfoldedDoseSpectrum(fName = DataFolder + fDataName.split('.')[0] + '_Dose_NUTS.pdf', plotTruth = False)
        #myBSD.sampleHMC()
        #myBSD.plotUnfoldedFluenceSpectrum(fName = DataFolder + fDataName.split('.')[0] + '_Fluence_HMC.pdf')
        #myBSD.plotUnfoldedDoseSpectrum(fName = DataFolder + fDataName.split('.')[0] + '_Dose_HMC.pdf', plotTruth = False)
        #myBSD.sampleMH(N=1000000,B=1000000)
        #myBSD.plotUnfolded(fName = DataFolder + fDataName.split('.')[0] + '_MH.pdf', plotTruth = False, plotSignificance = False)
        #myBSD.plotUnfolded(fName = DataFolder + fDataName.split('.')[0] + '_MH.pdf')

        #myBSD.sample(N=10000,B=10000)

        # Plot data and unfolding results
        #myBSD.plotData(fName = DataFolder + fDataName.split('.')[0] + '_Data.pdf')
        #myBSD.plotAutocorrelation()
        #myBSD.plotPosteriorPDF(confInt=0.95)
        #myBSD.plotCornerPDF(fName = DataFolder + fDataName.split('.')[0] + '_Corner_ADVI.pdf')
