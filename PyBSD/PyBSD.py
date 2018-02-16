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
import seaborn.apionly as sns
from matplotlib import rcParams

# Scipy
import scipy.stats as st
from scipy.stats.mstats import mode

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
fig_size =  [7.3*1.75,4.2*1.75]
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
    def __init__(self, migration=ROOT.TH2D(), sourcetruth=ROOT.TH1D(), migration2=ROOT.TH2D(), source2truth=ROOT.TH1D()):

        # Check the input instance type as follows:
        # migration = ROOT.TH2
        # truth = ROOT.TH1
        if not isinstance(migration, ROOT.TH2): raise TypeError("Migration matrix must be of type ROOT.TH2")
        if not isinstance(sourcetruth, ROOT.TH1): raise TypeError("Truth histogram for the source spectrum must be of type ROOT.TH1")
        if not isinstance(migration2, ROOT.TH2): raise TypeError("Second migration matrix must be of type ROOT.TH2")
        if not isinstance(source2truth, ROOT.TH1): raise TypeError("Second truth histogram for the source spectrum must be of type ROOT.TH1")

        # Copy the inputs to the object
        self.migration = copy.deepcopy(migration)
        self.sourcetruth = copy.deepcopy(sourcetruth)
        self.migration2 = copy.deepcopy(migration2)
        self.source2truth = copy.deepcopy(source2truth)

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
                                                (self.response2.GetBinContent(i+1,j+1)/self.source2truth.GetBinContent(i+1) if np.isfinite(self.response2.GetBinContent(i+1,j+1)/self.source2truth.GetBinContent(i+1)) else 0.))

    '''
    Function to plot the measure data histogram

    IMPORTANT NOTE: It is customary to plot data (D) with error bars equal to sqrt(D) since it is
                    assumed they come from an underlying Poisson distribution. This is a frequentist
                    approach and assumes the data has uncertainty. However, there is usually no uncertainy
                    in the number of events we counted, assuming we know how to count properly. The 
                    uncertainty is in the parameters of the underlying probability distribution function.

                    This is an important aspect of the Bayesian approach. Measured data has no uncertainty!
    '''
    def plotData(self,  fName='DataHistogram.jpg'):

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

    # Function to plot the response matrix
    def plotResponse(self,  fName='ResponseMatrix.jpg'):

        # Get bin values, errors, and edges
        binVal = np.array([[self.response.GetBinContent(i+1,j+1) for i in range(0, self.response.GetNbinsX())] for j in range(0, self.response.GetNbinsY())])
        binEdge = np.array([[self.response.GetXaxis().GetBinLowEdge(i+1) for i in range(0, self.response.GetNbinsX() + 1)],
                            [self.response.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.response.GetNbinsY() + 1)]])

        # Create a figure
        figResponse, axResponse = plt.subplots()

        # Color map
        cmap = palettable.matplotlib.Viridis_20.mpl_colormap
        cmap.set_bad(cmap(0.))
        cmap.set_over(cmap(1.)) 

        # Plot the response matrix
        X, Y = np.meshgrid(binEdge[0], binEdge[1])
        H = axResponse.pcolormesh(X, Y, binVal, norm = colors.LogNorm(), cmap = cmap, linewidth = 0, rasterized = True)

        # Set color limits for the plot
        H.set_clim(np.power(10, np.floor(np.log10(np.min(binVal[binVal>0])))), np.power(10, np.ceil(np.log10(np.max(binVal[binVal>0])))))

        # Add a colorbar
        cbar = figResponse.colorbar(H, ax=axResponse, pad = 0.01, aspect = 20., extend = 'both', spacing = 'uniform')
        #cbar.set_label('Response (cm$^2$)' if not self.response.GetZaxis().GetTitle() else self.response.GetZaxis().GetTitle())
        cbar.set_label('Response (cm$^2$)')  

        # Figure properties
        axResponse.set_xlabel('True Energy (keV)' if not self.response.GetXaxis().GetTitle() else self.response.GetXaxis().GetTitle())
        axResponse.set_ylabel('Measured Energy (keV)' if not self.response.GetYaxis().GetTitle() else self.response.GetYaxis().GetTitle())
        axResponse.set_xlim(min(binEdge[0]),max(binEdge[0]))
        axResponse.set_ylim(min(binEdge[1]),max(binEdge[1]))
        axResponse.set_xscale('log')
        axResponse.set_yscale('log')

        # Fine-tune figure 
        figResponse.tight_layout()

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Response matrix plot saved to: ' + fName

        # Show the figure
        plt.close(figResponse)

    # Function to plot the second response matrix
    def plotResponse2(self,  fName='ResponseMatrix2.jpg'):

        # Get bin values, errors, and edges
        binVal = np.array([[self.response2.GetBinContent(i+1,j+1) for i in range(0, self.response2.GetNbinsX())] for j in range(0, self.response2.GetNbinsY())])
        binEdge = np.array([[self.response2.GetXaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsX() + 1)],
                            [self.response2.GetYaxis().GetBinLowEdge(i+1) for i in range(0, self.response2.GetNbinsY() + 1)]])

        # Create a figure
        figResponse, axResponse = plt.subplots()

        # Color map
        cmap = palettable.matplotlib.Viridis_20.mpl_colormap
        cmap.set_bad(cmap(0.))
        cmap.set_over(cmap(1.)) 

        # Plot the response matrix
        X, Y = np.meshgrid(binEdge[0], binEdge[1])
        H = axResponse.pcolormesh(X, Y, binVal, norm = colors.LogNorm(), cmap = cmap, linewidth = 0, rasterized = True)

        # Set color limits for the plot
        H.set_clim(np.power(10, np.floor(np.log10(np.min(binVal[binVal>0])))), np.power(10, np.ceil(np.log10(np.max(binVal[binVal>0])))))

        # Add a colorbar
        cbar = figResponse.colorbar(H, ax=axResponse, pad = 0.01, aspect = 20., extend = 'both', spacing = 'uniform')
        #cbar.set_label('Response (cm$^2$)' if not self.response2.GetZaxis().GetTitle() else self.response2.GetZaxis().GetTitle())
        cbar.set_label('Response (cm$^2$)')  

        # Figure properties
        axResponse.set_xlabel('True Energy (keV)' if not self.response2.GetXaxis().GetTitle() else self.response2.GetXaxis().GetTitle())
        axResponse.set_ylabel('Measured Energy (keV)' if not self.response2.GetYaxis().GetTitle() else self.response2.GetYaxis().GetTitle())
        axResponse.set_xlim(min(binEdge[0]),max(binEdge[0]))
        axResponse.set_ylim(min(binEdge[1]),max(binEdge[1]))
        axResponse.set_xscale('log')
        axResponse.set_yscale('log')

        # Fine-tune figure 
        figResponse.tight_layout()

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
            
            # Calculate the z-score using the p-value
            if pvalue[i] > 1E-16:
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
   
    def plotUnfolded(self, fName='UnfoldedHistogram.pdf', plotTruth = True, plotSignificance = True):
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

        outerGS = gridspec.GridSpec(2, 1, height_ratios = [5, 10]) 
        #make nested gridspecs
        dataGS = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outerGS[0], hspace = 0.)
        recoGS = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec = outerGS[1], hspace = 0., wspace = 0.15)

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
        colPal = sns.color_palette('Purples')
        unfoldedCR = pm.stats.hpd(self.trace.Truth, alpha=0.05)
        pUnfoldedCR = axUnfolded.fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                            np.repeat(unfoldedCR[:,0,0], 2), 
                                            np.repeat(unfoldedCR[:,0,1], 2),
                                            color=colPal[-2],
                                            alpha=0.6)
    
        pUnfoldedCR2 = axUnfolded2.fill_between(sorted(np.concatenate((binRecoEdge[0][1:],binRecoEdge[0][:-1]))), 
                                            np.repeat(unfoldedCR[:,1,0], 2), 
                                            np.repeat(unfoldedCR[:,1,1], 2),
                                            color=colPal[-2],
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
            maxY = np.max(np.append(binTruthVal[0][np.isfinite(binRecoVal[0])],
                                    binTruthVal[1][np.isfinite(binRecoVal[1])]))

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

            self.var_alpha = theano.shared(value = 1.0, borrow = False)

            # Calculate the Geometric Factor from the normalize response matrix
            GeomFactResp1 = np.array([np.sum([self.response.GetBinContent(i+1,j+1) for j in range(0, self.response.GetNbinsY())]) for i in range(0, self.response.GetNbinsX())])
            GeomFactResp2 = np.array([np.sum([self.response2.GetBinContent(i+1,j+1) for j in range(0, self.response2.GetNbinsY())]) for i in range(0, self.response2.GetNbinsX())])

            # Calculate the scaling factor for upper bounds on the prior probabilities
            ScalFact1 = GeomFactResp1/(GeomFactResp1+GeomFactResp2)
            ScalFact1[np.isclose(ScalFact1, 0)] = 1E-15
            #ScalFact1[ScalFact1 < 1E-5] = 1E-5
            ScalFact2 = GeomFactResp2/(GeomFactResp1+GeomFactResp2)
            ScalFact2[np.isclose(ScalFact2, 0)] = 1E-15

            # Total number of data counts
            nCounts = np.array([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())])

            # Calculate Upper Bounds
            #uBound1 = np.ones(self.sourcetruth.GetNbinsX())*np.average(nCounts)*ScalFact1
            uBound1 = np.ones(self.sourcetruth.GetNbinsX())*np.max(nCounts)/(GeomFactResp1+GeomFactResp2)*ScalFact1
            uBound1[np.isinf(uBound1)] = np.max(uBound1[np.isfinite(uBound1)])
            #uBound1 = np.ones(self.sourcetruth.GetNbinsX())*0.001

            #uBound2 = np.ones(self.sourcetruth.GetNbinsX())*np.average(nCounts)*ScalFact2
            uBound2 = np.ones(self.sourcetruth.GetNbinsX())*np.max(nCounts)/(GeomFactResp1+GeomFactResp2)*ScalFact2
            #uBound2 = np.ones(self.sourcetruth.GetNbinsX())*nCounts*ScalFact2
            uBound2[np.isinf(uBound1)] = np.max(uBound2[np.isfinite(uBound2)])

            # Define the prior probability density pi(T)
            self.T = pm.Uniform('Truth', 
                                lower = np.zeros((self.sourcetruth.GetNbinsX(),2)),
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

# Load the response and measured data from the ROOT file
#fDataName = 'gamma_Power_10_10000_keV_alpha_-2_R_35_cm_Nr_200000000_ISO.root'
fDataName = '23JAN2018_LaBr3_BA_U1_B06_HL_150cm.root'
with ROOTFile('./../TestData/Canberra PD450-15-500AM/Response Matrix/electron_Uni_R_25_cm_ISO.root') as fResponse:
    with ROOTFile('./../TestData/Canberra PD450-15-500AM/Response Matrix/gamma_Uni_R_25_cm_ISO.root') as fResponse2:
        with ROOTFile('./../../23JAN2018 BA Unit 1 Boiler 6 Measurements/'+fDataName) as fData:
        #with ROOTFile('./../TestData/Canberra PD450-15-500AM/Gamma Power Law Spectrum/'+fDataName) as fData:
            # Test the class
            myBSD = PyBSD(migration = fResponse.Get('Energy Migration Matrix (Electron)'),
                          sourcetruth = fResponse.Get('Source Spectrum (Electron)'),
                          migration2 = fResponse2.Get('Energy Migration Matrix (Gamma)'),
                          source2truth = fResponse2.Get('Source Spectrum (Gamma)'),)

            myBSD.plotMigration()
            myBSD.plotResponse()
            myBSD.plotMigration2()
            myBSD.plotResponse2()
            #myBSD.plotGeometricFactor()
            #myBSD.plotTruth(fName = fDataName.split('.')[0] + '_Source.pdf')

            # Build the model
            myBSD.buildModel(data = fData.Get('Logarithmic Energy Spectrum'))
            #myBSD.buildModel(data = fData.Get('Detector Measured Spectrum'),
            #                datatruth = fData.Get('Source Spectrum (Electron)'),
            #                datatruth2 = fData.Get('Source Spectrum (Gamma)'))

            # Run Variational Inference
            myBSD.sampleADVI()
            #myBSD.plotELBO(fName = fDataName.split('.')[0] + '_ELBO.pdf')
            myBSD.plotUnfolded(fName = fDataName.split('.')[0] + '_ADVI.pdf', plotTruth = False, plotSignificance = False)
            #myBSD.plotUnfolded(fName = fDataName.split('.')[0] + '_ADVI.pdf')

            # Run MCMC Inference
            #myBSD.sampleNUTS()
            #myBSD.plotUnfolded(fName = fDataName.split('.')[0] + '_NUTS.pdf', plotTruth = False, plotSignificance = False)
            #myBSD.sampleHMC()
            #myBSD.plotUnfolded(fName = fDataName.split('.')[0] + '_HMC.pdf')
            #myBSD.sampleMH(N=1000000,B=1000000)
            #myBSD.plotUnfolded(fName = fDataName.split('.')[0] + '_MH.pdf', plotTruth = False, plotSignificance = False)
            #myBSD.plotUnfolded(fName = fDataName.split('.')[0] + '_MH.pdf')

            #myBSD.sample(N=10000,B=10000)

            # Plot data and unfolding results
            #myBSD.plotData(fName = fDataName.split('.')[0] + '_Data.pdf')
            #myBSD.plotAutocorrelation()
            #myBSD.plotPosteriorPDF(confInt=0.95)
            #myBSD.plotCornerPDF(fName = fDataName.split('.')[0] + '_Corner.pdf')
