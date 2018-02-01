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
                                                
                    # Self Normalize
                    #self.response.SetBinContent(i+1,
                    #                            j+1,
                    #                            (self.response.GetBinContent(i+1,j+1)/tSum if np.isfinite(self.response.GetBinContent(i+1,j+1)/tSum) else 0.))
        
        # Calculate the second response matrix (aka. conditional probability) using Eq. 5 from the Choudalakis paper
        # Response[i,j] = P(d = j|t = i) = P(t = i, d = j)/P(t = i)
        # Response[j|i] = M[d = j, t = i] / Truth[i]
        self.response2 = copy.deepcopy(migration2)
        for i in range(0, self.response2.GetNbinsX()):
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

    # Function to plot the unfolded spectrum
    def plotUnfolded(self, fName='UnfoldedHistogram.pdf'):
        # Prepare values to plot
        # Unfolded Value = Mode of the Bayesian posterior PDF
        # Unfolded Uncertainty = Interquartile range of the Bayesian posterior PDF
        # Edges: From the truth distribution
        binVal = np.array([pm.stats.hpd(self.trace.Truth[:,i], alpha=0.5)[1] for i in range(0, self.datatruth.GetNbinsX())])
        binValTruth = np.array([self.datatruth.GetBinContent(i+1) for i in range(0, self.datatruth.GetNbinsX())])
        binEdge = np.array([self.datatruth.GetBinLowEdge(i+1) for i in range(0, self.datatruth.GetNbinsX() + 1)])
        binValData = np.array([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())])
        binEdgeData = np.array([self.data.GetBinLowEdge(i+1) for i in range(0, self.data.GetNbinsX() + 1)])

        # Estimate the significance between truth and the unfolded spectrum by
        # calculating where the truth value fits in the the CDF of the unfolded MCMC tracefrom scipy import interpolate
        from scipy import interpolate
        significance = np.array([interpolate.interp1d(np.sort(self.trace.Truth[:,i]), np.array(range(len(self.trace.Truth[:,i])))/float(len(self.trace.Truth[:,i])), fill_value='extrapolate')(binValTruth[0][i]) for i in range(0, self.datatruth.GetNbinsX())])
        #print significance, significance.shape
        significance = np.maximum(np.zeros(significance.shape) + 0.0000001, np.minimum(np.ones(significance.shape) - 0.0000001, significance))
        #print significance
        significance = -st.norm.ppf(significance)
        #print significance
        '''
        for i in range(0, len(significance)):
            # Check if we have a large or small number of samples
            if binVal[i] >= 10:
                # Apply Normal distribution model
                significance[i] = -st.norm.ppf(significance[i])
            else:
                # Apply Poisson distribution model
                significance[i] = -st.poisson.ppf(significance[i], binVal[i])
        '''
   
        # Plot the truth spectrum
        #figUnfolded, [axData,axUnfolded,axSignificance] = plt.subplots(3,2, gridspec_kw = {'height_ratios':[4,4,3]},sharex=True)
        figUnfolded = plt.figure()
        gs = gridspec.GridSpec(3, 1, height_ratios = [4,4,3]) 

        # Plot the data spectrum
        axData = plt.subplot(gs[0,0])
        axData.plot(sorted(np.concatenate((binEdgeData[1:],binEdgeData[:-1]))), np.repeat(binValData, 2), lw=1.25, color='black', linestyle="-", drawstyle='steps', label='Measured Spectrum')

        # Plot the true spectrum and the unfolded spectrum
        axUnfolded = plt.subplot(gs[1,0])
        axUnfolded.plot(sorted(np.concatenate((binEdge[0][1:],binEdge[0][:-1]))), np.repeat(binValTruth[0], 2), lw=1.25, color='black', linestyle="-", drawstyle='steps', label='True Spectrum (Electron)')

        # Calculate the Bayesian credible regions and plot it over the data
        bCR = [0.68,0.95,0.997]
        crColPal = sns.color_palette('Purples')
        iCR = 5
        for i, cr in enumerate(bCR):
            binCR = pm.stats.hpd(self.trace.Truth[:], alpha=(1-cr))
            axUnfolded.fill_between(sorted(np.concatenate((binEdge[0][1:],binEdge[0][:-1]))), 
                                    np.repeat(binCR[:,0], 2), 
                                    np.repeat(binCR[:,1], 2), 
                                    alpha=iCR/10., 
                                    color=crColPal[iCR],
                                    label=str(cr*100)+ '% CR')

            iCR -= 1

        # Plot the 50% (mean) of the Bayesian posterior PDF
        axUnfolded.plot(sorted(np.concatenate((binEdge[0][1:],binEdge[0][:-1]))), 
                          np.repeat(binVal[0], 2),
                          lw=1.25, 
                          color='red', 
                          linestyle="-", 
                          drawstyle='steps', 
                          label='Reconstructed Spectrum')

        # Plot the significance between the unfolded spectrum and the true spectrum
        axSignificance = plt.subplot(gs[2,0], sharex = axUnfolded)

        axSignificance.fill_between(sorted(np.concatenate((binEdge[0][1:],binEdge[0][:-1]))), 0, np.repeat(significance[0], 2), where = np.repeat(significance[0], 2) > 0, color='red', alpha=0.2)
        axSignificance.fill_between(sorted(np.concatenate((binEdge[0][1:],binEdge[0][:-1]))), 0, np.repeat(significance[0], 2), where = np.repeat(significance[0], 2) < 0, color='blue', alpha=0.2)
        axSignificance.plot(sorted(np.concatenate((binEdge[0][1:],binEdge[0][:-1]))), np.repeat(significance[0], 2), lw=1.25, color='black', linestyle="-", drawstyle='steps')
         
        # Print the unfolded quantiles
        '''
        sumBinValTruth = np.sum(binValTruth)
        sumBinValQ = np.percentile(self.trace.Truth[:], [2.5,50,97.5], axis=0)
        axUnfolded.text(0.02, 0.95,
                            'True: ' + r'${0:g}$'.format(np.rint(sumBinValTruth)) + 
                            '\nReco: ' + r'${0:g}^{{+{1:g}}}_{{-{2:g}}}$'
                            .format(np.rint(np.sum(sumBinValQ[1,:])),
                                    np.rint(np.sum(sumBinValQ[2,:]) - np.sum(sumBinValQ[1,:])),
                                    np.rint(np.sum(sumBinValQ[1,:]) - np.sum(sumBinValQ[0,:]))),
                            transform=axUnfolded.transAxes, 
                            verticalalignment='top')
        '''

        # Compare True and Reco using significance
        axSignificance.text(0.02, 0.95, 
                              '$\sum$ Significance: {significance_tot:g} '
                              .expandtabs().format(significance_tot=np.sum(significance[0])),
                              transform=axSignificance.transAxes, 
                              verticalalignment='top', 
                              fontdict={'family' : 'monospace'})
        
        # Figure properties
        axData.set_xlabel('Measured Energy (keV)' if not self.data.GetXaxis().GetTitle() else self.data.GetXaxis().GetTitle())
        #axData.set_ylabel('Counts' if not self.data.GetYaxis().GetTitle() else self.data.GetYaxis().GetTitle())
        axData.set_ylabel('Counts')
        axData.set_xlim(min(binEdgeData),max(binEdgeData))
        axData.set_ylim(1., np.power(10, np.ceil(np.log10(np.max(binValData)))))
        axData.set_xscale('log', nonposy='clip')
        axData.set_yscale('log', nonposy='clip')
        #axData.set_title('Measured Spectrum')
        axData.legend(loc='best')
        #axData.set_yticks(axData.get_yticks()[:-2])
        axData.xaxis.set_ticks_position('top')
        axData.xaxis.set_label_position('top') 

        #axUnfolded.set_xlabel('True Energy (keV)')
        #axUnfolded.set_ylabel('Fluence (cm$^{-2}$)' if not self.datatruth.GetYaxis().GetTitle() else self.datatruth.GetYaxis().GetTitle())
        axUnfolded.set_ylabel('Fluence (cm$^{-2}$)')
        axUnfolded.set_xlim(min(binEdge[0]),max(binEdge[0]))
        axUnfolded.set_ylim(1., np.power(10, np.ceil(np.log10(np.max(np.maximum(binValTruth[0],binValTruth[1]))))))
        axUnfolded.set_xscale('log', nonposy='clip')
        axUnfolded.set_yscale('log', nonposy='clip')
        axUnfolded.legend(loc='best')
        #axUnfolded.set_title('Reconstructed Spectrum')
        axUnfolded.set_yticks(axUnfolded.get_yticks()[:-2])
        #axUnfolded.grid(linestyle='dotted', which="both")

        axSignificance.set_ylabel('Significance')  
        axSignificance.set_xlabel('True Energy (keV)')
        axSignificance.set_xlim(min(binEdge[0]),max(binEdge[0]))
        axSignificance.set_ylim(-5,5)
        axSignificance.set_xscale('log', nonposy='clip')

        plt.setp(axUnfolded.get_xticklabels(), visible=False)
        #plt.setp(axUnfolded2.get_yticklabels(), visible=False)

        # Fine-tune figure 
        figUnfolded.tight_layout()
        figUnfolded.subplots_adjust(hspace=0.)

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Unfolded plot saved to: ' + fName

        # Show the figure
        plt.close(figUnfolded)
    
   
    # Function to plot the unfolded spectrum
    def plotUnfoldedMultiDimensionalTrace(self, fName='UnfoldedHistogram.pdf'):
        # Prepare values to plot
        # Unfolded Value = Mode of the Bayesian posterior PDF
        # Unfolded Uncertainty = Interquartile range of the Bayesian posterior PDF
        # Edges: From the truth distribution
        print self.trace.Truth.shape
        binVal = np.array([[pm.stats.hpd(self.trace.Truth[:,i,0], alpha=0.5)[1] for i in range(0, self.datatruth.GetNbinsX())],
                           [pm.stats.hpd(self.trace.Truth[:,i,1], alpha=0.5)[1] for i in range(0, self.datatruth2.GetNbinsX())]])
        binValTruth = np.array([[self.datatruth.GetBinContent(i+1) for i in range(0, self.datatruth.GetNbinsX())],
                                [self.datatruth2.GetBinContent(i+1) for i in range(0, self.datatruth2.GetNbinsX())]])
        binEdge = np.array([[self.datatruth.GetBinLowEdge(i+1) for i in range(0, self.datatruth.GetNbinsX() + 1)],
                            [self.datatruth2.GetBinLowEdge(i+1) for i in range(0, self.datatruth2.GetNbinsX() + 1)]])
        binValData = np.array([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())])
        binEdgeData = np.array([self.data.GetBinLowEdge(i+1) for i in range(0, self.data.GetNbinsX() + 1)])

        # Estimate the significance between truth and the unfolded spectrum by
        # calculating where the truth value fits in the the CDF of the unfolded MCMC tracefrom scipy import interpolate
        from scipy import interpolate
        significance = np.array([[interpolate.interp1d(np.sort(self.trace.Truth[:,i,0]), np.array(range(len(self.trace.Truth[:,i,0])))/float(len(self.trace.Truth[:,i,0])), fill_value='extrapolate')(binValTruth[0][i]) for i in range(0, self.datatruth.GetNbinsX())],
                                 [interpolate.interp1d(np.sort(self.trace.Truth[:,i,1]), np.array(range(len(self.trace.Truth[:,i,1])))/float(len(self.trace.Truth[:,i,0])), fill_value='extrapolate')(binValTruth[1][i]) for i in range(0, self.datatruth2.GetNbinsX())]])
        #print significance, significance.shape
        significance = np.maximum(np.zeros(significance.shape) + 0.0000001, np.minimum(np.ones(significance.shape) - 0.0000001, significance))
        #print significance
        significance = -st.norm.ppf(significance)
        #print significance
        '''
        for i in range(0, len(significance)):
            # Check if we have a large or small number of samples
            if binVal[i] >= 10:
                # Apply Normal distribution model
                significance[i] = -st.norm.ppf(significance[i])
            else:
                # Apply Poisson distribution model
                significance[i] = -st.poisson.ppf(significance[i], binVal[i])
        '''
   
        # Plot the truth spectrum
        #figUnfolded, [axData,axUnfolded,axSignificance] = plt.subplots(3,2, gridspec_kw = {'height_ratios':[4,4,3]},sharex=True)
        figUnfolded = plt.figure()
        gs = gridspec.GridSpec(3, 2, height_ratios = [4,4,3]) 

        # Plot the data spectrum
        axData = plt.subplot(gs[0,0])
        axData.plot(sorted(np.concatenate((binEdgeData[1:],binEdgeData[:-1]))), np.repeat(binValData, 2), lw=1.25, color='black', linestyle="-", drawstyle='steps', label='Measured Spectrum')
        axData2 = plt.subplot(gs[0,1])
        axData2.plot(sorted(np.concatenate((binEdgeData[1:],binEdgeData[:-1]))), np.repeat(binValData, 2), lw=1.25, color='black', linestyle="-", drawstyle='steps', label='Measured Spectrum')

        # Plot the true spectrum and the unfolded spectrum
        axUnfolded = plt.subplot(gs[1,0])
        axUnfolded.plot(sorted(np.concatenate((binEdge[0][1:],binEdge[0][:-1]))), np.repeat(binValTruth[0], 2), lw=1.25, color='black', linestyle="-", drawstyle='steps', label='True Spectrum (Electron)')
        axUnfolded2 = plt.subplot(gs[1,1])
        axUnfolded2.plot(sorted(np.concatenate((binEdge[1][1:],binEdge[1][:-1]))), np.repeat(binValTruth[1], 2), lw=1.25, color='black', linestyle="-", drawstyle='steps', label='True Spectrum (Gamma)')

        # Calculate the Bayesian credible regions and plot it over the data
        bCR = [0.68,0.95,0.997]
        crColPal = sns.color_palette('Purples')
        iCR = 5
        for i, cr in enumerate(bCR):
            binCR = pm.stats.hpd(self.trace.Truth[:,:,0], alpha=(1-cr))
            axUnfolded.fill_between(sorted(np.concatenate((binEdge[0][1:],binEdge[0][:-1]))), 
                                    np.repeat(binCR[:,0], 2), 
                                    np.repeat(binCR[:,1], 2), 
                                    alpha=iCR/10., 
                                    color=crColPal[iCR],
                                    label=str(cr*100)+ '% CR')
            
            binCR2 = pm.stats.hpd(self.trace.Truth[:,:,1], alpha=(1-cr))
            axUnfolded2.fill_between(sorted(np.concatenate((binEdge[1][1:],binEdge[1][:-1]))), 
                                    np.repeat(binCR2[:,0], 2), 
                                    np.repeat(binCR2[:,1], 2), 
                                    alpha=iCR/10., 
                                    color=crColPal[iCR],
                                    label=str(cr*100)+ '% CR')

            iCR -= 1

        # Plot the 50% (mean) of the Bayesian posterior PDF
        axUnfolded.plot(sorted(np.concatenate((binEdge[0][1:],binEdge[0][:-1]))), 
                          np.repeat(binVal[0], 2),
                          lw=1.25, 
                          color='red', 
                          linestyle="-", 
                          drawstyle='steps', 
                          label='Reconstructed Spectrum')
        axUnfolded2.plot(sorted(np.concatenate((binEdge[1][1:],binEdge[1][:-1]))), 
                          np.repeat(binVal[1], 2),
                          lw=1.25, 
                          color='red', 
                          linestyle="-", 
                          drawstyle='steps', 
                          label='Reconstructed Spectrum')

        # Plot the significance between the unfolded spectrum and the true spectrum
        axSignificance = plt.subplot(gs[2,0], sharex = axUnfolded)

        axSignificance.fill_between(sorted(np.concatenate((binEdge[0][1:],binEdge[0][:-1]))), 0, np.repeat(significance[0], 2), where = np.repeat(significance[0], 2) > 0, color='red', alpha=0.2)
        axSignificance.fill_between(sorted(np.concatenate((binEdge[0][1:],binEdge[0][:-1]))), 0, np.repeat(significance[0], 2), where = np.repeat(significance[0], 2) < 0, color='blue', alpha=0.2)
        axSignificance.plot(sorted(np.concatenate((binEdge[0][1:],binEdge[0][:-1]))), np.repeat(significance[0], 2), lw=1.25, color='black', linestyle="-", drawstyle='steps')
        
        axSignificance2 = plt.subplot(gs[2,1], sharex = axUnfolded2)

        axSignificance2.fill_between(sorted(np.concatenate((binEdge[1][1:],binEdge[1][:-1]))), 0, np.repeat(significance[1], 2), where = np.repeat(significance[1], 2) > 0, color='red', alpha=0.2)
        axSignificance2.fill_between(sorted(np.concatenate((binEdge[1][1:],binEdge[1][:-1]))), 0, np.repeat(significance[1], 2), where = np.repeat(significance[1], 2) < 0, color='blue', alpha=0.2)
        axSignificance2.plot(sorted(np.concatenate((binEdge[1][1:],binEdge[1][:-1]))), np.repeat(significance[1], 2), lw=1.25, color='black', linestyle="-", drawstyle='steps')
        
        # Print the unfolded quantiles
        '''
        sumBinValTruth = np.sum(binValTruth)
        sumBinValQ = np.percentile(self.trace.Truth[:], [2.5,50,97.5], axis=0)
        axUnfolded.text(0.02, 0.95,
                            'True: ' + r'${0:g}$'.format(np.rint(sumBinValTruth)) + 
                            '\nReco: ' + r'${0:g}^{{+{1:g}}}_{{-{2:g}}}$'
                            .format(np.rint(np.sum(sumBinValQ[1,:])),
                                    np.rint(np.sum(sumBinValQ[2,:]) - np.sum(sumBinValQ[1,:])),
                                    np.rint(np.sum(sumBinValQ[1,:]) - np.sum(sumBinValQ[0,:]))),
                            transform=axUnfolded.transAxes, 
                            verticalalignment='top')
        '''

        # Compare True and Reco using significance
        axSignificance.text(0.02, 0.95, 
                              '$\sum$ Significance: {significance_tot:g} '
                              .expandtabs().format(significance_tot=np.sum(significance[0])),
                              transform=axSignificance.transAxes, 
                              verticalalignment='top', 
                              fontdict={'family' : 'monospace'})
        
        axSignificance2.text(0.02, 0.95, 
                              '$\sum$ Significance: {significance_tot:g} '
                              .expandtabs().format(significance_tot=np.sum(significance[1])),
                              transform=axSignificance2.transAxes, 
                              verticalalignment='top', 
                              fontdict={'family' : 'monospace'})
        
        # Figure properties
        axData.set_xlabel('Measured Energy (keV)' if not self.data.GetXaxis().GetTitle() else self.data.GetXaxis().GetTitle())
        #axData.set_ylabel('Counts' if not self.data.GetYaxis().GetTitle() else self.data.GetYaxis().GetTitle())
        axData.set_ylabel('Counts')
        axData.set_xlim(min(binEdgeData),max(binEdgeData))
        axData.set_ylim(1., np.power(10, np.ceil(np.log10(np.max(binValData)))))
        axData.set_xscale('log', nonposy='clip')
        axData.set_yscale('log', nonposy='clip')
        #axData.set_title('Measured Spectrum')
        axData.legend(loc='best')
        #axData.set_yticks(axData.get_yticks()[:-2])
        axData.xaxis.set_ticks_position('top')
        axData.xaxis.set_label_position('top') 

        axData2.set_xlabel('Measured Energy (keV)' if not self.data.GetXaxis().GetTitle() else self.data.GetXaxis().GetTitle())
        #axData2.set_ylabel('Counts' if not self.data.GetYaxis().GetTitle() else self.data.GetYaxis().GetTitle())
        axData2.set_ylabel('Counts')
        axData2.set_xlim(min(binEdgeData),max(binEdgeData))
        axData2.set_ylim(1., np.power(10, np.ceil(np.log10(np.max(binValData)))))
        axData2.set_xscale('log', nonposy='clip')
        axData2.set_yscale('log', nonposy='clip')
        #axData2.set_title('Measured Spectrum')
        axData2.legend(loc='best')
        #axData2.set_yticks(axData.get_yticks()[:-2])
        axData2.xaxis.set_ticks_position('top')
        axData2.xaxis.set_label_position('top') 
        axData2.yaxis.set_ticks_position('right')
        axData2.yaxis.set_label_position('right') 

        #axUnfolded.set_xlabel('True Energy (keV)')
        #axUnfolded.set_ylabel('Fluence (cm$^{-2}$)' if not self.datatruth.GetYaxis().GetTitle() else self.datatruth.GetYaxis().GetTitle())
        axUnfolded.set_ylabel('Fluence (cm$^{-2}$)')
        axUnfolded.set_xlim(min(binEdge[0]),max(binEdge[0]))
        axUnfolded.set_ylim(1., np.power(10, np.ceil(np.log10(np.max(np.maximum(binValTruth[0],binValTruth[1]))))))
        axUnfolded.set_xscale('log', nonposy='clip')
        axUnfolded.set_yscale('log', nonposy='clip')
        axUnfolded.legend(loc='best')
        #axUnfolded.set_title('Reconstructed Spectrum')
        axUnfolded.set_yticks(axUnfolded.get_yticks()[:-2])
        #axUnfolded.grid(linestyle='dotted', which="both")

        #axUnfolded2.set_xlabel('True Energy (keV)')
        #axUnfolded2.set_ylabel('Fluence (cm$^{-2}$)' if not self.datatruth.GetYaxis().GetTitle() else self.datatruth.GetYaxis().GetTitle())
        axUnfolded2.set_ylabel('Fluence (cm$^{-2}$)')
        axUnfolded2.set_xlim(min(binEdge[1]),max(binEdge[1]))
        axUnfolded2.set_ylim(1., np.power(10, np.ceil(np.log10(np.max(np.maximum(binValTruth[0],binValTruth[1]))))))
        axUnfolded2.set_xscale('log', nonposy='clip')
        axUnfolded2.set_yscale('log', nonposy='clip')
        axUnfolded2.legend(loc='best')
        #axUnfolded2.set_title('Reconstructed Spectrum')
        axUnfolded2.set_yticks(axUnfolded2.get_yticks()[:-2])
        #axUnfolded2.grid(linestyle='dotted', which="both")
        axUnfolded2.yaxis.set_ticks_position('right')
        axUnfolded2.yaxis.set_label_position('right') 

        axSignificance.set_ylabel('Significance')  
        axSignificance.set_xlabel('True Energy (keV)')
        axSignificance.set_xlim(min(binEdge[0]),max(binEdge[0]))
        axSignificance.set_ylim(-5,5)
        axSignificance.set_xscale('log', nonposy='clip')

        axSignificance2.set_ylabel('Significance')  
        axSignificance2.set_xlabel('True Energy (keV)')
        axSignificance2.set_xlim(min(binEdge[1]),max(binEdge[1]))
        axSignificance2.set_ylim(-5,5)
        axSignificance2.set_xscale('log', nonposy='clip')
        axSignificance2.yaxis.set_ticks_position('right')
        axSignificance2.yaxis.set_label_position('right')

        plt.setp(axUnfolded.get_xticklabels(), visible=False)
        plt.setp(axUnfolded2.get_xticklabels(), visible=False)
        #plt.setp(axUnfolded2.get_yticklabels(), visible=False)
        #plt.setp(axSignificance2.get_yticklabels(), visible=False)

        # Fine-tune figure 
        figUnfolded.tight_layout()
        figUnfolded.subplots_adjust(hspace=0., wspace = 0.075)

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

            
            # Define upper and lower bounds for the prior probabilities
            colSum = np.array([np.sum([self.response.GetBinContent(i+1,j+1) for j in range(0, self.response.GetNbinsY())]) for i in range(0, self.response.GetNbinsX())])
            colSum[colSum == 0.] = np.min(colSum[colSum > 0.])
            colSum2 = np.array([np.sum([self.response2.GetBinContent(i+1,j+1) for j in range(0, self.response2.GetNbinsY())]) for i in range(0, self.response2.GetNbinsX())])
            colSum2[colSum2 == 0.] = np.min(colSum2[colSum2 > 0.])
            netColSum = colSum + colSum2

            print colSum/netColSum
            print colSum2/netColSum

            # Total number of data counts
            nCounts = np.sum([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())])

            # Define the prior probability density pi(T)
            self.T = pm.Uniform('Truth', 
                                lower=np.zeros((self.sourcetruth.GetNbinsX(),2)),
                                upper=np.array([np.ones(self.sourcetruth.GetNbinsX())*np.max(nCounts)/np.min(colSum)*(colSum/netColSum),
                                                np.ones(self.source2truth.GetNbinsX())*np.max(nCounts)/np.min(colSum2)*(colSum2/netColSum)]).T, 
                                shape = (self.sourcetruth.GetNbinsX(),2))
            '''
            self.T2 = pm.Uniform('Truth2', 
                                lower=np.zeros(self.source2truth.GetNbinsX()),
                                upper=np.ones(self.source2truth.GetNbinsX())*np.max(nCounts)/np.min(colSum2)*(colSum2/netColSum), 
                                shape = (self.source2truth.GetNbinsX()))
            '''

            '''
            dataMean = 10000*np.mean([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())])
            dataStd = 1000*np.std([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())])

            #BoundedNormal = pm.Bound(pm.Normal, lower=0.)
            self.T = pm.Bound(pm.Normal, lower=0.0)('Truth', mu=dataMean, sd=dataStd, shape=(self.sourcetruth.GetNbinsX()))
            self.T2 = pm.Bound(pm.Normal, lower=0.0)('Truth2', mu=dataMean, sd=dataStd, shape=(self.source2truth.GetNbinsX()))
            '''

            # Define Eq.8
            # TODO: Add background & multiple response matrices/priors
            self.var_response3 = np.dstack((self.asMat([[self.response.GetBinContent(i+1,j+1) for i in range(0, self.response.GetNbinsX())] for j in range(0, self.response.GetNbinsY())]),
                                       self.asMat([[self.response2.GetBinContent(i+1,j+1) for i in range(0, self.response2.GetNbinsX())] for j in range(0, self.response2.GetNbinsY())])))
            print self.var_response3, self.var_response3.shape
            #self.var_response = theano.shared(value = self.asMat([[self.response.GetBinContent(i+1,j+1) for i in range(0, self.response.GetNbinsX())] for j in range(0, self.response.GetNbinsY())]))
            #self.var_response2 = theano.shared(value = self.asMat([[self.response2.GetBinContent(i+1,j+1) for i in range(0, self.response2.GetNbinsX())] for j in range(0, self.response2.GetNbinsY())]))
            #self.R = theano.tensor.dot(self.var_response, self.T) + theano.tensor.dot(self.var_response2, self.T2)
            self.R = theano.tensor.tensordot(self.var_response3, self.T, axes=2)
            
            # Define the Poisson likelihood, L(D|T) in Eq. 3, for the measured data  
            self.U = pm.Poisson('Likelihood', 
                                mu = self.R, 
                                observed = theano.shared(value = np.array([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())]), borrow = False), 
                                shape = (self.data.GetNbinsX(), 1))

    # Samples the posterior with N toy experiments
    # Saves the toys in self.trace, the unfolded distribution mean and mode in self.hunf and self.hunf_mode respectivel
    # the sqrt of the variance in self.hunf_err
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
            print mu, sds

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
#fDataName = 'electron_Power_10_10000_keV_alpha_-3_R_20_cm_Nr_200000000_ISO.root'
fDataName = 'gamma_Power_10_10000_keV_alpha_-2_R_35_cm_Nr_200000000_ISO.root'
with ROOTFile('./../TestData/Response Matrix Canberra PD450-15-500AM/electron_Uni_R_25_cm_ISO.root') as fResponse:
    with ROOTFile('./../TestData/Response Matrix Canberra PD450-15-500AM/gamma_Uni_R_25_cm_ISO.root') as fResponse2:
        with ROOTFile('./../TestData/Gamma Power Law Spectrum/'+fDataName) as fData:
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
            myBSD.buildModel(data = fData.Get('Detector Measured Spectrum'), 
                    #datatruth = fData.Get('Detector True Spectrum (Electron)'))
                    datatruth = fData.Get('Source Spectrum (Electron)'),
                    datatruth2 = fData.Get('Source Spectrum (Gamma)'))

            # Run Variational Inference
            #myBSD.sampleADVI()
            #myBSD.plotELBO(fName = fDataName.split('.')[0] + '_ELBO.pdf')
            #myBSD.plotUnfolded(fName = fDataName.split('.')[0] + '_ADVI.pdf')
            #myBSD.plotUnfoldedMultiDimensionalTrace(fName = fDataName.split('.')[0] + '_ADVI.pdf')

            # Run MCMC Inference
            #myBSD.sampleNUTS()
            #myBSD.plotUnfolded(fName = fDataName.split('.')[0] + '_NUTS.pdf')
            myBSD.sampleHMC()
            myBSD.plotUnfoldedMultiDimensionalTrace(fName = fDataName.split('.')[0] + '_HMC.pdf')

            #myBSD.sample(N=10000,B=10000)

            # Plot data and unfolding results
            #myBSD.plotData(fName = fDataName.split('.')[0] + '_Data.pdf')
            #myBSD.plotUnfolded(fName = fDataName.split('.')[0] + '.pdf')
            #myBSD.plotAutocorrelation()
            #myBSD.plotPosteriorPDF(confInt=0.95)
            #myBSD.plotCornerPDF(fName = fDataName.split('.')[0] + '_Corner.pdf')
