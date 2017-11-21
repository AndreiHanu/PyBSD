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
#from matplotlib import rcParams
from matplotlib import colors
import seaborn.apionly as sns

# Scipy
import scipy.stats as st

# Color palette library for Python
# How to choose a colour scheme for your data:
# http://earthobservatory.nasa.gov/blogs/elegantfigures/2013/08/05/subtleties-of-color-part-1-of-6/
import palettable

##########################################################################################
# Setting rcParams for publication quality graphs
'''
fig_width_pt = 246.0                    # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0      # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # Width in inches
fig_height = fig_width*golden_mean      # Height in inches
fig_size =  [fig_width, fig_height]
fig_size =  [7.3*1.75,4.2*1.75]
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
#        'text.usetex': True,
        'figure.figsize': fig_size}

# Update rcParams
rcParams.update(params)
'''

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
    def __init__(self, migration=ROOT.TH2D(), truth=ROOT.TH1D()):

        # Check the input instance type as follows:
        # migration = ROOT.TH2
        # truth = ROOT.TH1
        if not isinstance(migration, ROOT.TH2): raise TypeError("Migration matrix must be of type ROOT.TH2")
        if not isinstance(truth, ROOT.TH1): raise TypeError("Truth histogram must be of type ROOT.TH1")

        # Copy the inputs to the object
        self.migration = copy.deepcopy(migration)
        self.truth = copy.deepcopy(truth)

        # Calculate the response matrix (aka. conditional probability) using Eq. 5 from the Choudalakis paper
        # Response[i,j] = P(d = j|t = i) = P(t = i, d = j)/P(t = i)
        # Response[j|i] = M[d = j, t = i] / Truth[i]
        self.response = copy.deepcopy(migration)
        #tSum = np.sum([[self.response.GetBinContent(i+1,j+1) for i in range(0, self.response.GetNbinsX())] for j in range(0, self.response.GetNbinsY())])
        for i in range(0, self.response.GetNbinsX()):
            tSum = np.sum([self.response.GetBinContent(i+1,j+1) for j in range(0, self.response.GetNbinsY())])
            for j in range(0, self.response.GetNbinsY()):
                #self.response.SetBinContent(i+1, j+1, self.response.GetBinContent(i+1,j+1)/self.truth.GetBinContent(i+1))
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.response.SetBinContent(i+1,
                                                j+1,
                                                (self.response.GetBinContent(i+1,j+1)/tSum if np.isfinite(self.response.GetBinContent(i+1,j+1)/tSum) else 0.))

    # Function to plot the measure data histogram
    def plotData(self,  fName='DataHistogram.jpg', withErrors=False, confInt=0.995):

        # Get bin values, errors, and edges
        binVal = np.array([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())])
        binValErr = st.norm.ppf(confInt)*np.sqrt(binVal)
        binEdge = np.array([self.data.GetBinLowEdge(i+1) for i in range(0, self.data.GetNbinsX() + 1)])
        binCenter = np.array([self.data.GetBinCenter(i+1) for i in range(0, self.data.GetNbinsX())])

        # Create a figure
        figData, axData = plt.subplots()

        # Plot the data
        axData.plot(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), np.repeat(binVal, 2), lw=1.25, color='black', linestyle="-", drawstyle='steps')

        # Plot data errors if selected
        if withErrors:
            axData.fill_between(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), 
                                np.repeat(binVal - binValErr, 2), 
                                np.repeat(binVal + binValErr, 2), 
                                color='gray', 
                                alpha=0.3, 
                                label= str(confInt*100)+ '% Confidence')

        # Figure properties
        axData.set_xlabel('Measured Energy (keV)' if not self.data.GetXaxis().GetTitle() else self.data.GetXaxis().GetTitle())
        axData.set_ylabel('# of Events' if not self.data.GetYaxis().GetTitle() else self.data.GetYaxis().GetTitle())
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
        #cbar.set_label('Response' if not self.response.GetZaxis().GetTitle() else self.response.GetZaxis().GetTitle())
        cbar.set_label('Response')  

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

    # Function to plot the truth histogram
    def plotTruth(self, fName='TruthHistogram.jpg', withErrors=False, confInt=0.995):
        # Get bin values, errors, and edges
        binVal = np.array([self.truth.GetBinContent(i+1) for i in range(0, self.truth.GetNbinsX())])
        binValErr = st.norm.ppf(confInt)*np.sqrt(binVal)
        binEdge = np.array([self.truth.GetBinLowEdge(i+1) for i in range(0, self.truth.GetNbinsX() + 1)])
        binCenter = np.array([self.truth.GetBinCenter(i+1) for i in range(0, self.truth.GetNbinsX())])

        # Create a figure
        figTruth, axTruth = plt.subplots()

        # Plot the truth
        axTruth.plot(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), np.repeat(binVal, 2), lw=1.25, color='black', linestyle="-", drawstyle='steps')

        # Plot truth errors if selected
        if withErrors:
            axData.fill_between(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), 
                                np.repeat(binVal - binValErr, 2), 
                                np.repeat(binVal + binValErr, 2), 
                                color='gray', 
                                alpha=0.3, 
                                label= str(confInt*100)+ '% Confidence')

        # Figure properties
        axTruth.set_xlabel('True Energy (keV)' if not self.truth.GetXaxis().GetTitle() else self.truth.GetXaxis().GetTitle())
        axTruth.set_ylabel('# of Events' if not self.truth.GetYaxis().GetTitle() else self.truth.GetYaxis().GetTitle())
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

    '''
    The goodness of the unfolding algorithm is calculated by assuming the total # of events in each bin of the true and measured histograms 
    are Poisson distributed. This is a natural assumption of spectroscopic applications as the unfolding process cannot produce negative 
    results as they would be unphysical.

    Therefore, if the # of events in each bin is Poisson distributed, we can estimate the goodness-of-fit for the unfolding using the Poisson 
    z-score.
    '''
    def zscoresigned(self, observed, expected):
        pvalue = np.zeros(observed.size)
        for i in range(observed.size):
            if observed[i] > expected[i]:
                pvalue[i] = 1 - ROOT.Math.inc_gamma_c(observed[i], expected[i])
            else:
                pvalue[i] = ROOT.Math.inc_gamma_c(observed[i] + 1, expected[i])
        
        #Calculate the signed z-score
        with np.errstate(divide='ignore', invalid='ignore'):
            zscore = np.true_divide((observed - expected), np.sqrt(expected))
            zscore[~ np.isfinite(zscore)] = 0  # -inf inf NaN

        # Return signed z-score only if p-value < 0.5
        # See: https://arxiv.org/pdf/1111.2062.pdf
        zscore[pvalue >= 0.5] = 0.

        return zscore

    # Function to plot the unfolded spectrum
    def plotUnfolded(self, fName='UnfoldedHistogram.pdf', withErrors=False, confInt=0.995):
        # Prepare values to plot:
        # Unfolded: Mode of the PDF
        # Unfolded Error: Interquartile Range
        # Edges: From the truth distribution
        binValMean = np.mean(self.trace.Truth[:], axis = 0)
        binVal = st.mode(np.rint(self.trace.Truth[:]), axis = 0)[0].flatten()

        binValErr = st.norm.ppf(confInt)*np.std(np.rint(self.trace.Truth[:]), axis = 0)
        binValTruth = np.array([np.sum([self.datatruth.GetBinContent(i+1,j+1) for j in range(0, self.datatruth.GetNbinsY())]) for i in range(0, self.datatruth.GetNbinsX())])
        binEdge = np.array([self.truth.GetBinLowEdge(i+1) for i in range(0, self.truth.GetNbinsX() + 1)])
        binCenter = np.array([self.truth.GetBinCenter(i+1) for i in range(0, self.truth.GetNbinsX())])

        # Calculate the statistical significance using the signed zscore method
        significance = self.zscoresigned(binVal, binValTruth)

        # Create a figure
        figUnfolded, axUnfolded = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]}, sharex = True)

        # Plot the truth spectrum
        axUnfolded[0].plot(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), np.repeat(binValTruth, 2), lw=1.25, color='black', linestyle="-", drawstyle='steps', label='True')
        
        # Plot unfolded spectrum with errors if selected
        if withErrors:
            axUnfolded[0].errorbar([(binEdge[i+1]+binEdge[i])/2 for i in range(0, len(binEdge)-1)],
                            binVal, 
                            yerr=binValErr,
                            xerr=[(binEdge[i+1]-binEdge[i])/2 for i in range(0, len(binEdge)-1)],
                            capsize=0, 
                            ls='none', 
                            color='red',
                            elinewidth=1.25,
                            fmt='-',
                            label= 'Reconstructed (' + str(confInt*100) + '% Confidence)')

        else:
            axUnfolded[0].plot(binCenter, binValTruth, lw=1.25, color='red', linestyle="-", drawstyle='steps', label= 'Reconstructed')

        # Plot the signifiance between the unfolded spectrum and the true spectrum
        axUnfolded[1].fill_between(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), 
                                    0, 
                                    np.repeat(significance, 2), 
                                    where = np.repeat(significance, 2) > 0,
                                    color='red', 
                                    alpha=0.2)

        axUnfolded[1].fill_between(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), 
                                    0, 
                                    np.repeat(significance, 2), 
                                    where = np.repeat(significance, 2) < 0,
                                    color='blue', 
                                    alpha=0.2)
    
        axUnfolded[1].plot(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))),
                            np.repeat(significance, 2),
                            lw=1.25,
                            color='black',
                            linestyle="-",
                            drawstyle='steps')
        
        # Print some usefull stats on the graphs 
        axUnfolded[0].text(0.02, 0.95, 
                            'N(True): {n_true:g} \
                            \nN(Reco): {n_reco:g} $\pm$ {n_reco_err:g}'
                            .expandtabs().format(n_true=np.sum(binValTruth), \
                                                 n_reco=np.sum(binVal), \
                                                 n_reco_err=np.sqrt(np.sum(binValErr**2))), 
                            transform=axUnfolded[0].transAxes, 
                            verticalalignment='top', 
                            fontdict={'family' : 'monospace'})

        axUnfolded[1].text(0.02, 0.95, 
                              '$\sum$ Significance: {significance_tot:g} '
                              .expandtabs().format(significance_tot=np.sum(significance)),
                              transform=axUnfolded[1].transAxes, 
                              verticalalignment='top', 
                              fontdict={'family' : 'monospace'})

        # Figure properties
        #axUnfolded[0].set_xlabel('True Energy (keV)')
        axUnfolded[0].set_ylabel('# of Events')
        axUnfolded[0].set_xlim(min(binEdge),max(binEdge))
        axUnfolded[0].set_ylim(1E0, 
                            np.power(10, np.ceil(np.log10(np.max(binValTruth)))))
        axUnfolded[0].set_xscale('log', nonposy='clip')
        axUnfolded[0].set_yscale('log', nonposy='clip')
        #axUnfolded[0].grid(linestyle='dotted', which="both")

        axUnfolded[1].set_ylabel('Significance')  
        axUnfolded[1].set_xlabel('True Energy (keV)')
        axUnfolded[1].set_xlim(min(binEdge),max(binEdge))
        axUnfolded[1].set_ylim(-5,5)
        axUnfolded[1].set_xscale('log', nonposy='clip')

        # Fine-tune figure 
        figUnfolded.tight_layout()
        figUnfolded.subplots_adjust(hspace=0)

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Unfolded plot saved to: ' + fName

        # Show the figure
        plt.close(figUnfolded)

    # Function to plot the posterior PDF in an unfolded bin
    def plotPosteriorPDF(self, fName='PosteriorPDF.jpg'):

        for i in range(0, self.truth.GetNbinsX()):
            figPosteriorPDF, axPosteriorPDF = plt.subplots(2,1)

            # Plot a histogram of the PDF for this chain
            entries, bin_edges, patches = axPosteriorPDF[0].hist(self.trace.Truth[:,i], 
                                                                 bins = 'auto', 
                                                                 range = (np.floor(self.trace.Truth[:,i].min()), np.ceil(self.trace.Truth[:,i].max())),
                                                                 normed=True)

            axPosteriorPDF[0].axvline(st.mode(np.rint(self.trace.Truth[:,i]))[0], color='r', linestyle='-')
            

            # Plot the CDF
            axPosteriorPDF[1].hist(self.trace.Truth[:,i], bins='auto', normed=True, cumulative=True)

            # Fine-tune figure 
            figPosteriorPDF.tight_layout()

            # Save the figure 
            plt.savefig(fName.split('.')[0] + '_TruthBin_' + str(i) + '.' + fName.split('.')[1], bbox_inches="tight")
            print 'Posterior PDF plot saved to: ' + fName.split('.')[0] + '_TruthBin_' + str(i) + '.' + fName.split('.')[1]

            # Show the figure
            plt.close(figPosteriorPDF)

    # Transform an array of doubles into a Theano-type array so that it can be used in the model
    def asMat(self, x):
        return np.asarray(x,dtype=theano.config.floatX)

    # Set the value of positive regularization parameter (Alpha)
    def setAlpha(self, alpha):
        with self.model:
            self.var_alpha.set_value(float(alpha), borrow = False)

    # Run
    def run(self, data=ROOT.TH1D(), datatruth=ROOT.TH1D(), background=ROOT.TH1D()):
        # Check the input instance type as follows: 
        # data == ROOT.TH1
        # datatruth == ROOT.TH1
        # background == ROOT.TH1
        if not isinstance(data, ROOT.TH1): raise TypeError("Data histogram must be of type ROOT.TH1")
        if not isinstance(datatruth, ROOT.TH1): raise TypeError("Data truth histogram must be of type ROOT.TH1")
        if not isinstance(background, ROOT.TH1): raise TypeError("Background histogram must be of type ROOT.TH1")

        # Copy the inputs to the object
        self.data = copy.deepcopy(data)
        self.datatruth = copy.deepcopy(datatruth)
        self.background = copy.deepcopy(background)

        # Run Inference
        with pm.Model() as self.model:
            # Positive regularization parameter
            self.var_alpha = theano.shared(value = 1.0, borrow = False)

            # Define the prior probability density pi(T)
            '''
            self.T = pm.Uniform('Truth', 
                                0., 
                                10*np.amax([np.sum([self.datatruth.GetBinContent(i+1,j+1) for j in range(0, self.datatruth.GetNbinsY())]) for i in range(0, self.datatruth.GetNbinsX())]), 
                                shape = (self.truth.GetNbinsX()), 
                                testval = np.array([np.sum([self.datatruth.GetBinContent(i+1,j+1) for j in range(0, self.datatruth.GetNbinsY())]) for i in range(0, self.datatruth.GetNbinsX())]))
            '''
            self.T = pm.Uniform('Truth', 
                                0.,
                                10*np.amax([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())]), 
                                shape = (self.truth.GetNbinsX()))

            # Define Eq.8
            # TODO: Add background & multiple response matrices/priors
            self.var_response = theano.shared(value = self.asMat([[self.response.GetBinContent(i+1,j+1) for i in range(0, self.response.GetNbinsX())] for j in range(0, self.response.GetNbinsY())]))
            self.R = theano.tensor.dot(self.var_response, self.T)
            
            # Define the Poisson likelihood, L(D|T) in Eq. 3, for the measured data  
            self.U = pm.Poisson('Likelihood', 
                                mu = self.R, 
                                observed = theano.shared(value = np.array([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())]), borrow = False), 
                                shape = (self.data.GetNbinsX(), 1))

    # Samples the posterior with N toy experiments
    # Saves the toys in self.trace, the unfolded distribution mean and mode in self.hunf and self.hunf_mode respectivel
    # the sqrt of the variance in self.hunf_err
    def sample(self, N = 10000, B = 10000):
        self.Samples = N
        self.Burn = B
        with self.model:
            # Select the Metropolis Hastings algorithm for inference
            #step = pm.Metropolis()
            step = pm.NUTS()

            # Find the MAP estimate
            start = pm.find_MAP(model = self.model)

            # Sample
            self.trace = pm.sample(self.Samples,
                                   tune = self.Burn,
                                   step = step,
                                   start = start,
                                   chains = 1, 
                                   njobs = 1)

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
fDataName = 'electron_Gauss_1000_100_keV_R_20_cm_Nr_200000000_ISO.root'
with ROOTFile('./../TestData/electron_Uni_R_20_cm_ISO.root') as fResponse:
    with ROOTFile('./../TestData/'+fDataName) as fData:
        # Test the class
        myBSD = PyBSD(fResponse.Get('Energy Migration Matrix (Electron)'), fResponse.Get('Source Spectrum (Electron)'))

        myBSD.plotMigration()
        myBSD.plotResponse()
        myBSD.plotTruth(fName = fDataName.split('.')[0] + '_Source.pdf')

        # Run Inference
        myBSD.run(data = fData.Get('Detector Measured Spectrum'), 
                  datatruth = fData.Get('Energy Migration Matrix (Electron)'))
                  #datatruth = fData.Get('Source Spectrum (Electron)'))
        myBSD.setAlpha(1.)
        myBSD.sample(N=1000,B=1000)

        # Plot data and unfolding results
        myBSD.plotData(withErrors=True, confInt=0.95, fName = fDataName.split('.')[0] + '_Data.pdf')
        myBSD.plotUnfolded(withErrors=True, confInt=0.95, fName = fDataName.split('.')[0] + '_Unfolded.pdf')
        #myBSD.plotPosteriorPDF()
