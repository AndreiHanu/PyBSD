import numpy as np
import pymc3 as pm
import ROOT 

# Theano
import theano
import theano.tensor

# Copy function
import copy

# Matplotlib - 2D plotting library
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import colors

# Scipy
import scipy.stats as st

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
            #tSum = np.sum([self.response.GetBinContent(i+1,j+1) for j in range(0, self.response.GetNbinsY())])
            for j in range(0, self.response.GetNbinsY()):
                self.response.SetBinContent(i+1, j+1, self.response.GetBinContent(i+1,j+1)/self.truth.GetBinContent(i+1))
                #self.response.SetBinContent(i+1, j+1, self.response.GetBinContent(i+1,j+1)/tSum)

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
    
    # Function to plot the unfolded spectrum
    def plotUnfolded(self, fName='UnfoldedHistogram.jpg', withErrors=False, confInt=0.995):
        # Get bin values, errors, and edges
        binVal = np.array([np.median(self.trace.Truth[:, i]) for i in range(0, self.truth.GetNbinsX())])
        binValTruth = np.array([self.datatruth.GetBinContent(i+1) for i in range(0, self.datatruth.GetNbinsX())])
        binValErr = st.norm.ppf(confInt)*np.sqrt(binVal)
        binEdge = np.array([self.truth.GetBinLowEdge(i+1) for i in range(0, self.truth.GetNbinsX() + 1)])
        binCenter = np.array([self.truth.GetBinCenter(i+1) for i in range(0, self.truth.GetNbinsX())])

        # Create a figure
        figUnfolded, axUnfolded = plt.subplots()

        # Plot the unfolded spectrum
        axUnfolded.plot(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), np.repeat(binVal, 2), lw=1.25, color='black', linestyle="-", drawstyle='steps')

        # Plot the data truth spectrum for comparisson
        axUnfolded.plot(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), np.repeat(binValTruth, 2), lw=1.25, color='red', linestyle="-", drawstyle='steps')
        
        # Plot unfolded spectrum errors if selected
        if withErrors:
            axUnfolded.fill_between(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), 
                                np.repeat(binVal - binValErr, 2), 
                                np.repeat(binVal + binValErr, 2), 
                                color='gray', 
                                alpha=0.3, 
                                label= str(confInt*100)+ '% Confidence')

        # Figure properties
        axUnfolded.set_xlabel('True Energy (keV)')
        axUnfolded.set_ylabel('# of Events')
        axUnfolded.set_xlim(min(binEdge),max(binEdge))
        axUnfolded.set_ylim(1E0, 
                            np.power(10, np.ceil(np.log10(np.max(binVal)))))
        axUnfolded.set_xscale('log')
        axUnfolded.set_yscale('log')

        # Fine-tune figure 
        figUnfolded.tight_layout()

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Unfolded plot saved to: ' + fName

        # Show the figure
        plt.close(figUnfolded)

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
                                10*np.amax([self.truth.GetBinContent(i+1) for i in range(0, self.truth.GetNbinsX())]), 
                                shape = (self.truth.GetNbinsX()), 
                                testval = np.array([self.truth.GetBinContent(i+1) for i in range(0, self.truth.GetNbinsX())]))

            # Define Eq.8
            # TODO: Add background & multiple response matrices/priors
            self.var_response = theano.shared(value = self.asMat([[self.response.GetBinContent(i+1,j+1) for i in range(0, self.response.GetNbinsX())] for j in range(0, self.response.GetNbinsY())]))
            self.R = theano.tensor.dot(self.var_response, self.T)
            
            # Define the Poisson likelihood, L(D|T) in Eq. 3, for the measured data  
            self.U = pm.Poisson('Unfolded', 
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
            # Use the Metropolis Hastings algorithm for inference
            step = pm.Metropolis()
            start = pm.find_MAP(model = self.model)
            self.trace = pm.sample(self.Samples,
                                   tune = self.Burn,
                                   step = step,
                                   start = start,
                                   chains = 4, 
                                   njobs = 4)
            
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
with ROOTFile('./../TestData/electron_Uni_R_20_cm_ISO.root') as fResponse:
    with ROOTFile('./../TestData/electron_Exp_100_keV_R_20_cm_Nr_200000000_ISO.root') as fData:
        # Test the class
        myBSD = PyBSD(fResponse.Get('Energy Migration Matrix (Electron)'), fResponse.Get('Source Spectrum (Electron)'))

        myBSD.plotMigration()
        myBSD.plotResponse()
        myBSD.plotTruth()

        # Run Inference
        myBSD.run(data = fData.Get('Detector Measured Spectrum'), 
                  #datatruth = fData.Get('Energy Migration Matrix (Electron)')
                  datatruth = fData.Get('Source Spectrum (Electron)'))
        myBSD.setAlpha(1.)
        myBSD.sample(N=100000,B=100000)


        # Plot data and unfolding results
        myBSD.plotData(withErrors=True, confInt=0.995)
        myBSD.plotUnfolded()

    
