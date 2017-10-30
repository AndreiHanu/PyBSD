import numpy as np
import pymc3 as pm
import ROOT 

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
        axMigration.set_xlabel('LOL Energy (keV)' if not self.migration.GetXaxis().GetTitle() else self.migration.GetXaxis().GetTitle())
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
    
    # Run
    def run(self, data=ROOT.TH1D(), background=ROOT.TH1D()):
        # Check the input instance type as follows: 
        # data == ROOT.TH1
        # background = ROOT.TH1
        if not isinstance(data, ROOT.TH1): raise TypeError("Data histogram must be of type ROOT.TH1")
        if not isinstance(background, ROOT.TH1): raise TypeError("Background histogram must be of type ROOT.TH1")

        # Copy the inputs to the object
        self.data = copy.deepcopy(data)
        self.background = copy.deepcopy(background)

# ROOT file context manager
class ROOTFile(object):

    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.file = ROOT.TFile.Open(self.filename, 'read')
        return self.file

    def __exit__(self, exception_type, exception_value, traceback):
        self.file.Close()

# Load the data from the ROOT file
with ROOTFile('./../TestData/electron_Exp_500_keV_R_100_cm_Nr_2000000000_ISO.root') as fData:
    # Test the class
    myBSD = PyBSD(fData.Get('Energy Migration Matrix (Electron)'), fData.Get('Source Spectrum (Electron)'))

    myBSD.run(fData.Get('Detector Measured Spectrum'))

    myBSD.plotData(withErrors=True, confInt=0.995)
    myBSD.plotMigration()
    myBSD.plotTruth()
