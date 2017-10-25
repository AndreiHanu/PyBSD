import numpy as np
import pymc3 as pm
import ROOT 

# Matplotlib - 2D plotting library
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import colors

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

    def __init__(self, data=ROOT.TH1D(), migration=ROOT.TH2D(), truth=ROOT.TH1D()):

        # Input Parameters
        self.data = data
        self.migration = migration
        self.truth = truth

    def plotData(self, fName, withErrors = False):
        # Check the data type is a TH1 instance
        if not isinstance(self.data, ROOT.TH1):
            raise TypeError("Data must be an instance of ROOT.TH1")

        # Get bin values, errors, and edges
        binVal = np.array([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())])
        binValErr = np.array([self.data.GetBinError(i+1) for i in range(0, self.data.GetNbinsX())])
        binEdge = np.array([self.data.GetBinLowEdge(i+1) for i in range(0, self.data.GetNbinsX() + 1)])
        binCenter = np.array([self.data.GetBinCenter(i+1) for i in range(0, self.data.GetNbinsX())])

        # Create a figure
        figData, axData = plt.subplots()

        # Plot the data
        axData.plot(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))),
                    np.repeat(binVal, 2),
                    lw=1.25,
                    color='black',
                    linestyle="-",
                    drawstyle='steps')

        # Plot data errors if selected
        if withErrors:
            axData.errorbar(binCenter,binVal,yerr=binValErr,capsize=0,ls='none',elinewidth=1.25, fmt='-')

        # Figure Properties
        axData.set_xlabel('True Energy (keV)' if not self.data.GetXaxis().GetTitle() else self.data.GetXaxis().GetTitle())
        axData.set_ylabel('# of Events' if not self.data.GetYaxis().GetTitle() else self.data.GetYaxis().GetTitle())
        axData.set_xlim(min(binEdge),max(binEdge))
        axData.set_ylim(np.power(10, np.floor(np.log10(np.min(binVal[binVal > 0])))), np.power(10, np.ceil(np.log10(np.max(binVal)))))
        axData.set_xscale('log')
        axData.set_yscale('log')

        # Fine-tune figure 
        figData.tight_layout()

        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Data plot saved to: ' + fName

        # Show the figure
        plt.close(figData)

# Load the data from the ROOT file
fData = ROOT.TFile.Open('./../TestData/electron_Exp_1000_keV_R_20_cm_Nr_200000000_ISO.root')

# Test the class
myBSD = PyBSD()
myBSD.data = fData.Get('Detector Measured Spectrum')
myBSD.plotData('Test.jpg')
