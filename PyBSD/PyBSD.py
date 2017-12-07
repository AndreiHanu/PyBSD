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
import seaborn.apionly as sns

# Scipy
import scipy.stats as st
from scipy.stats.mstats import mode

# Color palette library for Python
# How to choose a colour scheme for your data:
# http://earthobservatory.nasa.gov/blogs/elegantfigures/2013/08/05/subtleties-of-color-part-1-of-6/
import palettable

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
        binCenter = np.array([self.data.GetBinCenter(i+1) for i in range(0, self.data.GetNbinsX())])

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
    def plotTruth(self, fName='TruthHistogram.jpg'):
        # Get bin values, errors, and edges
        binVal = np.array([self.truth.GetBinContent(i+1) for i in range(0, self.truth.GetNbinsX())])
        binEdge = np.array([self.truth.GetBinLowEdge(i+1) for i in range(0, self.truth.GetNbinsX() + 1)])
        binCenter = np.array([self.truth.GetBinCenter(i+1) for i in range(0, self.truth.GetNbinsX())])

        # Create a figure
        figTruth, axTruth = plt.subplots()

        # Plot the truth
        axTruth.plot(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), np.repeat(binVal, 2), lw=1.25, color='black', linestyle="-", drawstyle='steps')

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

    def zscoresigned(self, observed, expected):
        """
        The goodness of the unfolding algorithm is calculated by assuming the total # of events in each bin of the true and measured histograms 
        are Poisson distributed. This is a natural assumption of spectroscopic applications as the unfolding process cannot produce negative 
        results as they would be unphysical.

        Therefore, if the # of events in each bin is Poisson distributed, we can estimate the goodness-of-fit for the unfolding using the Poisson 
        z-score.
        """
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
    def plotUnfolded(self, fName='UnfoldedHistogram.pdf'):
        # Prepare values to plot
        # Unfolded Value = Mode of the Bayesian posterior PDF
        # Unfolded Uncertainty = Interquartile range of the Bayesian posterior PDF
        # Edges: From the truth distribution
        binVal = mode(np.rint(self.trace.Truth[:]), axis = 0)[0].flatten()
        binValTruth = np.array([np.sum([self.datatruth.GetBinContent(i+1,j+1) for j in range(0, self.datatruth.GetNbinsY())]) for i in range(0, self.datatruth.GetNbinsX())])
        binEdge = np.array([self.truth.GetBinLowEdge(i+1) for i in range(0, self.truth.GetNbinsX() + 1)])
        binCenter = np.array([self.truth.GetBinCenter(i+1) for i in range(0, self.truth.GetNbinsX())])

        # Calculate the statistical significance using the signed zscore method
        significance = self.zscoresigned(binVal, binValTruth)

        # Create a figure
        figUnfolded, axUnfolded = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]}, sharex = True)

        # Plot the truth spectrum
        axUnfolded[0].plot(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), np.repeat(binValTruth, 2), lw=1.25, color='black', linestyle="-", drawstyle='steps', label='True')
        
        # Calculate the Bayesian credible regions and plot it over the data
        bCR = [0.68,0.95,0.97]
        crColPal = sns.color_palette('Purples')
        iCR = 5
        for cr in bCR:
            binEdges = sorted(np.concatenate((binEdge[1:],binEdge[:-1])))
            binCR = pm.stats.hpd(self.trace.Truth[:], alpha=(1-cr))
            axUnfolded[0].fill_between(binEdges, np.repeat(binCR[:,0], 2), np.repeat(binCR[:,1], 2), 
                                       alpha=iCR/10., 
                                       color=crColPal[iCR],
                                       label=str(cr*100)+ '% CR')
            iCR -= 1

        # Plot the mode of the Bayesian posterior PDF
        axUnfolded[0].plot(sorted(np.concatenate((binEdge[1:],binEdge[:-1]))), 
                          np.repeat(binVal, 2),
                          lw=1.25, 
                          color='red', 
                          linestyle="-", 
                          drawstyle='steps', 
                          label='Unfolded (Mode)')

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
        
        # Print the unfolded quantiles
        sumBinValTruth = np.sum(binValTruth)
        sumBinValQ = np.percentile(self.trace.Truth[:], [2.5,50,97.5], axis=0)

        axUnfolded[0].text(0.02, 0.95,
                            'True: {:g}\n'.format(sumBinValTruth) + 
                            'Reco: ' + r'${0:g}^{{+{1:g}}}_{{-{2:g}}}$'
                            .format(np.sum(sumBinValQ[1,:]),
                                    np.sum(sumBinValQ[2,:]) - np.sum(sumBinValQ[1,:]),
                                    np.sum(sumBinValQ[1,:]) - np.sum(sumBinValQ[0,:])),
                            transform=axUnfolded[0].transAxes, 
                            verticalalignment='top')

        # Compare True and Reco using zscore
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
        #axUnfolded[0].legend(loc='upper right')
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
    def plotPosteriorPDF(self, fName='PosteriorPDF.jpg', confInt = 0.95):

        for i in range(0, self.truth.GetNbinsX()):
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

            axPosteriorPDF.fill_between(sorted(np.concatenate((edgePosteriorPDF[1:],edgePosteriorPDF[:-1]))),
                                        0,
                                        np.repeat(binPosteriorPDF, 2),
                                        interpolate=False,
                                        where=((sorted(np.concatenate((edgePosteriorPDF[1:],edgePosteriorPDF[:-1]))) >= iqrPosteriorPDF[0]) & 
                                        (sorted(np.concatenate((edgePosteriorPDF[1:],edgePosteriorPDF[:-1]))) <= iqrPosteriorPDF[1])),
                                        color='red', 
                                        alpha=0.2)

            axPosteriorPDF.axvline(mode(np.rint(self.trace.Truth[:,i]))[0], color='b', linestyle='-')
            

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

        figCornerPDF = corner.corner(self.trace.Truth[:,0:20],
                       show_titles=True, title_kwargs={"fontsize": 12})
        # Save the figure 
        plt.savefig(fName, bbox_inches="tight")
        print 'Corner plot saved to: ' + fName

        # Show the figure
        plt.close(figCornerPDF)

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
            #self.T = pm.Uniform('Truth', 
            #                    0.,
            #                    np.sum([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())]), 
            #                    shape = (self.truth.GetNbinsX()))

            dataMean = np.mean([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())])
            dataStd = np.std([self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX())])

            #BoundedNormal = pm.Bound(pm.Normal, lower=0.)
            self.T = pm.Bound(pm.Normal, lower=0.0)('Truth', mu=dataMean, sd=dataStd, shape=(self.truth.GetNbinsX()))
            
            #dataSum = np.sum(self.data.GetBinContent(i+1) for i in range(0, self.data.GetNbinsX()))
            #self.T = dataSum*np.array([[self.response.GetBinContent(i+1,j+1) for i in range(0, self.response.GetNbinsX())] for j in range(0, self.response.GetNbinsY())]).sum(axis=0)

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
            step = pm.Metropolis()
            #step = pm.DiscreteMetropolis()
            step = pm.NUTS()
            #step = pm.HamiltonianMC()

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
fDataName = 'I131_R_20_cm_Nr_20000000_ISO.root'
with ROOTFile('./../TestData/Response Matrix Canberra PD450-15-500AM/gamma_Uni_R_20_cm_ISO.root') as fResponse:
    with ROOTFile('./../TestData/'+fDataName) as fData:
        # Test the class
        myBSD = PyBSD(fResponse.Get('Energy Migration Matrix (Gamma)'), fResponse.Get('Source Spectrum (Gamma)'))

        myBSD.plotMigration()
        myBSD.plotResponse()
        myBSD.plotTruth(fName = fDataName.split('.')[0] + '_Source.pdf')

        # Run Inference
        myBSD.run(data = fData.Get('Detector Measured Spectrum'), 
                  datatruth = fData.Get('Energy Migration Matrix (Gamma)'))

        myBSD.setAlpha(1.)
        myBSD.sample(N=100000,B=10000)

        # Plot data and unfolding results
        myBSD.plotData(fName = fDataName.split('.')[0] + '_Data.pdf')
        myBSD.plotUnfolded(fName = fDataName.split('.')[0] + '_Unfolded.pdf')
        #myBSD.plotPosteriorPDF(confInt=0.95)
        #myBSD.plotCornerPDF()
