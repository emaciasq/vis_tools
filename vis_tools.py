#!/usr/bin/env python
import numpy as np
import os
import copy
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

#-------------------------------------------------
################### CONSTANTS ####################
#-------------------------------------------------

light_speed = 2.99792458e8 # m/s

#-------------------------------------------------

class vis_obj(object):
    '''
    Class for interferometric visibilities objects.
    ATTRIBUTES:
    - u: u coordinate of visibilities (in lambdas, array).
    - v: v coordinate of visibilities (in lambdas, array).
    - r: real part of visibilities (in Jy, array).
    - i: imaginary part of visibilitites (in Jy, array).
    - wt: weights of visibilities (array).
    - uvwave: uv distance, not deprojected (in lambdas, array).
    - name: optional name for the object.
    METHODS:
    - import_vis: imports visibilities from csv or fits files.
    - deproject: deprojects visibilities using an inclination and PA.
    - bin_vis: bins the visibilities.
    - plot_vis: plots visibilities vs uvdistance (deprojected or not).
    - export_csv: exports visibilities (binned uvdistance or full
    u,v coordinates).
    - export_fits: exports visibilities (full u,v coordinates) to a fits file.
    '''
    def __init__(self,u=None,v=None,r=None,i=None,wt=None,name='',\
    input_file=None):
        '''
        INPUTS:
        Two possible ways:
        - u, v, r, i, wt: arrays that will be in the object.
        - input_file: file with the arrays. If set, it will
        call the method import_vis.
        OPTIONAL INPUTS:
        - name: optional name for the object.
        '''
        if input_file: # We read the visibilities from a file
            self.import_vis(input_file)
        else: # We create the object from the arrays provided
            if (u == None) or (v == None) or (r == None) or (i == None) or \
            (wt == None):
                raise IOError('Error in input: if input_file is not provided,'
                'u, v, r, i, and wt have to be given as inputs')
            if (len(v) != len(u)) or (len(r) != len(u)) or (len(i) != len(u)) \
            or (len(wt) != len(u)):
                raise IOError('Error in input: u, v, r, i, and wt need to be'
                '1-D arrays of the same length')
            self.u = u
            self.v = v
            self.r = r
            self.i = i
            self.wt = wt

        self.uvwave = np.sqrt(self.u**2. + self.v**2.)

        self.name = name
        # Initialization of some attributes used later
        self.bin_centers = None
        self.deproj = False # Are the binned visibilities deprojected?

    def import_vis(self,input_file):
        '''
        Imports visibilities from a csv or a fits file. Depending on the
        extension of the input file, it will choose in which format to import
        the data.
        INPUTS:
        - input_file: name of file to be imported.
        FORMAT of input files:
        - CSV file:
        It needs to have 5 columns:
                - u (lambdas)
                - v (lambdas)
                - Real part of visibilities (Jy)
                - Imaginary part of visibilities (Jy)
                - weight of visibility point.
        The data should start at the second row, with the first one being
        the names of each column.
        - FITS file:
        It should have only one extension, with an array of dimensions (5,N),
        where N is the number of visibility points. The 5 axis should be:
                - u (lambdas)
                - v (lambdas)
                - Real part of visibilities (Jy)
                - Imaginary part of visibilities (Jy)
                - weight of visibility point.
        The position of each of these axis should be enclosed in the header, in
        the UAXIS, VAXIS, RAXIS, IAXIS, and WTAXIS parameters.
        '''
        if type(input_file) is not str:
            raise IOError('input_file shoud be a string')
        if input_file[-4:] == '.csv':
            data = np.genfromtxt(input_file,delimiter=',',skip_header=1)
            u = data[:,0]
            v = data[:,1]
            r = data[:,2]
            i = data[:,3]
            wt = data[:,4]
        elif input_file[-5:] == '.fits':
            fits_file = pyfits.open(input_file)
            data = fits_file[0].data
            header = fits_file[0].header
            u = data[header['UAXIS'],:]
            v = data[header['VAXIS'],:]
            r = data[header['RAXIS'],:]
            i = data[header['IAXIS'],:]
            wt = data[header['WTAXIS'],:]

        self.u = u
        self.v = v
        self.r = r
        self.r_noshift = r
        self.i = i
        self.i_noshift = i
        self.wt = wt

    def deproject(self, inc, pa):
        '''
        Method that deprojects the visibilities using an inclination
        and position angle.
        From Zhang et al. 2016
        INPUTS:
        - inc: inclination in degrees.
        - pa: position angle in degrees (from N to E).
        OUTPUTS:
        - self.rho: uv distance of points in the deprojected plane (in lambdas,
        array).
        '''
        inc = inc * np.pi/180.
        pa = pa * np.pi/180.
        uprime = (self.u * np.cos(pa) - self.v * np.sin(pa)) * np.cos(inc)
        vprime = self.u * np.sin(pa) + self.v * np.cos(pa)
        self.rho = np.sqrt(uprime**2. + vprime**2.)

    def phase_shift(self, x_shift, y_shift):
        '''
        Method to apply a shift to the phase center.
        From Pearson 1999.
        INPUTS:
        - x_shift: Shift in RA (in marcsec)
        - y_shift: Shift in Dec (in marcsec)
        OUTPUTS:
        - Shifted real and imaginary parts.
        '''
        x_shift *= np.pi / 1000. / 3600. / 180. # To radians
        y_shift *= np.pi / 1000. / 3600. / 180.

        self.r = self.r_noshift * np.cos(-2. * np.pi * (x_shift*self.u
        + y_shift*self.v))
        self.i = self.i_noshift * np.cos(-2. * np.pi * (x_shift*self.u
        + y_shift*self.v))

    def bin_vis(self, nbins=20, deproj=True, use_wt=True):
        '''
        Method to bin the visibilities.
        INPUTS:
        - deproj: If True, bin deprojected visibilities.
        - nbins: number of bins to bin the data.
        - use_wt: If False, it will not use the weights of each visibility to
        calculate weighted means in each bin, and will do a normal average
        instead.
        OUTPUTS:
        - self.deproj: Are the binned visibilities deprojected? (boolean)
        - self.bin_centers: position of bins of visibilities (in lambdas,
        array).
        - self.r_binned: binned values of real part of visibilities (in Jy,
        array).
        - self.i_binned: binned values of imaginary part of visibilities (in Jy,
         array).
        - self.r_sigma: standard deviation of values within bins for real part
        of visibilities (in Jy, array).
        - self.i_sigma: standard deviation of values within bins for imaginary
        part of visibilities (in Jy, array).
        - self.r_err: error of the mean within bins for real part of
        visibilities (in Jy, array).
        - self.i_err: error of the mean within bins for imaginary part of
        visibilities (in Jy, array).
        '''
        if deproj:
            try:
                uvwave = self.rho
                self.deproj = True
            except:
                raise IOError('You have not deprojected the visibilities yet.'
                'Run with deproj=False or run self.deproject() first.')
        else:
            uvwave = self.uvwave
            self.deproj = False

        if use_wt:
            wt = self.wt
        else:
            wt = 1.0

        maxbin = max(uvwave)
        bin_width = maxbin/nbins
        self.bin_centers = np.ones(shape=(nbins))
        self.r_binned = np.ones(shape=(nbins))
        self.r_sigma = np.ones(shape=(nbins))
        self.r_err = np.ones(shape=(nbins))
        self.i_binned = np.ones(shape=(nbins))
        self.i_sigma = np.ones(shape=(nbins))
        self.i_err = np.ones(shape=(nbins))
        for i in range(nbins):
            self.bin_centers[i] = bin_width*i + bin_width/2.
            inbin = np.where((uvwave >= self.bin_centers[i] - bin_width/2.) &
            (uvwave < self.bin_centers[i] + bin_width/2.))
            if len(inbin[0]) > 1:
                if use_wt:
                    N_inbin = len(self.r[inbin][wt[inbin] > 0])
                else:
                    N_inbin = len(self.r[inbin])
            else:
                N_inbin = 0
            if N_inbin != 0:
                if np.isnan(self.r[inbin]).any():
                    print('WARNING: NaN in self.r')
                if np.isnan(self.i[inbin]).any():
                    print('WARNING: NaN in self.i')
                # Real part
                self.r_binned[i], wtsum = np.average(self.r[inbin],
                weights=wt[inbin], returned=True)
                if use_wt:
                    # Weighted standard deviation of sample
                    self.r_sigma[i] = np.sum(wt[inbin] * (self.r[inbin] -
                    self.r_binned[i])**2.) / ((N_inbin - 1)*wtsum / N_inbin)
                    # Error of weighted mean
                    self.r_err[i] = ( np.std(self.r[inbin][wt[inbin] > 0]) *
                    np.sqrt(np.sum(wt[inbin]**2.))/wtsum )
                else:
                    self.r_sigma[i] = np.std(self.r[inbin])
                    self.r_err[i] = self.r_sigma[i] / np.sqrt(N_inbin)
                # Imaginary part
                self.i_binned[i], wtsum = np.average(self.i[inbin],
                weights=wt[inbin], returned=True)
                if use_wt:
                    # Weighted standard deviation of sample
                    self.i_sigma[i] = np.sum(wt[inbin] * (self.i[inbin] -
                    self.i_binned[i])**2.) / ((N_inbin - 1)*wtsum / N_inbin)
                    # Error of weighted mean
                    self.i_err[i] = ( np.std(self.i[inbin][wt[inbin] > 0]) *
                    np.sqrt(np.sum(wt[inbin]**2.))/wtsum )
                else:
                    self.i_sigma[i] = np.std(self.i[inbin])
                    self.i_err[i] = self.i_sigma[i] / np.sqrt(N_inbin)
            else:
                # If there are no points inside the bin, assign NaNs.
                self.r_binned[i] = np.nan
                self.r_sigma[i] = np.nan
                self.r_err[i] = np.nan
                self.i_binned[i] = np.nan
                self.i_sigma[i] = np.nan
                self.i_err[i] = np.nan

    def plot_vis(self,real=True,imaginary=False,deproj=None,nbins=20,\
    errtype='wt',outfile='plot',overwrite=False,xlim=[],ylim=[]):
        '''
        Plots visibilities vs uvdistance (deprojected or not).
        INPUTS:
        - real: plot real part of visibilities? (boolean)
        - imaginary: plot imaginary part of visibilities? (boolean)
        - deproj: plot deprojected visibilities (if calculated)? (boolean)
        - nbins: number of bins to bin the data, if you have not already binned
        them. (integer)
        - errtype: Type of error bars used for plots. If set to 'wt' it will
        use the error calculated from the weighted mean. If not, it will use
        the std deviation in each bin. (string)
        - outfile: name (without ".pdf" extension) of the output file with the
        plot. (string)
        - overwrite: overwrite the existing plot if found? (boolean)
        - xlim, ylim: x and y axis limits for plots (in klambdas and Jy).
        '''
        deproj = deproj if deproj is not None else self.deproj
        if self.bin_centers == None:
            print('WARNING: Running bin_vis with nbins='+str(nbins))
            self.bin_vis(nbins=nbins,deproj=deproj)
        if deproj:
            if self.deproj:
                outfile = outfile+'.deproj'
            else:
                raise IOError('You set deproj=True, but your binned'
                'visibilities are not deprojected.')
        else:
            if self.deproj:
                raise IOError('You set deproj=False, but your binned'
                'visibilities are deprojected.')

        # Plot real part:
        if real:
            if errtype == 'wt':
                err = self.r_err
            else:
                err = self.r_sigma
            if (os.path.isfile(outfile+'.real_vs_uvrad.pdf') == False) \
            or (overwrite):
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                ax.set_ylabel('Real (Jy)')
                ax.set_xlabel(r'uv distance (k$\lambda$)')
                ax.errorbar(self.bin_centers/1000., self.r_binned, err,
                fmt='bo', ms=5)
                ax.axhline(y=0.0,color='k',linestyle='--')
                if xlim:
                    ax.set_xlim([xlim[0],xlim[1]])
                if ylim:
                    ax.set_ylim([ylim[0],ylim[1]])
                plt.savefig(outfile+'.real_vs_uvrad.pdf')
                plt.close(fig)
            else:
                print('WARNING, plot already exists and you do not want to'
                'overwrite it')

        # Plot imaginary part:
        if imaginary:
            if errtype == 'wt':
                err = self.i_err
            else:
                err = self.i_sigma
            if (os.path.isfile(outfile+'.imaginary_vs_uvrad.pdf') == False) \
            or (overwrite):
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                ax.set_ylabel('Imaginary (Jy)')
                ax.set_xlabel(r'uv distance (k$\lambda$)')
                ax.errorbar(self.bin_centers/1000., self.i_binned, err,
                fmt='bo', ms=5)
                ax.axhline(y=0.0,color='k',linestyle='--')
                if xlim:
                    ax.set_xlim([xlim[0],xlim[1]])
                if ylim:
                    ax.set_ylim([ylim[0],ylim[1]])
                plt.savefig(outfile+'.imaginary_vs_uvrad.pdf')
                plt.close(fig)
            else:
                print('WARNING, plot already exists and you do not want to'
                'overwrite it')

    def export_csv(self,binned=True,errtype='wt',outfile='visibilities_deproj',\
    overwrite=False):
        '''
        Method to export the visibilities to a csv file.
        INPUTS:
        - binned: Export binned visibilities vs deprojected uvdistance (True)
        or full non-deprojected u,v coordinates with weights (False)?
        - errtype: Type of error, only used if binned=True. If set to 'wt' it
        will use the error calculated from the weighted mean. If not, it will
        use the std deviation in each bin (string).
        - outfile: name (without ".csv" extension) of the output file (string).
        - overwrite: overwrite the existing file, if found? (boolean)
        '''
        if binned:
            if os.path.isfile(outfile+'_binned.csv') and (overwrite == False):
                raise IOError('EXPORT: CSV file exists and you do not want to'
                'overwrite it.')
            if self.deproj == False:
                raise IOError('EXPORT: visibilities are not deprojected.')
            if self.bin_centers == None:
                raise IOError('EXPORT: visibilities are not binned.')
            outfile = outfile + '_binned'
            if errtype == 'wt':
                err_r = self.r_err
                err_i = self.i_err
            else:
                err_r = self.r_sigma
                err_i = self.i_sigma

            f = open(outfile+'.csv','w')
            f.write('Rho (lambdas),Real(Jy),Err_Real(Jy),Imag(Jy),'
            'Err_Imag(Jy)\n')
            for i in range(len(self.bin_centers)):
                f.write('{},{},{},{},{}\n'.format(str(self.bin_centers[i]),
                str(self.r_binned[i]), str(err_r[i]), str(self.i_binned[i]),
                str(err_i[i])))
            f.close()
        else:
            if os.path.isfile(outfile+'.csv') and (overwrite == False):
                raise IOError('EXPORT: CSV file exists and you do not want to'
                'overwrite it.')
            f = open(outfile+'.csv','w')
            f.write('u(lambdas),v(lambdas),Real(Jy),Imag(Jy),Weight\n')
            for i in range(len(self.r)):
                f.write('{},{},{},{},{}\n'.format(str(self.u[i]),
                str(self.v[i]), str(self.r[i]), str(self.i[i]),
                str(self.wt[i])))
            f.close()

    def export_fits(self,outfile='visibilities_deproj',overwrite=False):
        '''
        Method to export the full set of visibilities (non-deprojected u,v
        coordinates) to a fits file.
        INPUTS:
        - outfile: name (without ".csv" extension) of the output file (string).
        - overwrite: overwrite the existing file, if found? (boolean)
        '''
        if os.path.isfile(outfile+'.fits') and (overwrite == False):
            raise IOError('EXPORT: FITS file exists and you do not want to'
            'overwrite it.')
        data = np.vstack((self.u,self.v, self.r, self.i, self.wt))

        fits_file = pyfits.PrimaryHDU(data)
        # Header
        fits_file.header.set('NAME', self.name)
        fits_file.header.set('UAXIS', 0)
        fits_file.header.set('VAXIS', 1)
        fits_file.header.set('RAXIS', 2)
        fits_file.header.set('IAXIS', 3)
        fits_file.header.set('WTAXIS', 4)

        fits_file.writeto(outfile+'.fits', clobber=overwrite)

def residual_vis(model_vis, obs_vis, spwids=[], binned = False, deproj = True):
    '''
    NOTE: This assumes that the model visibilities have been calculated from
    the uv coverage of the observed visibilities, so that the u,v coordinates
    of each point are the same. This function does not do any interpolation that
    would be needed to calculate the residuals of a more general model.
    '''
    if len(model_vis.u) != len(obsvis.u):
        raise IOError('residual_vis: model and observation visibilities have'
        'different dimensions. They probably have different spectral windows.')

    res_vis = copy.deepcopy(model_vis)
    res_vis.r = obs_vis.r - model_vis.r
    res_vis.i = obs_vis.i - model_vis.i

    if binned: # calculating residuals of binned visibilities
        if model_vis.bin_centers == None:
            print('WARNING: Running bin_vis for model_vis with default nbins.')
            model_vis.bin_vis(deproj=deproj)
        if obs_vis.bin_centers == None:
            print('WARNING: Running bin_vis for obs_vis with default nbins.')
            obs_vis.bin_vis(deproj=deproj)
        res_vis.r_binned = obs_vis.r_binned - model_vis.r_binned
        res_vis.i_binned = obs_vis.i_binned - model_vis.i_binned

    return res_vis

def plot_mod_vis(model_vis,obsvis,resvis=None,real=True,imaginary=False,
    deproj=True,errtype='wt',outfile='model',overwrite=False,normalize=False,
    xlim=[],ylim=[]):
    '''
    Function to plot visiblities of model and of observation.
    INPUTS:
    - model_vis: vis_obj object with visiblities of model.
    - obsvis: vis_obj object with visibilities of observations (if you want to
    overplot them).
    - resvis: vis_obj object with visibilities of residuals.
    - real: plot real part of visibilities?
    - imaginary: plot imaginary part of visibilities?
    - deproj: plot deprojected visibilities (if calculated)?
    - errtype: Type of error bars used for plots. If set to 'wt' it will use
    the error calculated from the weights. If not, it will use the std deviation
    in each bin.
    - outfile: name (without ".pdf" extension) of the output file with the plot.
    - overwrite: overwrite the existing plot if found?
    - xlim, ylim: x and y axis limits for plots (in klambdas and Jy).
    '''
    # Check binning of visibilities
    if model_vis.bin_centers == None:
        print('WARNING: Running bin_vis for model_vis with default parameters')
        model_vis.bin_vis(deproj=deproj)
    if obsvis.bin_centers == None:
        print('WARNING: Running bin_vis for obsvis with default parameters')
        obsvis.bin_vis(deproj=deproj)
    plotres = False
    if resvis != None:
        plotres = True
        if resvis.bin_centers == None:
            print('WARNING: Running bin_vis for resvis with default parameters')
            resvis.bin_vis(deproj=deproj)

    # Check deprojection of visibilities
    if deproj:
        if model_vis.deproj:
            outfile = outfile+'.deproj'
        else:
            raise IOError('You set deproj=True, but your model binned'
            'visibilities are not deprojected.')
        if obsvis.deproj == False:
            raise IOError('You set deproj=True, but your observed binned'
            'visibilities are not deprojected.')
        if plotres:
            if resvis.deproj == False:
                raise IOError('You set deproj=True, but your residual binned'
                'visibilities are not deprojected.')
    else:
        if model_vis.deproj:
            raise IOError('You set deproj=False, but your model binned'
            'visibilities are deprojected.')
        if obsvis.deproj:
            raise IOError('You set deproj=False, but your observed binned'
            'visibilities are deprojected.')
        if plotres:
            if resvis.deproj:
                raise IOError('You set deproj=False, but your residual binned'
                'visibilities are deprojected.')

    # Start the plotting
    if real:
        if errtype == 'wt':
            err = obsvis.r_err
        else:
            err = obsvis.r_sigma
        if normalize:
            obsr = obsvis.r_binned / np.nanmax(obsvis.r_binned[0:10])
            err = err / np.nanmax(obsvis.r_binned[0:10])
            modr = model_vis.r_binned / np.nanmax(model_vis.r_binned[0:10])
            if plotres:
                resr = resvis.r_binned / np.nanmax(resvis.r_binned[0:10])
        else:
            obsr = obsvis.r_binned
            modr = model_vis.r_binned
            if plotres:
                resr = resvis.r_binned
        if (os.path.isfile(outfile+'.real_vs_uvrad.pdf') == False) \
        or (overwrite):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            if normalize:
                ax.set_ylabel('Real')
            else:
                ax.set_ylabel('Real (Jy)')
            ax.set_xlabel(r'uv distance (k$\lambda$)')
            ax.errorbar(obsvis.bin_centers/1000.,obsr, err, fmt='bo', ms=5)
            if plotres:
                ax.errorbar(resvis.bin_centers/1000., resr, err, fmt='o',
                c='grey', ms=5)
            ax.plot(model_vis.bin_centers[np.isnan(modr)==False]/1000.,
            modr[np.isnan(modr)==False], 'r-')
            ax.axhline(y=0.0,color='k', linestyle='--')
            if xlim:
                ax.set_xlim([xlim[0],xlim[1]])
            if ylim:
                ax.set_ylim([ylim[0],ylim[1]])
            plt.savefig(outfile+'.real_vs_uvrad.pdf')
            plt.close(fig)
        else:
            print('WARNING, plot already exists and you do not want to'
            'overwrite it')

    if imaginary:
        if errtype == 'wt':
            err = obsvis.i_err
        else:
            err = obsvis.i_sigma
        # if normalize:
        #     obsi = obsvis.i_binned / np.nanmax(obsvis.i_binned)
        #     err = err / np.nanmax(obsvis.i_binned)
        #     modi = model_vis.i_binned / np.nanmax(model_vis.i_binned)
        #     if plotres:
        #         resi = resvis.i_binned / np.nanmax(resvis.i_binned)
        # else:
        obsi = obsvis.i_binned
        modi = model_vis.i_binned
        if plotres:
            resi = resvis.i_binned
        if (os.path.isfile(outfile+'.imaginary_vs_uvrad.pdf') == False) \
        or (overwrite):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            if normalize:
                ax.set_ylabel('Imaginary')
            else:
                ax.set_ylabel('Imaginary (Jy)')
            ax.set_xlabel(r'uv distance (k$\lambda$)')
            ax.errorbar(obsvis.bin_centers/1000., obsi, err, fmt='bo', ms=5)
            if plotres:
                ax.errorbar(resvis.bin_centers/1000., resi, err, fmt='ro', ms=5)
            ax.plot(model_vis.bin_centers/1000.,modi,'r-')
            ax.axhline(y=0.0,color='k',linestyle='--')
            if xlim:
                ax.set_xlim([xlim[0],xlim[1]])
            if ylim:
                ax.set_ylim([ylim[0],ylim[1]])
            plt.savefig(outfile+'.imaginary_vs_uvrad.pdf')
            plt.close(fig)
        else:
            print('WARNING, plot already exists and you do not want to'
            'overwrite it')
