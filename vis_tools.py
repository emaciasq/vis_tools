#!/usr/bin/env python
import numpy as np
import os
import copy
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

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
    - r_noshift, i_noshift: backup of real and imaginary parts
    without any shifts (in Jy, array).
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
    def __init__(self, u=None, v=None, r=None, i=None, wt=None, name='',
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
            if (np.all(u) == None) or (np.all(v) == None) or (np.all(r) == None)\
            or (np.all(i) == None) or (np.all(wt) == None):
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

        self.sigma = 1.0 / np.sqrt(self.wt)
        self.uvwave = np.sqrt(self.u**2. + self.v**2.)

        self.name = name
        # Initialization of some attributes used later
        self.bin_centers = None
        self.deproj = False # Are the binned visibilities deprojected?
        self.r_noshift = self.r
        self.i_noshift = self.i

    def import_vis(self, input_file):
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
        - NPZ (binary) file:
        Binary file with:
                - u (lambdas)
                - v (lambdas)
                - V (visibilities, as in re+j*im; Jy)
                - weights
        - TXT file:
        ASCII file with a 1 line header and 5 columns:
                - u (lambdas)
                - v (lambdas)
                - Real part of visibilities (Jy)
                - Imaginary part of visibilities (Jy)
                - weight of visibility point.
        '''
        if type(input_file) is not str:
            raise IOError('input_file shoud be a string')
        if input_file[-4:] == '.csv':
            data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
            u = data[:,0]
            v = data[:,1]
            r = data[:,2]
            i = data[:,3]
            wt = data[:,4]
        if input_file[-4:] == '.txt':
            data = np.genfromtxt(input_file, delimiter='\t', skip_header=1)
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
        elif input_file[-4:] == '.npz':
            data = np.load(input_file)
            u = data['u']
            v = data['v']
            vis = data['V']
            r = vis.real
            i = vis.imag
            wt = data['weights']

        self.u = u
        self.v = v
        self.r = r
        self.i = i
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
        - x_shift: Shift in RA (in marcsec).
        - y_shift: Shift in Dec (in marcsec).
        OUTPUTS:
        - Shifted real and imaginary parts.

        NOTE:
        The shift is defined as the offset that one needs to apply to the
        absolute coordinates, i.e., if the phase center is at 100,100
        (arbitrary units to simplify the example), and you want it to be at
        101,99, then the shift would be +1,-1. In the equations below, the
        sign of the offset is changed as it is taken into account as a
        modification to the origin of coordinates. Following the above example,
        if the position 100,100 is the original phase center, it would be the
        origin (0,0), and 101,99 would be the position +1,-1. If we want the
        latter to be the new phase center (i.e., the new 0,0 position), we need
        to apply an offset equal to -1,+1.
        '''
        x_shift *= -np.pi / 1000. / 3600. / 180. # To radians
        y_shift *= -np.pi / 1000. / 3600. / 180.

        # self.r = self.r_noshift * np.cos(-2. * np.pi * (x_shift*self.u
        # + y_shift*self.v))
        # self.i = self.i_noshift * np.sin(-2. * np.pi * (x_shift*self.u
        # + y_shift*self.v))
        shift = np.exp(-2.0 * np.pi * 1.0j *
        (self.u * -x_shift + self.v * -y_shift))
        vis_shifted = (self.r_noshift + self.i_noshift * 1.0j) * shift
        self.r = vis_shifted.real
        self.i = vis_shifted.imag

    def bin_vis(self, nbins=20, lambda_lim = None, lambda_min = None,
    deproj=True, use_wt=True, imag=True):
        '''
        Method to bin the visibilities.
        INPUTS:
        - deproj: If True, bin deprojected visibilities.
        - nbins: number of bins to bin the data. If one wants to use different
        bin sizes at different ranges of uv distance, nbins can be given as a
        list. In that case, lambda_lim needs to be defined to give the borders
        of the regions with different bin sizes.
        - lambda_lim: maximum uv distance (in lambdas) to be used. If not given,
        it uses the maximum uv distance in the visibilities. If nbins is given
        as a list with N elements, lambda_lim needs to have N or N-1 elements.
        If it has N-1, the last lambda_lim is assumed to be the maximum uv
        distance in the visibilities.
        - lambda_min: minimum uv distance (in lambdas) to be used. If not given,
        it uses the minimum uv distance in the visibilities. If nbins is given
        as a list, lambda_min is only used for the first binning part.
        - use_wt: If False, it will not use the weights of each visibility t
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
        of visibilities (in Jy, array). (weights ignored)
        - self.i_sigma: standard deviation of values within bins for imaginary
        part of visibilities (in Jy, array). (weights ignored)
        - self.r_err: error of the mean within bins for real part of
        visibilities (in Jy, array).
        - self.i_err: error of the mean within bins for imaginary part of
        visibilities (in Jy, array).
        '''
        # Checking correct inputs
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

        if type(nbins) is list:
            if type(lambda_lim) is list:
                if len(nbins) > len(lambda_lim)+1:
                    raise IOError('lambda_lim should have the same number '+
                    'of elements as nbins, or the same minus 1.')
                elif len(nbins) == len(lambda_lim)+1:
                    lambda_lim.append(np.max(uvwave))
                elif len(nbins) < len(lambda_lim):
                    raise IOError('lambda_lim should have the same number '+
                    'of elements as nbins, or the same minus 1.')
            elif len(nbins) > 2:
                raise IOError('If nbins has more than two elements, lambda_lim'+
                ' should be a list with the same number of elements as nbins, '+
                'or the same minus 1.')
            elif len(nbins) == 2:
                if lambda_lim == None:
                    raise IOError('If nbins has two elements, lambda_lim needs'+
                    ' at least one value.')
                lambda_lim = [lambda_lim,np.max(uvwave)]
            elif len(nbins) == 1:
                if lambda_lim == None:
                    lambda_lim = [np.max(uvwave)]
                else:
                    lambda_lim = [lambda_lim]
        else:
            if type(lambda_lim) is list:
                raise IOError('If lambda_lim is given as a list, nbins needs '+
                'to be a list as well.')
            elif lambda_lim == None:
                lambda_lim = np.max(uvwave)

        if use_wt:
            wt = self.wt
        else:
            wt = 1.0

        if type(nbins) is list:
            ntot = sum(nbins)
            self.bin_centers = np.ones(shape=(ntot))
            self.r_binned = np.ones(shape=(ntot))
            self.r_err = np.ones(shape=(ntot))
            self.i_binned = np.ones(shape=(ntot))
            self.i_err = np.ones(shape=(ntot))
            i_min = 0
            for i in range(len(nbins)):
                if i == 0:
                    if lambda_min is None:
                        lambda_min = np.min(uvwave)
                else:
                    lambda_min = lambda_lim[i-1]
                range_bins = (lambda_min,lambda_lim[i])

                binning_r, bin_edges, binnum = binned_statistic(uvwave, self.r*wt,
                'sum', nbins[i], range_bins)
                binning_wt = binned_statistic(uvwave, wt,
                'sum', nbins[i], range_bins)[0]
                if imag:
                    binning_i = binned_statistic(uvwave, self.i*wt,
                    'sum', nbins[i], range_bins)[0]
                    binning_i[np.where(binning_wt == 0.)] = np.nan
                binning_r[np.where(binning_wt == 0.)] = np.nan
                binning_wt[np.where(binning_wt == 0.)] = np.nan

                bin_width = (bin_edges[1]-bin_edges[0])
                self.bin_centers[i_min:nbins[i]] = bin_edges[1:] - bin_width/2.0
                self.r_binned[i_min:nbins[i]] = binning_r / binning_wt
                self.r_err[i_min:nbins[i]] = np.sqrt(1.0 / binning_wt)
                if imag:
                    self.i_binned[i_min:nbins[i]] = binning_i / binning_wt
                    self.i_err[i_min:nbins[i]] = np.sqrt(1.0 / binning_wt)
                else:
                    self.i_binned[i_min:nbins[i]] = None
                    self.i_err[i_min:nbins[i]] = None
                i_min += nbins[i]
        else:
            if lambda_min is None:
                lambda_min = np.min(uvwave)
            range_bins = (lambda_min,lambda_lim)
            binning_r, bin_edges, binnum = binned_statistic(uvwave, self.r*wt,
            'sum', nbins, range_bins)
            binning_wt = binned_statistic(uvwave, wt,
            'sum', nbins, range_bins)[0]
            binning_r_std = binned_statistic(uvwave, self.r,
            'std', nbins, range_bins)[0]
            if imag:
                binning_i = binned_statistic(uvwave, self.i*wt,
                'sum', nbins, range_bins)[0]
                binning_i_std = binned_statistic(uvwave, self.i,
                'std', nbins, range_bins)[0]
                binning_i[np.where(binning_wt == 0.)] = np.nan
                binning_i_std[np.where(binning_wt == 0.)] = np.nan
            binning_r[np.where(binning_wt == 0.)] = np.nan
            binning_r_std[np.where(binning_wt == 0.)] = np.nan
            # Not used now, part of the work in progress below
            # binning_N = (np.bincount(binnum)[1:]).astype('float')
            # binning_N[np.where(binning_wt == 0.)] = np.nan
            binning_wt[np.where(binning_wt == 0.)] = np.nan

            bin_width = (bin_edges[1]-bin_edges[0])
            self.bin_centers = bin_edges[1:] - bin_width/2.0
            self.r_binned = binning_r / binning_wt
            self.r_err = np.sqrt(1.0 / binning_wt)
            self.r_sigma = binning_r_std

            # Possible corrections to the error, work in progress
            # chisq = []
            # bootstrap_factor = []
            # for i in range(nbins):
            #     if binning_N[i] > 1:
            #         inbin = np.where(binnum==(i+1))
            #         chisq.append(np.sum((self.r[inbin] - self.r_binned[i])**2.
            #          * wt[inbin] ) / (binning_N[i]-1))
            #         bootstrap_factor.append(np.sum((self.r[inbin] -
            #         self.r_binned[i])**2. * wt[inbin]**2.)/ (binning_N[i]-1))
            #     else:
            #         chisq.append(1.0)
            #         bootstrap_factor.append(1.0)
            # # Correcting for under or over dispersion
            # self.r_err2 = self.r_err * np.sqrt(np.array(chisq))
            # # Bootstrapping solution
            # self.r_err3 = np.sqrt(binning_N*np.array(bootstrap_factor))/binning_wt
            if imag:
                self.i_binned = binning_i / binning_wt
                self.i_err = np.sqrt(1.0 / binning_wt)
                self.i_sigma = binning_i_std
            else:
                self.i_binned = None
                self.i_err = None
                self.i_sigma = None

    def plot_vis(self, real=True, imaginary=False, binned=True, deproj=None,
    nbins=20, errtype='wt', outfile='plot', overwrite=False, xlim=[], ylim=[]):
        '''
        Plots visibilities vs uvdistance (deprojected or not).
        INPUTS:
        - real: plot real part of visibilities? (boolean)
        - imaginary: plot imaginary part of visibilities? (boolean)
        - binned: plot binned visibilities? (boolean)
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
        if binned:
            if np.all(self.bin_centers) == None:
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
            xx = self.bin_centers
        else:
            if deproj:
                try:
                    xx = self.rho
                except:
                    raise IOError('You set deproj=True, but you have not '+
                    'deprojected your visibilities yet.')
            else:
                xx = self.uvwave

        # Plot real part:
        if real:
            if binned:
                yy = self.r_binned
                if errtype == 'wt':
                    err = self.r_err
                else:
                    err = self.r_sigma
            else:
                yy = self.r
                err = self.sigma
            if (os.path.isfile(outfile+'.real_vs_uvrad.pdf') == False) \
            or (overwrite):
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                ax.set_ylabel('Real (Jy)')
                ax.set_xlabel(r'uv distance (k$\lambda$)')
                ax.errorbar(xx/1000., yy, err, fmt='bo', ms=5)
                ax.axhline(y=0.0, color='k', linestyle='--')
                if xlim:
                    ax.set_xlim([xlim[0], xlim[1]])
                if ylim:
                    ax.set_ylim([ylim[0], ylim[1]])
                plt.savefig(outfile+'.real_vs_uvrad.pdf')
                plt.close(fig)
            else:
                print('WARNING, plot already exists and you do not want to'
                'overwrite it')

        # Plot imaginary part:
        if imaginary:
            if binned:
                yy = self.i_binned
                if errtype == 'wt':
                    err = self.i_err
                else:
                    err = self.i_sigma
            else:
                yy = self.i
                err = np.nan
            if (os.path.isfile(outfile+'.imaginary_vs_uvrad.pdf') == False) \
            or (overwrite):
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                ax.set_ylabel('Imaginary (Jy)')
                ax.set_xlabel(r'uv distance (k$\lambda$)')
                ax.errorbar(xx/1000., yy, err, fmt='bo', ms=5)
                ax.axhline(y=0.0, color='k', linestyle='--')
                if xlim:
                    ax.set_xlim([xlim[0], xlim[1]])
                if ylim:
                    ax.set_ylim([ylim[0], ylim[1]])
                plt.savefig(outfile+'.imaginary_vs_uvrad.pdf')
                plt.close(fig)
            else:
                print('WARNING, plot already exists and you do not want to'
                'overwrite it')

    def append(self, vis2):
        '''
        Method to append another vis_obj to the existing one.
        INPUTS:
        - vis2: vis_obj that wants to be appended.
        NOTE:
        Keep in mind that you will need to deproject and/or bin the visibilities
        again if you want to use those.
        '''
        self.u = np.concatenate([self.u, vis2.u])
        self.v = np.concatenate([self.v, vis2.v])
        self.r = np.concatenate([self.r, vis2.r])
        self.i = np.concatenate([self.i, vis2.i])
        self.wt = np.concatenate([self.wt, vis2.wt])
        self.sigma = 1.0 / np.sqrt(self.wt)
        self.uvwave = np.sqrt(self.u**2. + self.v**2.)
        self.r_noshift = self.r
        self.i_noshift = self.i

    def export_csv(self, binned=True, errtype='wt',
    outfile='visibilities_deproj', overwrite=False):
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

    def export_txt(self, outfile='visibilities', overwrite=False):
        '''
        Method to export the visibilities to a txt file, with the format used by
        the code Frankestein.
        INPUTS:
        - outfile: name (without ".txt" extension) of the output file (string).
        - overwrite: overwrite the existing file, if found? (boolean)
        '''
        if os.path.isfile(outfile+'.txt') and (overwrite == False):
            raise IOError('EXPORT: TXT file exists and you do not want to'
            'overwrite it.')
        data = np.column_stack((self.u,self.v, self.r, self.i, self.wt))
        header = 'u(lambdas)\tv(lambdas)\tRe(Jy)\tIm(Jy)\tWeight'
        np.savetxt(outfile+'.txt', data, delimiter='\t',header=header)

    def export_npz(self, outfile='visibilities', overwrite=False):
        '''
        Method to export the visibilities to an npz file, with the format used by
        the code Frankestein.
        INPUTS:
        - outfile: name (without ".npz" extension) of the output file (string).
        - overwrite: overwrite the existing file, if found? (boolean)
        '''
        if os.path.isfile(outfile+'.npz') and (overwrite == False):
            raise IOError('EXPORT: NPZ file exists and you do not want to'
            'overwrite it.')
        data_vis = self.r + 1j*self.i
        np.savez(outfile+'.npz', u=self.u, v=self.v, V=data_vis,
        weights=self.wt)

    def export_fits(self, outfile='visibilities', overwrite=False):
        '''
        Method to export the full set of visibilities (non-deprojected u,v
        coordinates) to a fits file.
        INPUTS:
        - outfile: name (without ".fits" extension) of the output file (string).
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
    the uv coverage of the observed visibilities, so the u,v coordinates
    of each point are the same. This function does not do any interpolation that
    would be needed to calculate the residuals of a more general model.
    '''
    if len(model_vis.u) != len(obs_vis.u):
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

def plot_mod_vis(model_vis, obsvis, resvis=None, real=True, imaginary=False,
    deproj=True, errtype='wt', outfile='model', overwrite=False,
    normalize=False, xlim=[], ylim=[]):
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
