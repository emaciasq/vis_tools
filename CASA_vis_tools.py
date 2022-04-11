# This script uses tasks and tools that are built inside CASA,
# so it has be run within it. import will not work for the same reason,
# so it has be run using execfile().
import numpy as np
import os
import vis_tools
import scipy.ndimage
# from simutil import simutil
# for CASA 6
from casatasks.private.simutil import simutil

#-------------------------------------------------
################### CONSTANTS ####################
#-------------------------------------------------

light_speed = 2.99792458e8 # m/s

#-------------------------------------------------

class CASA_vis_obj(vis_tools.vis_obj):
    '''
    Class for interferometric visibilities objects retrieved
    from measurement sets (MS) using CASA.
    ATTRIBUTES:
    All visobj attributes:
    - self.u: u coordinate of visibilities (in lambdas, array).
    - self.v: v coordinate of visibilities (in lambdas, array).
    - self.r: real part of visibilities (in Jy, array).
    - self.i: imaginary part of visibilitites (in Jy, array).
    - self.wt: weights of visibilities (array).
    - self.wl: wavelength of each datapoint (in m, array).
    - self.uvwave: uv distance (in lambdas, array).
    Extra attributes:
    - self.spwids: spwi ids used.
    '''
    def __init__(self, mydat, freqs, name='', spwids=[], avg_pols=False):
        '''
        INPUTS:
        - mydat: list of dictionaries returned by ms.getdata()
        - freqs: array of frequencies for each channel and spw (in Hz).
        OPTIONAL INPUTS:
        - name: name of the measurement set from where these visibilities were
        taken from.
        - spwids: spectral windows ids for which these visibilities have been
        computed.
        - avg_pols: If True, it will do a weighted average of the polarizations.
        '''
        if type(mydat) is not list:
            mydat = [mydat]
        # if (type(freqs) is not list) and (type(freqs) is not np.ndarray):
        #     freqs = [freqs]
        # self.freqs = freqs # frequencies in Hz
        mydat = np.array(mydat)
        wl_spws = light_speed / freqs # wavelengths in meters
        rr = []
        ii = []
        uu = []
        vv = []
        wt = []
        wl = []
        for i,dat in enumerate(mydat): # For all spws
            if avg_pols: # If we want to average the polarizations
                wt_temp = np.zeros_like(dat['real'])
                for j in range(dat['real'].shape[0]): # For all polarizations
                    for k in range(dat['real'].shape[1]): # For every channel
                        wt_temp[j,k,:] = dat['weight'][j,:]
                wt_temp[dat['flag'] == True] = 0.0
                real_temp = ( np.sum(dat['real'] * wt_temp, axis=0) /
                np.sum(wt_temp, axis=0) )
                imag_temp = ( np.sum(dat['imaginary'] * wt_temp, axis=0) /
                np.sum(wt_temp, axis=0) )
                wt_temp = np.sum(wt_temp, axis=0)
                # We build the u, and v arrays with the same shape
                # as the visibilities
                u_temp = np.zeros_like(real_temp)
                v_temp = np.zeros_like(real_temp)
                wl_temp = np.ones_like(real_temp)
                for k in range(dat['real'].shape[1]): # For every channel
                    u_temp[k,:] = dat['u'] / wl_spws[i][k]
                    v_temp[k,:] = dat['v'] / wl_spws[i][k]
                    wl_temp[k,:] *= wl_spws[i][k]
            else:
                # We build the u, v, and wt arrays with the same shape
                # as the visibilities
                u_temp = np.zeros_like(dat['real'])
                v_temp = np.zeros_like(dat['real'])
                wt_temp = np.zeros_like(dat['real'])
                wl_temp = np.ones_like(dat['real'])
                real_temp = dat['real']
                imag_temp = dat['imaginary']
                for j in range(dat['real'].shape[0]): # For all polarizations
                    for k in range(dat['real'].shape[1]): # For every channel
                        u_temp[j,k,:] = dat['u'] / wl_spws[i][k]
                        v_temp[j,k,:] = dat['v'] / wl_spws[i][k]
                        wt_temp[j,k,:] = dat['weight'][j,:]
                        wl_temp[j,k,:] *= wl_spws[i][k]
                wt_temp[dat['flag'] == True] = 0.0
            # We select points that are not flagged
            # The indexing will flatten the array into a 1D array
            uu.append(u_temp[wt_temp != 0.0])
            vv.append(v_temp[wt_temp != 0.0])
            wt.append(wt_temp[wt_temp != 0.0])
            rr.append(real_temp[wt_temp != 0.0])
            ii.append(imag_temp[wt_temp != 0.0])
            wl.append(wl_temp[wt_temp != 0.0])
        # We concatenate all spws together
        u = np.concatenate(uu,axis=0)
        v = np.concatenate(vv,axis=0)
        r = np.concatenate(rr,axis=0)
        i = np.concatenate(ii,axis=0)
        wt = np.concatenate(wt,axis=0)
        wl = np.concatenate(wl,axis=0)

        self.spwids = spwids

        super(CASA_vis_obj,self).__init__(u=u,v=v,r=r,i=i,wt=wt,wl=wl,name=name)

def get_sim_model(calms, model_images, freqs, fwidths, pa=0.0, indirection='',
    del_cross=False, residuals=True):
    '''
    Function that simulates a radiointerferometric observation out of a
    model fits image. It will produce the visibilities of the model using
    the same uv coverage as the provided calibrated visibilities. It needs
    to simulate a model image for each spw of the observations. The visibilities
    will be averaged in time (limited by scans) and frequency (limited by spws).
    The calibrated visibilities should already be splitted, with no other
    sources, and with the calibrated data in the 'data' datacolumn.

    INPUT parameters:
    - calms: calibrated (observed) visibilities.
    - model_images: list of model images at different frequencies, in fits
    format. These frequencies should be the central frequencies of the spws in
    the observations. They need to be in the same order, so first check the
    listobs of the observations. It has to be a list, even if you are simulating
    just one frequency.
    - freqs: list of central frequencies of the spws and model_images. It has to
    be a list, even if you are simulating just one frequency. In GHz.
    - fwidths: width of the spws of the observations. It can be a list with an
    element for each spw (freqs), or it can be just one value, and it will be
    assumed that all spws have the same width. In MHz.
    OPTIONAL parameters:
    - pa: position angle of the disk (from north to east). Provide it just if
    the disk needs to be rotated (DIAD images need to be rotated).
    - indirection: coordinates of the center of the model image. If not
    provided, it will look for this information in the header of the fits files.
    - residuals: Calculate residual visibilities (observation - model)?
    - del_cross: If True, it will delete cross polarizations. Usually only used
    for VLA observations, not for ALMA.
    OUTPUT:
    A vis_obj object with the visiblities in it.
    It will also create a simulated measurement set.

    NOTE:
    For now, calms needs to have only one channel per spw.
    '''
    if len(model_images) != len(freqs):
        raise IOError('GET_SIM_MODEL: Number of frequencies should be the same'+
        ' as the number of input model images.')
    # We get the spectral windows of calms
    # ms.open(calms)
    # ms.selectinit(reset=True)
    # ms.selectinit()
    # axis_info = ms.getspectralwindowinfo()
    # ms.close()
    tb.open(calms)
    spwids = np.unique(tb.getcol("DATA_DESC_ID"))
    tb.close()
    # Get the frequency information
    tb.open(calms+'/SPECTRAL_WINDOW')
    freqstb = tb.getcol("CHAN_FREQ")
    tb.close()

    obs_spwids = []
    RefFreqs = []
    for key in spwids:
        obs_spwids.append(int(key))
        # RefFreqs.append(round(axis_info[key]['RefFreq']/1e9,3)) # in GHz
        RefFreqs.append(round(freqstb[:,key]/1e9,6)) # in GHz
    obs_spwids = np.array(obs_spwids)
    RefFreqs = np.array(RefFreqs)

    mydat = []
    if residuals:
        resdat = []
    spwids = []
    for freqid,freq0 in enumerate(freqs):
        freq0 = round(freq0,6) # we round the frequency to have 6 decimal digits
        freq = str(freq0) + 'GHz'
        fitsimage = model_images[freqid]
        if type(fwidths) is list:
            widthf = str(fwidths[freqid]) + 'MHz'
        else:
            widthf = str(fwidths) + 'MHz'
        # We find the spwid for this frequency
        try:
            spwid = obs_spwids[RefFreqs == freq0][0]
            spwids.append(spwid)
        except:
            raise ValueError('GET_SIM_MODEL: Frequency '+freq+' is not one of '+
            'the reference frequencies of calms. It could be a rounding issue.')

        # Rotating image
        imObj = pyfits.open(fitsimage)
        Header = imObj[0].header  # we keep the header
        mod_image = imObj[0].data[:,:] # image in matrix
        imObj.close()
        if pa != 0.0:
            rotangle = -(pa)
            rotdisk = scipy.ndimage.interpolation.rotate(mod_image, rotangle,
            reshape=False)
            fitsimage = fitsimage[:-4]+'rot.fits' # Name of rotated image
            rotImObj = pyfits.writeto(fitsimage, rotdisk, Header, clobber=True)

        # We get the inbright and pixel size
        stats = imstat(fitsimage)
        if 'CUNIT1' in Header.keys():
            if Header['CUNIT1'] == 'deg':
                delt = Header['CDELT1'] * np.pi / 180. # to radians
            elif Header['CUNIT1'] == 'rad':
                delt = Header['CDELT1']
            else:
                raise IOError('GET_SIM_MODEL: Potentially weird coordinate '+
                'units. Please use deg or rad.')
        else:
            print('WARNING: Assuming units of model coordinates are deg.')
            delt = Header['CDELT1'] * np.pi / 180. # to radians

        if 'BUNIT' in Header.keys():
            if Header['BUNIT'] == 'Jy/pixel':
                inbright = str(stats['max'][0])+'Jy/pixel'
            elif Header['BUNIT'] == 'W.m-2.pixel-1': # MCFOST format, nu*Fnu
                inbright = stats['max'][0] / (freq0*1e9) / 1e-26 # to Jy
                inbright = str(inbright)+'Jy/pixel'
            elif Header['BUNIT'] == 'erg/s/cm2/Hz':
                inbright = str(stats['max'][0]*(delt**2.)*1.0e23)+'Jy/pixel'
            else:
                raise IOError('GET_SIM_MODEL: Potentially weird intensity '+
                'units. Please use Jy/pixel, W.m-2.pixel-1, or erg/s/cm2/Hz.')
        else:
            print('WARNING: Assuming units of model are erg s-1 cm-2 Hz-1.')
            inbright = str(stats['max'][0]*(delt**2.)*1.0e23)+'Jy/pixel'
        delta = np.abs(delt)*180./np.pi*3600. # to arcsec

        # We import the image into CASA format
        imname0 = fitsimage[:-4]+'image'
        importfits(fitsimage=fitsimage, imagename=imname0, overwrite=True,
        defaultaxes=False)
        # os.system('rm '+fitsimage)

        # We modify the image to include the stokes and frequency axis.
        util = simutil()
        imname = fitsimage[:-4]+'fixed.image'
        util.modifymodel(inimage=imname0,outimage=imname,inbright=inbright,
                        indirection=indirection,incell=str(delta)+'arcsec',
                        incenter=freq,inwidth=widthf,innchan=1)
        os.system('rm -r '+imname0)

        # We split the calibrated visibilities in spw
        modelms = fitsimage[:-4]+'model_vis.spw'+str(spwid)+'freq'+freq+'.ms'
        if os.path.isdir(modelms) == False:
            split(vis=calms, outputvis=modelms, spw=str(spwid), keepflags=False,
            datacolumn='data')
            # We remove the pointing table
            tb.open(modelms+'/POINTING',nomodify=False)
            tb.removerows(range(tb.nrows()))
            tb.done()
        if residuals:
            residualms = (fitsimage[:-4]+'model_vis.spw'+str(spwid)+'freq'+freq+
            '.residuals_ms')
            if os.path.isdir(residualms) == False:
                os.system('cp -r ' + modelms + ' ' + residualms)

        # We simulate the observation
        sm.openfromms(modelms)
        sm.setvp()
        #sm.summary()
        sm.predict(imagename=imname)
        sm.done()
        os.system('rm -r '+imname)

        # Extract visibilities of the model
        ms.open(modelms, nomodify=(del_cross==False))
        ms.selectinit(reset=True)
        modeldata = ms.getdata(['real','imaginary','u','v','weight','flag','data'])
        if del_cross:
            # If True, we flag the cross polarizations
            modeldata['real'][1,:,:] = modeldata['real'][0,:,:]
            modeldata['real'][2,:,:] = modeldata['real'][0,:,:]
            modeldata['imaginary'][1,:,:] = modeldata['imaginary'][0,:,:]
            modeldata['imaginary'][2,:,:] = modeldata['imaginary'][0,:,:]
            modeldata['data'][1,:,:] = modeldata['data'][0,:,:]
            modeldata['data'][2,:,:] = modeldata['data'][0,:,:]
            modeldata['flag'][1,:,:] = True
            modeldata['flag'][2,:,:] = True
            ms.putdata({'data':modeldata['data']})
        mydat.append(modeldata)
        ms.close()

        # Residuals
        if residuals:
            # Extract visibilities of observations
            ms.open(calms)
            ms.selectinit(reset=True)
            ms.selectinit(datadescid=spwid)
            resdata = ms.getdata(['real','imaginary','u','v','weight','flag','data'])
            ms.close()
            # Subtract model from observations
            resdata['real'] = resdata['real'] - modeldata['real']
            resdata['imaginary'] = resdata['imaginary'] - modeldata['imaginary']
            resdata['data'] = resdata['data'] - modeldata['data']
            if del_cross:
                resdata['flag'][1,:,:] = True
                resdata['flag'][2,:,:] = True
            resdat.append(resdata)
            # Save residuals to ms
            ms.open(residualms,nomodify=False)
            ms.selectinit(reset=True)
            ms.putdata({'data':resdata['data']})
            ms.close()

    model_vis = CASA_vis_obj(mydat, np.array([freqs]).T*1e9, name = 'model',
    spwids = spwids)

    if residuals:
        res_vis = CASA_vis_obj(resdat, np.array([freqs]).T*1e9,
        name = 'residuals', spwids = spwids)
        return model_vis, res_vis
    else:
        return model_vis

def get_vis_obs(calms, spwids=None, avg_pols=False, del_cross=False):
    '''
    Function that retrieves the visibilities of a calibrated measurement set.

    INPUT:
    - calms: calibrated measurement set. It should be splitted, with no other
    sources, and with the calibrated data in the 'data' datacolumn.
    - spwids: list of spwids for which you want to get the visibilities.
    - avg_pols: If True, it will do a weighted average of the polarizations.
    - del_cross: If True, it will delete cross polarizations. Usually only used
    for VLA observations, not for ALMA.
    OUTPUT:
    A vis_obj object with the visiblities in it.

    NOTE:
    The table tool is in principle the simplest method to obtain information
    from a measurement set. Note, however, that the function tb.getcol() only
    works if each spw has the same number of channels. tb has the getvarcol()
    function that can be used with table with varying number of rows, but when
    using this to retrieve the visibilities, the results are returned in a
    dictionary that loses all the format of the data, which makes it impossible
    to handle.
    On the other hand, one can use the ms tool. This tool returns dictionaries
    for each spw, conserving the format of the data. However, it cannot retrieve
    the frequency information of each channel, only the representative freq,
    channel width, number of channels, and freq of first channel for each spw.
    Therefore, we use a mix of ms and tb to retrieve the frequency of each
    channel, and the use the ms tool to retrieve the visibilities.
    '''
    # Extract information of the spws
    ms.open(calms)
    ms.selectinit(reset=True)
    axis_info = ms.getspectralwindowinfo()
    if spwids == None:
        spwids = axis_info.keys()

    # Extract information of the channels
    tb.open(calms+'/SPECTRAL_WINDOW')
    freqstb = tb.getvarcol("CHAN_FREQ")
    tb.close()
    # For some reason tb.getvarcol() creates a dict where the keys are different
    # from ms.getspectralwindowinfo(). Instead of being 0,...,N, the keys are
    # r1,...rN+1

    mydat = []
    freqs = []
    for spwid in spwids:
        tbkey = 'r{}'.format(str(int(spwid)+1))
        # We ensure that we are selecting the corresponding spwid
        if freqstb[tbkey][0,0] != axis_info[spwid]['Chan1Freq']:
            raise IOError('The frequencies between ms.getspectralwindowinfo() '+
            ' and tb.getvarcol() do not match. Try splitting the data to make'+
            ' sure that the first spw in your MS has spwid=0.')
        freqs.append(freqstb[tbkey][:,0])
        # Extract visibilities of observation
        ms.selectinit(reset=True)
        ms.selectinit(datadescid=int(spwid))
        obsdata = ms.getdata(['real','imaginary','u','v','weight','flag'])
        if del_cross:
            obsdata['flag'][1,:,:] = True
            obsdata['flag'][2,:,:] = True
        mydat.append(obsdata)
    ms.close()

    obsdat = CASA_vis_obj(mydat, np.array(freqs), name = calms, spwids = spwids,
    avg_pols=avg_pols)

    return obsdat
