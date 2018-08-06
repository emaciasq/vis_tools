# This script uses tasks and tools that are built inside CASA,
# so it has be run within it. import will not work for the same reason,
# so it has be run using execfile().
import numpy as np
import os
import vis_tools
import scipy.ndimage
from simutil import simutil

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
    - self.uvwave: uv distance (in lambdas, array).
    Extra attributes:
    - self.wl: array with wavelength of each spw in meters.
    - self.freqs: attay with frequency of each spw in Hz.
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
        self.freqs = freqs # frequencies in Hz
        mydat = np.array(mydat)
        self.wl = light_speed / self.freqs # wavelengths in meters
        rr = []
        ii = []
        uu = []
        vv = []
        wt = []
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
                for k in range(dat['real'].shape[1]): # For every channel
                    u_temp[k,:] = dat['u'] / self.wl[i,k]
                    v_temp[k,:] = dat['v'] / self.wl[i,k]
            else:
                # We build the u, v, and wt arrays with the same shape
                # as the visibilities
                u_temp = np.zeros_like(dat['real'])
                v_temp = np.zeros_like(dat['real'])
                wt_temp = np.zeros_like(dat['real'])
                real_temp = dat['real']
                imag_temp = dat['imaginary']
                for j in range(dat['real'].shape[0]): # For all polarizations
                    for k in range(dat['real'].shape[1]): # For every channel
                        u_temp[j,k,:] = dat['u'] / self.wl[i,k]
                        v_temp[j,k,:] = dat['v'] / self.wl[i,k]
                        wt_temp[j,k,:] = dat['weight'][j,:]
                wt_temp[dat['flag'] == True] = 0.0
            # We select points that are not flagged
            # The indexing will flatten the array into a 1D array
            uu.append(u_temp[wt_temp != 0.0])
            vv.append(v_temp[wt_temp != 0.0])
            wt.append(wt_temp[wt_temp != 0.0])
            rr.append(real_temp[wt_temp != 0.0])
            ii.append(imag_temp[wt_temp != 0.0])
        # We concatenate all spws together
        u = np.concatenate(uu,axis=0)
        v = np.concatenate(vv,axis=0)
        r = np.concatenate(rr,axis=0)
        i = np.concatenate(ii,axis=0)
        wt = np.concatenate(wt,axis=0)

        self.spwids = spwids

        super(CASA_vis_obj,self).__init__(u=u,v=v,r=r,i=i,wt=wt,name=name)

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
    be a list, even if you are simulating just one frequency. In GHz, with just
    six significant digits.
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
    As of now, calms needs to have only one channel per spw.
    '''
    if len(model_images) != len(freqs):
        raise IOError('GET_SIM_MODEL: Number of frequencies should be the same'+
        ' as the number of input model images.')
    # We get the spectral windows of calms
    ms.open(calms)
    ms.selectinit(reset=True)
    ms.selectinit()
    axis_info = ms.getspectralwindowinfo()
    ms.close()
    obs_spwids = []
    RefFreqs = []
    for key in axis_info.keys():
        obs_spwids.append(int(key))
        RefFreqs.append(round(axis_info[key]['RefFreq']/1e9,3)) # in GHz
    obs_spwids = np.array(obs_spwids)
    RefFreqs = np.array(RefFreqs)

    mydat = []
    if residuals:
        resdat = []
    spwids = []
    for freqid,freq0 in enumerate(freqs):
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

        if pa != 0.0:
            # Rotating image
            imObj = pyfits.open(fitsimage)
            Header = imObj[0].header  # we keep the header
            mod_image = imObj[0].data[:,:] # image in matrix
            imObj.close()

            rotangle = -(pa)
            rotdisk = scipy.ndimage.interpolation.rotate(mod_image, rotangle,
            reshape=False)
            fitsimage = fitsimage[:-4]+'rot.fits' # Name of rotated image
            rotImObj = pyfits.writeto(fitsimage, rotdisk, Header, clobber=True)

        # We get the inbright and pixel size
        stats = imstat(fitsimage)
        delt = Header['cdelt1'] * np.pi / 180. # in radians
        inbright = str(stats['max'][0]*(delt**2.)*1.0e23)+'Jy/pixel'
        delta = delt*180./np.pi*3600. # in arcsec

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
    '''
    # We average the calibrated visibilities in channels and time
    # obsms = calms
    # Extract information of the channels
    # ms.open(calms)
    # ms.selectinit(reset=True)
    # ms.selectinit()
    # This method of getting the spw information does not seem to work always
    # axis_info = ms.getspectralwindowinfo()

    tb.open(calms)
    # data = tb.getcol("DATA")
    # flag = tb.getcol("FLAG")
    # uvw = tb.getcol("UVW")
    # weight = tb.getcol("WEIGHT")
    if spwids == None:
        spwids = np.unique(tb.getcol("DATA_DESC_ID"))
    tb.close()
    # Get the frequency information
    tb.open(calms+'/SPECTRAL_WINDOW')
    freqstb = tb.getcol("CHAN_FREQ")
    tb.close()

    ms.open(calms)
    ms.selectinit(reset=True)
    mydat = []
    freqs = []
    for spwid in spwids:
        freqs.append(freqstb[:,spwid])
        # Extract visibilities of observation
        ms.selectinit(reset=True)
        ms.selectinit(datadescid=spwid)
        obsdata = ms.getdata(['real','imaginary','u','v','weight','flag'])
        if del_cross:
            obsdata['flag'][1,:,:] = True
            obsdata['flag'][2,:,:] = True
        mydat.append(obsdata)
    ms.close()

    obsdat = CASA_vis_obj(mydat, np.array(freqs), name = calms, spwids = spwids,
    avg_pols=avg_pols)

    return obsdat
