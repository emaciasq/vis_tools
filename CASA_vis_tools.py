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
    def __init__(self,mydat,freqs,name='',spwids=[]):
        '''
        INPUTS:
        - mydat: list of dictionaries returned by ms.getdata()
        - freqs: list of frequencies for which there is an element in mydat
        OPTIONAL INPUTS:
        - name: name of the measurement set from where these visibilities were
        taken from.
        - spwids: spectral windows ids for which these visibilities have been
        computed.
        (in GHz).
        '''
        if type(mydat) is not list:
            mydat = [mydat]
        if (type(freqs) is not list) and (type(freqs) is not np.ndarray):
            freqs = [freqs]
        self.freqs = np.array(freqs) * 1.0e9 # frequencies in Hz
        self.wl = light_speed / (np.array(freqs) * 1.0e9) # wavelengths in meters
        rr = []
        ii = []
        uu = []
        vv = []
        wt = []
        for i,dat in enumerate(mydat):
            # We first build the u, v, and wt arrays with the same shape
            # as the visibilities
            u_temp = np.zeros_like(dat['real'])
            v_temp = np.zeros_like(dat['real'])
            wt_temp = np.zeros_like(dat['real'])
            for j in range(dat['real'].shape[0]): # For all polarizations
                for k in range(dat['real'].shape[1]): # For every channel
                    u_temp[j,k,:] = dat['u']
                    v_temp[j,k,:] = dat['v']
                    wt_temp[j,k,:] = dat['weight'][j,:]
            # We select points that are not flagged
            # The indexing will flatten the array into a 1D array
            uu.append(u_temp[dat['flag'] == False] / self.wl[i])
            vv.append(v_temp[dat['flag'] == False] / self.wl[i])
            wt.append(wt_temp[dat['flag'] == False])
            rr.append(dat['real'][dat['flag'] == False])
            ii.append(dat['imaginary'][dat['flag'] == False])
        # We concatenate all spws together
        u = np.concatenate(uu,axis=0)
        v = np.concatenate(vv,axis=0)
        r = np.concatenate(rr,axis=0)
        i = np.concatenate(ii,axis=0)
        wt = np.concatenate(wt,axis=0)

        self.spwids = spwids

        super(CASA_vis_obj,self).__init__(u=u,v=v,r=r,i=i,wt=wt,name=name)

def get_sim_model(calms,model_images,freqs,fwidths,pa=0.0,indirection='',spw_del=0,
    residuals = True):
    '''
    Function that simulates a radiointerferometric observation out of a
    model fits image. It will produce the visibilities of the model using
    the same uv coverage as the provided calibrated visibilities. It needs
    to simulate a model image for each spw of the observations. The visibilities
    will be averaged in time (limited by scans) and frequency (limited by spws).
    The calibrated visibilities should already be splitted, with no other sources,
    and with the calibrated data in the 'data' datacolumn.

    INPUT parameters:
    - calms: calibrated (observed) visibilities
    - model_images: list of model images at different frequencies, in fits format.
    These frequencies should be the central frequencies of the spws in the
    observations. They need to be in the same order, so first check the listobs
    of the observations. It has to be a list, even if you are simulating just
    one frequency.
    - freqs: list of central frequencies of the spws and model_images. It has to
    be a list, even if you are simulating just one frequency. In GHz.
    - fwidths: width of the spws of the observations. It can be a list with an
    element for each spw (freqs), or it can be just one value, and it will be assumed
    that all spws have the same width. In MHz.
    Optional parameters:
    - pa: position angle of the disk (from north to east). Provide it just if the disk
    needs to be rotated (DIAD images need to be rotated).
    - indirection: coordinates of the center of the model image. If not provided, it
    will look for this information in the header of the fits files.
    - spw_del: "delay" in the spw id of the calibrated ms. In order to know if you need
    to set this value, check the output of listobs (for the calibrated visibilities), and
    then check the spw id of the first spectral window. If it is 0, you don't need to use
    this parameter. If it is not 0, use that number.
    - residuals: Calculate residual visibilities (observation - model)?

    OUTPUT:
    A vis_obj object with the visiblities in it.
    It will also create a simulated measurement set.
    '''
    if len(model_images) != len(freqs):
        raise IOError('GET_SIM_MODEL: Number of frequencies should be the same as the number of input model images.')
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
        RefFreqs.append(round(axis_info[key]['RefFreq']/1e9,6)) # in GHz
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
            raise ValueError('GET_SIM_MODEL: Frequency '+freq+' is not one of the reference frequencies of calms.')

        if pa != 0.0:
            # Rotating image
            imObj = pyfits.open(fitsimage)
            Header = imObj[0].header  # we keep the header
            mod_image = imObj[0].data[:,:] # image in matrix
            imObj.close()

            #rotangle = -(pa - 90.0)
            rotangle = -(pa)
            rotdisk = scipy.ndimage.interpolation.rotate(mod_image,rotangle,reshape=False)
            fitsimage = fitsimage[:-4]+'rot.fits' # Name of rotated image
            rotImObj = pyfits.writeto(fitsimage,rotdisk,Header,clobber=True)

        # We get the inbright and pixel size
        stats = imstat(fitsimage)
        #head = imhead(fitsimage)
        #delt = head['incr'][0]
        #inbright = str(stats['max'][0]*(delt**2.)*1.0e23)+'Jy/pixel'
        #delta = delt*180./np.pi*3600.
        delt = Header['cdelt1'] * np.pi / 180. # in radians
        inbright = str(stats['max'][0]*(delt**2.)*1.0e23)+'Jy/pixel'
        delta = delt*180./np.pi*3600. # in arcsec


        # We import the image into CASA format
        imname0 = fitsimage[:-4]+'image'
        importfits(fitsimage=fitsimage,imagename=imname0,overwrite=True,defaultaxes=False)
        os.system('rm '+fitsimage)

        # We modify the image to include the stokes and frequency axis.
        util = simutil()
        imname = fitsimage[:-4]+'fixed.image'
        util.modifymodel(inimage=imname0,outimage=imname,inbright=inbright,
                        indirection=indirection,incell=str(delta)+'arcsec',
                        incenter=freq,inwidth=widthf,innchan=1)
        os.system('rm -r '+imname0)

        # We split the calibrated visibilities in spw
        # and average in channels and time
        #obsms = calms+'.spw'+str(spwid)+'.avg.ms'
        modelms = fitsimage[:-4]+'model_vis.avg.spw'+str(spwid+spw_del)+'freq'+freq+'.ms'
        if os.path.isdir(modelms) == False:
            split(vis=calms,outputvis=modelms,spw=str(spwid+spw_del),keepflags=False,width=20000,timebin='1e8s',datacolumn='data')
            # We remove the pointing table
            tb.open(modelms+'/POINTING',nomodify=False)
            tb.removerows(range(tb.nrows()))
            tb.done()
        if residuals:
            residualms = fitsimage[:-4]+'model_vis.avg.spw'+str(spwid+spw_del)+'freq'+freq+'.residuals_ms'
#            os.system('cp -r ' + modelms + ' ' + residualms)
            if os.path.isdir(residualms) == False:
                split(vis=calms,outputvis=residualms,spw=str(spwid+spw_del),keepflags=False,width=20000,timebin='1e8s',datacolumn='data')
                # We remove the pointing table
                tb.open(residualms+'/POINTING',nomodify=False)
                tb.removerows(range(tb.nrows()))
                tb.done()

        # We simulate the observation
        sm.openfromms(modelms)
        sm.setvp()
        #sm.summary()
        sm.predict(imagename=imname)
        sm.done()
        os.system('rm -r '+imname)

        # Extract visibilities of the model
        ms.open(modelms)
        ms.selectinit(reset=True)
#        ms.selectinit(datadescid=spwid+spw_del)
        modeldata = ms.getdata(['real','imaginary','u','v','weight','flag','data'])
        mydat.append(modeldata) # this returns a dictionary of arrays
        ms.close()

        # Residuals
        if residuals:
            obsms = calms+'.avg.ms'
            if os.path.isdir(obsms) == False:
                split(vis=calms,outputvis=obsms,keepflags=False,width=20000,timebin='1e8s',datacolumn='data')
            # Extract visibilities of observations
            ms.open(obsms)
            ms.selectinit(reset=True)
            ms.selectinit(datadescid=spwid+spw_del)
            resdata = ms.getdata(['real','imaginary','u','v','weight','flag','data'])
            ms.close()
            # Subtract model from observations
            resdata['real'] = resdata['real'] - modeldata['real']
            resdata['imaginary'] = resdata['imaginary'] - modeldata['imaginary']
            resdata['data'] = resdata['data'] - modeldata['data']
            resdat.append(resdata)
            # Save residuals to ms
            ms.open(residualms,nomodify=False)
            ms.selectinit(reset=True)
#            ms.selectinit(datadescid=spwid+spw_del)
            ms.putdata({'data':resdata['data']})
            ms.close()

    model_vis = CASA_vis_obj(mydat,freqs, name = 'model', spwids = spwids)

    if residuals:
        res_vis = CASA_vis_obj(resdat, freqs, name = 'residuals', spwids = spwids)
        return model_vis, res_vis
    else:
        return model_vis

def get_vis_obs(calms,spwids=None):
    '''
    Function that retrieves the visibilities of a calibrated measurement set.

    INPUT:
    - calms: calibrated measurement set. It should be splitted, with no other sources,
    and with the calibrated data in the 'data' datacolumn.
    - spwids: list of spwids for which you want to get the visibilities.
    OUTPUT:
    A vis_obj object with the visiblities in it.
    '''
    # We average the calibrated visibilities in channels and time
    obsms = calms+'.avg.ms'
    if os.path.isdir(obsms) == False:
        split(vis=calms,outputvis=obsms,keepflags=False,width=20000,timebin='1e8s',datacolumn='data')

    # Extract information of the channels
    ms.open(obsms)
    ms.selectinit(reset=True)
    ms.selectinit()
    axis_info = ms.getspectralwindowinfo()

    if spwids == None:
        spwids = axis_info.keys()
    mydat = []
    freqs = []
    for spwid in spwids:
        freqs.append(axis_info[spwid]['RefFreq'] / 1.0e9)
        # Extract visibilities of observation
        ms.selectinit(reset=True)
        ms.selectinit(datadescid=int(spwid))
        mydat.append(ms.getdata(['real','imaginary','u','v','weight','flag']))
    ms.close()

    obsdat = CASA_vis_obj(mydat,freqs, name = obsms, spwids = spwids)

    return obsdat
