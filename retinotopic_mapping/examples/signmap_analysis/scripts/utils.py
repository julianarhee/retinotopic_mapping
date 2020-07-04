# -*- coding: utf-8 -*-

"""
Created on Thu Jun 25 18:00:19 2020

@author: julianarhee
"""
import os
import glob
import cv2
import re
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy import misc, interpolate, stats, signal


# Label final patches
def assign_patch_name(trial, plot=True):
    if plot:
        trial.plotFinalPatchBorders(borderWidth=2, fontSize=20)
        plt.show()
         
    selected_patches = raw_input('List patches (comma-sep, no spaces): ')
    patch_ids = ['patch%02d' % int(s) for s in selected_patches.split(',')]
    names = []
    for patch in patch_ids:
        patch_name = raw_input('%s, name = ' % patch)
        names.append([patch, patch_name])
    print("Labeled patches:\n  %s" % str(names))

    finalPatchesMarked = dict(trial.finalPatches)

    for i, namePair in enumerate(names):
        currPatch = finalPatchesMarked.pop(namePair[0])
        newPatchDict = {namePair[1]:currPatch}
        finalPatchesMarked.update(newPatchDict)
        
    trial.finalPatchesMarked = finalPatchesMarked

    return trial



# Data saving
def save_params(params, data_id='DATAID', dst_dir='/tmp'):
    params_fpath = os.path.join(dst_dir, '%s_params.json' % data_id)
    with open(params_fpath, 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    print("saved params:\n  %s" % params_fpath)


def create_params(verbose=False, **kwargs):

    params = {'phaseMapFilterSigma': 1.,
            'signMapFilterSigma': 9.,
            'signMapThr': 0.3,
            'eccMapFilterSigma': 15.0,
            'splitLocalMinCutStep': 10.,
            'closeIter': 3,
            'openIter': 3,
            'dilationIter': 15,
            'borderWidth': 1,
            'smallPatchThr': 400, #100,
            'visualSpacePixelSize': 0.5,
            'visualSpaceCloseIter': 15,
            'splitOverlapThr': 1.1,
            'mergeOverlapThr': 0.1,
            
            # Screen info
            'lmin_alt': -58.5, #lmin_azi, #alt, #-33.66
            'lmax_alt': 58.5, #lmax_azi, #alt, # 33.66
            'lmin_azi': -58.5, #lmin_azi,
            'lmax_azi': 58.5, #lmax_azi,

            # Preprocessing
            'mag_thr': 0.02, # mag_thr,
            'smooth_fwhm': 7.0 #smooth_fwhm
            }

    print("updating params")
    for k, v in kwargs.items():
        if verbose:
            print(k,v)
        params.update({k: v})
    
    return params


# Screen remapping
def get_widefield_dims():
    return (117.0, 67.0)

def get_screen_limits():
    (screen_w, screen_h) = get_widefield_dims()
    screen_w_deg = screen_w/2.
    screen_h_deg = screen_h/2.

    return screen_w_deg, screen_h_deg

def map_phase_to_screen(azi_map, alt_map, vmin=0, vmax=2*np.pi):
    screen_w_deg, screen_h_deg = get_screen_limits()

    lmin_azi, lmax_azi = (-screen_w_deg, screen_w_deg)
    lmin_alt, lmax_alt = (-screen_w_deg, screen_w_deg) #(-33.66, 33.66)
    screen_h_min, screen_h_max = (-screen_h_deg, screen_h_deg)

    print("Old min/max: %.2f, %.2f" % (vmin, vmax))
    print("[AZI] New min/max: %.2f, %.2f" % (lmin_azi, lmax_azi))
    print("[ALT] New min/max: %.2f, %.2f" % (lmin_alt, lmax_alt))

    azi_lin = convert_range(azi_map, newmin=lmin_azi, newmax=lmax_azi, oldmin=vmin, oldmax=vmax)
    alt_lin = convert_range(alt_map, newmin=lmin_alt, newmax=lmax_alt, oldmin=vmin, oldmax=vmax)

    new_limits = (lmin_azi, lmax_azi)

    alt_screen_lim = (lmax_alt-screen_h_max)/(lmax_alt*2.)

    return azi_lin, alt_lin, new_limits, alt_screen_lim

# Data loading
retino_analysis_base = 'analyzed_data/Retinotopy/phase_encoding/Images_Cartesian_Constant'  
results_sub_dir = 'Analyses/timecourse/not_motion_corrected/excludeEdges_averageFrames_11_minusRollingMean/phase_decoding/Files'

def get_map_datapaths(animalid, session, verbose=False,
                base_dir='/n/coxfs01/widefield-data/analyzed_data/Retinotopy/phase_encoding/Images_Cartesian_Constant',
                results_sub_dir = 'Analyses/timecourse/not_motion_corrected/excludeEdges_averageFrames_11_minusRollingMean/phase_decoding/Files'):
    
    mappaths = glob.glob(os.path.join(base_dir, animalid, '%s*' % session, results_sub_dir, '*.npz'))
    if verbose:
        for i, p in enumerate(mappaths):
            print(i, p)

    condition_nums = {1: 'left',
                      2: 'right',
                      3: 'top',
                      4: 'bottom'}

    mappath_d = {}
    for mpath in mappaths:

        cond_name = re.findall(r'cond\d{1}_maps', mpath)[0]
        cond_num = int(re.findall(r'\d{1}', cond_name)[0])
        curr_cond = condition_nums[cond_num]
        
        mappath_d.update({curr_cond: mpath})

    return mappath_d


def get_map_data(animalid, session, dims=None, power_metric='mag', 
                    smooth_fwhm=7.0, smooth_first=True, verbose=False):

    mappath_d = get_map_datapaths(animalid, session, verbose=verbose)
    condition_keys = {'azimuth': ['left', 'right'],
                      'altitude': ['top', 'bottom']}

    mapdata = {}
    all_mapdata = {}
    for cond, bar_pos in condition_keys.items():
        cond_data = dict((bpos, np.load(mappath_d[bpos])) for bpos in bar_pos if bpos in mappath_d.keys())
        phasemap_, magmap_ = convert_to_absolute(cond_data, 
                                                smooth_fwhm=smooth_fwhm, smooth=smooth_first)

        # Rescale if needed
        if dims is not None:
            widefield_d1, widefield_d2 = dims #vasculature_map.shape
            print("Vasculature img sz (%i, %i)" % (widefield_d1, widefield_d2))

        map_d1, map_d2 = phasemap_.shape
        resize_maps_d1 = widefield_d1/float(map_d1)
        resize_maps_d2 = widefield_d2/float(map_d2)

        if map_d1 != widefield_d1 or map_d2 != widefield_d2:
            phasemap_ = cv2.resize(phasemap_, (widefield_d2, widefield_d1), fx=resize_maps_d1, fy=resize_maps_d2)
            magmap_ = cv2.resize(magmap_, (widefield_d2, widefield_d1), fx=resize_maps_d1, fy=resize_maps_d2)

        mapdata[cond] = {'phase': phasemap_, 
                        power_metric: magmap_}    
        all_mapdata.update(cond_data)

    return mapdata

def get_widefield_session(animalid, 
                          base_dir='/n/coxfs01/widefield-data/analyzed_data/Retinotopy/phase_encoding/Images_Cartesian_Constant'):
    all_sessions = glob.glob(os.path.join(base_dir, animalid, '2019*'))
    assert len(all_sessions)>0, 'No sesssions found for %s.\n(src: %s)' % (animalid, base_dir)
    found_sessions = np.unique([os.path.split(session)[-1].split('_')[0] \
                                for session in all_sessions])
    print('[%s]: Found %i sessions' % (animalid, len(found_sessions)))
    for si, session in enumerate(found_sessions):
        print(si, session)
    ix = -1
    if len(found_sessions) > 1:
        ix = input("Select IDX of session to use: ")
    session = found_sessions[ix]
    return session

def get_surface_image(animalid, session, rootdir='/n/coxfs01/2p-data'):
    surface_path = glob.glob(os.path.join(rootdir, animalid, 'macro_maps', '20*', '*urf*.png'))[0]
    surface_img = cv2.imread(surface_path, -1)
    return surface_img


def convert_range(oldval, newmin=None, newmax=None, oldmax=None, oldmin=None):
    oldrange = (oldmax - oldmin)
    newrange = (newmax - newmin)
    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
    return newval


def convert_to_absolute(cond_data, smooth_fwhm=7, smooth=True, power_metric='mag'):
    '''combine absolute, or shift single-cond map so that
    
    if AZI, 0=left, 2*np.pi=right 
    if ALT, 0=bottom 2*np.pi= top
    
    Use this to convert to linear coords, centered around 0.
    
    power_metric: can be 'mag' or 'magRatio' (for Map type saved in analyzed maps).
    
    '''
    vmin = 0
    vmax = 2*np.pi

    if len(cond_data.keys()) > 1:
        c1 = 'left' if 'left' in cond_data.keys() else 'top'
        c2 = 'right' if c1=='left' else 'bottom'
    
        # Phase maps
        if smooth:
            m1 = shift_map(smooth_array(cond_data[c1]['phaseMap'], smooth_fwhm, phaseArray=True))
            m2 = shift_map(smooth_array(cond_data[c2]['phaseMap'], smooth_fwhm, phaseArray=True))
        else:
            m1 = shift_map(cond_data[c1]['phaseMap'])
            m2 = shift_map(cond_data[c2]['phaseMap'])
            
        combined_phase_map = stats.circmean(np.dstack([m1, m2]), axis=-1, low=vmin, high=vmax) 
        
        # Mag maps
        p1 = cond_data[c1]['%sMap' % power_metric]
        p2 = cond_data[c2]['%sMap' % power_metric]
        combined_mag_map = np.mean(np.dstack([p1, p2]), axis=-1)

    else:
        if 'right' in cond_data.keys() and 'top' not in cond_data.keys():
            m1 = cond_data['right']['phaseMap'].copy()
            m2 = cond_data['right']['phaseMap'].copy()*-1
            p1 = cond_data['right']['%sMap' % power_metric].copy()
            
        elif 'top' in cond_data.keys() and 'right' not in cond_data.keys():
            m1 = cond_data['top']['phaseMap'].copy()
            m2 = cond_data['top']['phaseMap'].copy()*-1
            p1 = cond_data['top']['%sMap' % power_metric].copy()
        
        # Phase maps
        combined_phase_map = (m2-m1)/2.
        
        # Mag maps
        combined_mag_map = p1
        
        if smooth:
            combined_phase_map = smooth_array(combined_phase_map, smooth_fwhm, phaseArray=True)
            combined_mag_map = smooth_array(combined_mag_map, smooth_fwhm, phaseArray=False)
        
        # Shift maps
        combined_phase_map = shift_map(combined_phase_map) # values should go from 0 to 2*pi        
    
    return combined_phase_map, combined_mag_map #_shift


def smooth_array(inputArray, fwhm, phaseArray=False):
    '''copied from 2p-pipeline
    '''
    szList=np.array([None,None,None,11,None,21,None,27,None,31,None,37,None,43,None,49,None,53,None,59,None,55,None,69,None,79,None,89,None,99])
    sigmaList=np.array([None,None,None,.9,None,1.7,None,2.6,None,3.4,None,4.3,None,5.1,None,6.4,None,6.8,None,7.6,None,8.5,None,9.4,None,10.3,None,11.2,None,12])
    sigma=sigmaList[fwhm]
    sz=szList[fwhm]
    #print(sigma, sz)
    if phaseArray:
        outputArray = smooth_phase_array(inputArray,sigma,sz)
    else:
        outputArray=cv2.GaussianBlur(inputArray, (sz,sz), sigma, sigma)
        
    return outputArray
        
def smooth_phase_array(theta,sigma,sz):
    #build 2D Gaussian Kernel
    kernelX = cv2.getGaussianKernel(sz, sigma); 
    kernelY = cv2.getGaussianKernel(sz, sigma); 
    kernelXY = kernelX * kernelY.transpose(); 
    kernelXY_norm=np.true_divide(kernelXY,np.max(kernelXY.flatten()))
    
    #get x and y components of unit-length vector
    componentX=np.cos(theta)
    componentY=np.sin(theta)
    
    #convolce
    componentX_smooth=signal.convolve2d(componentX,kernelXY_norm,mode='same',boundary='symm')
    componentY_smooth=signal.convolve2d(componentY,kernelXY_norm,mode='same',boundary='symm')

    theta_smooth=np.arctan2(componentY_smooth,componentX_smooth)
    return theta_smooth


def shift_map(phase_az):
    phaseC_az=np.copy(phase_az)
    minv = phase_az[~np.isnan(phase_az)].min()
    maxv = phase_az[~np.isnan(phase_az)].max()
    #print(minv, maxv)
    if (minv < 0): # and maxv > 0):
        phaseC_az[phase_az<0]=-phase_az[phase_az<0]
        #print("flipped neg", phaseC_az.min(), phaseC_az.max())
        phaseC_az[phase_az>0]=(2*np.pi)-phase_az[phase_az>0]
        #print("flipped pos", phaseC_az.min(), phaseC_az.max())

    else:
        print("Already non-negative (min/max: %.2f, %.2f)" % (phaseC_az.min(), phaseC_az.max()))
    return phaseC_az


# Plotting
def label_figure(fig, data_id):
    fig.text(0, .97, data_id)


def plot_mapdata(mapdata, power_metric='mag', cmap='nipy_spectral', vmin=None, vmax=None):
    fig = plt.figure()

    azi_map = mapdata['azimuth']['phase'].copy() #mapdata['azimuth'].copy()
    alt_map = mapdata['altitude']['phase'].copy()
    
    azi_pwr = mapdata['azimuth']['%s' % power_metric]/mapdata['azimuth']['%s' % power_metric].max()
    alt_pwr = mapdata['altitude']['%s' % power_metric]/mapdata['altitude']['%s' % power_metric].max()

    if vmin is None:
        vmin = alt_map.min()
    if vmax is None:
        vmax = alt_map.max()

    fig.add_subplot(221)
    plt.imshow(azi_map, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Azimuth')

    fig.add_subplot(222)
    plt.imshow(alt_map, cmap=cmap, vmin=vmin, vmax=vmax) #,  vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Altitude')


    fig.add_subplot(223)
    plt.imshow(azi_pwr, cmap='hot', vmin=0, vmax=1)
    plt.colorbar()

    fig.add_subplot(224)
    plt.imshow(alt_pwr, cmap='hot',  vmin=0, vmax=1)
    plt.colorbar()

    return fig

def plot_retinotopy(azi_lin, alt_lin, vmin=-58.50, vmax=58.50, 
                    cmap='nipy_spectral', screen_lim_pos=None):
    fig = plt.figure()

    fig.add_subplot(121)
    plt.title('Azimuth')
    plt.imshow(azi_lin, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(orientation='horizontal', shrink=0.8)


    fig.add_subplot(122)
    plt.title('Altitude')
    plt.imshow(alt_lin, cmap=cmap, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(orientation='horizontal', shrink=0.8)

    if screen_lim_pos is not None:
        cb.ax.plot(screen_lim_pos, 0.5, 'w.') # my data is between 0 and 1
        cb.ax.plot(1-screen_lim_pos, 0.5, 'w.') # my data is between 0 and 1

    return fig

def plot_input_maps(azimuth_phase_map, altitude_phase_map, 
                    azimuth_power_map, altitude_power_map, 
                    cmap='nipy_spectral', vmin_az=-58.50, vmax_az=58.50,
                    vmin_el=None, vmax_el=None):
    if vmin_el is None:
        vmin_el = vmin_az
    if vmax_el is None:
        vmax_el = vmax_az
        
    f = plt.figure(figsize=(8, 6))
    ax1 = f.add_subplot(221)

    fig1 = ax1.imshow(altitude_phase_map, vmin=vmin_el, vmax=vmax_el, cmap=cmap, interpolation='nearest')
    ax1.set_axis_off()
    ax1.set_title('altitude map')
    _ = f.colorbar(fig1, orientation='horizontal', shrink=0.5)

    ax2 = f.add_subplot(222)
    fig2 = ax2.imshow(azimuth_phase_map, vmin=vmin_az, vmax=vmax_az, cmap=cmap, interpolation='nearest')
    ax2.set_axis_off()
    ax2.set_title('azimuth map')
    _ = f.colorbar(fig2, orientation='horizontal', shrink=0.5)

    ax3 = f.add_subplot(223)
    fig3 = ax3.imshow(altitude_power_map, vmin=0, vmax=1, cmap='hot', interpolation='nearest')
    ax3.set_axis_off()
    ax3.set_title('altitude power map')
    _ = f.colorbar(fig3, orientation='horizontal', shrink=0.6)

    ax4 = f.add_subplot(224)
    fig4 = ax4.imshow(azimuth_power_map, vmin=0, vmax=1, cmap='hot', interpolation='nearest')
    ax4.set_axis_off()
    ax4.set_title('azimuth power map')
    _ = f.colorbar(fig4, orientation='horizontal', shrink=0.5)

    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    return f

