# -*- coding: utf-8 -*-

"""
Created on Thu Jun 25 18:00:19 2020

@author: julianarhee
"""

import os
import glob
import cv2
import re
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
#from scipy import misc, interpolate, stats, signal
#import scipy.stats as spstats  
from matplotlib.colors import LinearSegmentedColormap
import skimage.external.tifffile as tf
      
import retinotopic_mapping
import retinotopic_mapping.RetinotopicMapping as rm
import retinotopic_mapping.tools.PlottingTools as pt
from retinotopic_mapping.tools import FileTools as ft
import utils as util

# Hard code super duper long paths (carryover from WF cleanup branch)
# retino_analysis_base = 'analyzed_data/Retinotopy/phase_encoding/Images_Cartesian_Constant'  
# results_sub_dir = 'Analyses/timecourse/not_motion_corrected/excludeEdges_averageFrames_11_minusRollingMean/phase_decoding/Files'
# retino_base_dir = os.path.join(widefield_root, retino_analysis_base)


# Options


def args_to_dict(astr):
    if astr.startswith('--p_'):
        astr = astr[4:]
    return astr

def extract_options(options):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-D', '--root', dest='rootdir', default='/n/coxfs01/2p-data',
                        help='Data root for 2p [default: /n/coxfs01/2p-data')
    parser.add_argument('-W', '--widefield', dest='widefield_root', default='/n/coxfs01/widefield-data',
                        help='Data root for widefield [default: /n/coxfs01/widefied-data]')
    parser.add_argument('-A', '--aggr', dest='aggregate_dir', default='/n/coxfs01/julianarhee/aggregate-visual-areas',
                        help='aggregate dir / save based dir [default: /n/coxfs01/julianarhee/aggregate-visual-areas')
     
    parser.add_argument('-i', '--animalid', help='Animalid, e.g., JC001')
    parser.add_argument('-s', '--smooth', dest='smooth_fwhm', default=7.0, help='FWHM for smoothing retino maps')    
    parser.add_argument('-m', '--power', dest='power_metric', default='magRatio', help='power metric (mag, magRatio)')
    parser.add_argument('-c', '--cmap', default='nipy_spectral', help='cmap for phase maps [default: nipy_spectral]')
    parser.add_argument('-t', '--mag-thr', dest='mag_thr', default=0.02)

    parser.add_argument('-f', '--fig', dest='figext', default='png', help='figure save type [defaullt: png]')
    parser.add_argument('--new', dest='create_new', action="store_true") 
    
     
    
    args, extras = parser.parse_known_args(options)

    opt_params = {args_to_dict(k): v for k,v in zip(extras[::2], extras[1::2])}
    for k, v in opt_params.items():
        if any(map(str.isdigit, v)):
            opt_params.update({k: float(v)})
       
    return args, opt_params

#argv = '--p_phaseMapFilterSigma 2. --p_signMapFilterSigma 9.0'.split()
#parser = argparse.ArgumentParser()
#args, extras = parser.parse_known_args(argv)


def main(options):
    args, opt_params = extract_options(options)

    rootdir = args.rootdir #'/n/coxfs01/2p-data'
    widefield_root = args.widefield_root #'/n/coxfs01/widefield-data'
    aggregate_dir = args.aggregate_dir #'/n/coxfs01/julianarhee/aggregate-visual-areas'
    base_dir = os.path.join(aggregate_dir, 'widefield-maps', 'signmaps')

    animalid = args.animalid #'JC084'
    smooth_fwhm = int(args.smooth_fwhm) #7
    power_metric = args.power_metric # 'mag'
    cmap_phase = args.cmap #'nipy_spectral'
    figext = args.figext #'png'
    mag_thr = float(args.mag_thr) #0.02
    create_new = args.create_new #True

    shift_maps = True
    smooth_first = True 
    is_save = True


    vmin = 0 if shift_maps else -np.pi
    vmax = 2*np.pi if shift_maps else  np.pi

    # Get widefield FOV
    session = util.get_widefield_session(animalid) #, base_dir=retino_base_dir)
    vasculature_map = util.get_surface_image(animalid, session, rootdir=rootdir)
    widefield_d1, widefield_d2 = vasculature_map.shape

    # Set output dir
    data_id = '%s_%s' % (animalid, session)
    print("Data ID: %s" % data_id)
    dst_dir = os.path.join(base_dir, 'retinotopic-mapper', data_id)
    print("Saving output to:    \n%s" % dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Get analyzed results
    mapdata = util.get_map_data(animalid, session, dims=(widefield_d1, widefield_d2), 
                                power_metric=power_metric, smooth_fwhm=smooth_fwhm, 
                                smooth_first=smooth_first)
    condition_keys = {'azimuth': ['left', 'right'],
                    'altitude': ['top', 'bottom']}

    fig = util.plot_mapdata(mapdata, power_metric=power_metric, cmap=cmap_phase, 
                            vmin=vmin, vmax=vmax)


    # Get linear screen info
    azi_map = mapdata['azimuth']['phase'].copy()
    alt_map = mapdata['altitude']['phase'].copy()
    azi_lin, alt_lin, (lmin_azi, lmax_azi), alt_screen_lim = util.map_phase_to_screen(
                                                                azi_map, alt_map, 
                                                                vmin=vmin, vmax=vmax)

    fig = util.plot_retinotopy(azi_lin, alt_lin, vmin=lmin_azi, vmax=lmax_azi,
                        screen_lim_pos=alt_screen_lim, cmap=cmap_phase)

    # Save input figures
    azi_phase = azi_lin.copy()
    alt_phase = alt_lin.copy()
    azi_power = mapdata['azimuth'][power_metric]/mapdata['azimuth'][power_metric].max()
    alt_power = mapdata['altitude'][power_metric]/mapdata['altitude'][power_metric].max()
    fig = util.plot_input_maps(azi_phase, alt_phase, azi_power, alt_power, 
                        cmap=cmap_phase, vmin_az=lmin_azi, vmax_az=lmax_azi)
    util.label_figure(fig, data_id)
    plt.savefig(os.path.join(dst_dir, '%s_input_maps.png' % data_id))


    # ------------------------------------
    # RETINOTOPIC MAPPING TRIAL OBJECT
    # ------------------------------------a

    # Create params
    params = util.create_params(verbose=True, 
                                mag_thr=mag_thr, smooth_fwhm=smooth_fwhm,
                                lmin_azi=lmin_azi, lmax_azi=lmax_azi, **opt_params)


    #                            phaseMapFilterSigma=2,
    #                            signMapFilterSigma=9.,
    #                            signMapThr=0.35,
    #                            eccMapFilterSigma=10.0,
    #                            splitLocalMinCutStep=5., 
    #                            smallPatchThr=400)

    util.save_params(params, data_id=data_id, dst_dir=dst_dir)

    # Get trial object
    results_dpaths= glob.glob(os.path.join(dst_dir, '%s_M%s*.pkl' % (session, animalid))) #"160211_M214522_Trial1.pkl"
    if len(results_dpaths)==1:
        results_outfile = results_dpaths[0]
    elif len(results_dpaths) > 1:
        for ri, r in enumerate(results_dpaths):
            print(ri, r)
        ix = input("select IDX of trial object to load: ")
        results_outfile = results_dpaths[int(ri)]
    else:
        create_new = True
        print("no file found, creating new")
        results_outfile = os.path.join(dst_dir, '%s_M%s.pkl' % (session, animalid))
        print("saving to:\n %s" % results_outfile)


    if create_new:
        # Create trial object
        trial = rm.RetinotopicMappingTrial(altPosMap=alt_phase, #altitude_phase_map,
                                        aziPosMap=azi_phase, #azimuth_phase_map,
                                        altPowerMap=alt_power, #altitude_power_map,
                                        aziPowerMap=azi_power, #azimuth_power_map,
                                        vasculatureMap=vasculature_map,
                                        mouseID=animalid,
                                        dateRecorded=session,
                                        comments='This is an example.',
                                        params=params)
    else:
        print("loading... \n  %s" % results_outfile)
        trial = rm.loadTrial(results_outfile)


    if create_new:
        trial.params = params
        # _ = trial._getSignMap(isPlot=True, isFixedRange=True, cmap=cmap_phase)
        trial.processTrial(isPlot=False)
        trial.plotTrial(isSave=is_save, saveFolder=dst_dir)

        # Visualize final patches to label
        f = trial.plotFinalPatchBorders(borderWidth=2, fontSize=20)
        figname = '%s_final_patch_borders' % (trial.getName())
        f.savefig(os.path.join(dst_dir, '%s.%s' % (figname, figext)))

        # Assign final patches
        trial = util.assign_patch_name(trial, plot=False)

        # Save results
        trial_dict = trial.generateTrialDict()
        if is_save:
            ft.saveFile(results_outfile,trial_dict)



if __name__ == '__main__':
    main(sys.argv[1:])
    