__author__ = 'junz'

import os
import glob
import cv2
import re

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
rootdir = '/n/coxfs01/2p-data'
widefield_root = '/n/coxfs01/widefield-data'
aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'
base_dir = os.path.join(aggregate_dir, 'widefield-maps', 'signmaps')

animalid = 'JC084'
shift_maps = True
smooth_fwhm = 7
smooth_first = True 
power_metric = 'mag'
cmap_phase = 'nipy_spectral'


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
#reload(util)
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

print(lmin_azi, lmax_azi) #reload(util)


# RETINOTOPIC MAPPING TRIAL OBJECT
mag_thr = 0.03

#params = {'phaseMapFilterSigma': 1.,
#          'signMapFilterSigma': 9.,
#          'signMapThr': 0.3,
#          'eccMapFilterSigma': 15.0,
#          'splitLocalMinCutStep': 10.,
#          'closeIter': 3,
#          'openIter': 3,
#          'dilationIter': 15,
#          'borderWidth': 1,
#          'smallPatchThr': 400, #100,
#          'visualSpacePixelSize': 0.5,
#          'visualSpaceCloseIter': 15,
#          'splitOverlapThr': 1.1,
#          'mergeOverlapThr': 0.1,
#          
#          # Screen info
#          'lmin_alt': lmin_azi, #alt,
#          'lmax_alt': lmax_azi, #alt,
#          'lmin_azi': lmin_azi,
#          'lmaz_azi': lmax_azi,
#
#          # Preprocessing
#          'mag_thr': mag_thr,
#          'smooth_fwhm': smooth_fwhm
#          }
#

reload(util)
mag_thr=0.02
params = util.create_params(verbose=True, mag_thr=mag_thr, smooth_fwhm=smooth_fwhm,
                            lmin_azi=lmin_azi, lmax_azi=lmax_azi)
util.save_params(params, data_id=data_id, dst_dir=dst_dir)

azi_lin.max()
cmap_phase
azi_phase = azi_lin.copy()
alt_phase = alt_lin.copy()
azi_power = mapdata['azimuth'][power_metric]/mapdata['azimuth'][power_metric].max()
alt_power = mapdata['altitude'][power_metric]/mapdata['altitude'][power_metric].max()
print(azi_power.max())
fig = util.plot_input_maps(azi_phase, alt_phase, azi_power, alt_power, 
                    cmap=cmap_phase, vmin_az=lmin_azi, vmax_az=lmax_azi)

util.label_figure(fig, data_id)
plt.savefig(os.path.join(dst_dir, '%s_input_maps.png' % data_id))


# Get trial object
create_new = False
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

is_save = True

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


# _ = trial._getSignMap(isPlot=True, isFixedRange=True, cmap=cmap_phase)

trial.params = params
trial.processTrial(isPlot=True)

trial_dict = trial.generateTrialDict()
trial.plotTrial(isSave=is_save, saveFolder=dst_dir)
#plt.show()

if is_save:
    ft.saveFile(trial.getName()+'.pkl',trial_dict)



