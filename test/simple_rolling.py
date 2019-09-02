import numpy as np
import matplotlib.pylab as plt
import healpy as hp
from lsst.sims.featureScheduler.modelObservatory import Model_observatory
from lsst.sims.featureScheduler.schedulers import Core_scheduler
from lsst.sims.featureScheduler.utils import standard_goals, calc_norm_factor, create_season_offset
import lsst.sims.featureScheduler.basis_functions as bf
from lsst.sims.featureScheduler.surveys import (generate_dd_surveys, Greedy_survey,
                                                Blob_survey)
from lsst.sims.featureScheduler import sim_runner
import lsst.sims.featureScheduler.detailers as detailers
import sys
import subprocess
import os
import argparse


def gen_greedy_surveys(nside, nexp=1, target_maps=None, mod_year=None, day_offset=None,
                       norm_factor=None, max_season=10):
    """
    Make a quick set of greedy surveys
    """
    target_map = standard_goals(nside=nside)
    norm_factor = calc_norm_factor(target_map)
    # Let's remove the bluer filters since this should only be near twilight
    filters = ['r', 'i', 'z', 'y']
    surveys = []

    detailer = detailers.Camera_rot_detailer(min_rot=-87., max_rot=87.)

    for filtername in filters:
        bfs = []
        bfs.append(bf.M5_diff_basis_function(filtername=filtername, nside=nside))
        target_list = [tm[filtername] for tm in target_maps]
        bfs.append(bf.Target_map_modulo_basis_function(filtername=filtername,
                                                       target_maps=target_list,
                                                       season_modulo=mod_year, day_offset=day_offset,
                                                       out_of_bounds_val=np.nan, nside=nside,
                                                       norm_factor=norm_factor,
                                                       max_season=max_season))
        bfs.append(bf.Slewtime_basis_function(filtername=filtername, nside=nside))
        bfs.append(bf.Strict_filter_basis_function(filtername=filtername))
        # Masks, give these 0 weight
        bfs.append(bf.Zenith_shadow_mask_basis_function(nside=nside, shadow_minutes=60., max_alt=76.))
        bfs.append(bf.Moon_avoidance_basis_function(nside=nside, moon_distance=40.))

        bfs.append(bf.Filter_loaded_basis_function(filternames=filtername))
        bfs.append(bf.Planet_mask_basis_function(nside=nside))

        weights = np.array([3.0, 0.3, 3., 3., 0., 0., 0., 0.])
        surveys.append(Greedy_survey(bfs, weights, block_size=1, filtername=filtername,
                                     dither=True, nside=nside, ignore_obs='DD', nexp=nexp,
                                     detailers=[detailer]))

    return surveys


def generate_blobs(nside, mixed_pairs=False, nexp=1, no_pairs=False, offset=None, template_weight=6.,
                   target_maps=None, norm_factor=None, mod_year=2, max_season=10, day_offset=None):
    target_map = standard_goals(nside=nside)
    norm_factor = calc_norm_factor(target_map)

    # List to hold all the surveys (for easy plotting later)
    surveys = []

    # Set up observations to be taken in blocks
    filter1s = ['u', 'g', 'r', 'i', 'z', 'y']
    if mixed_pairs:
        filter2s = [None, 'r', 'i', 'z', None, None]
    else:
        filter2s = [None, 'g', 'r', 'i', None, None]

    if no_pairs:
        filter2s = [None, None, None, None, None, None]

    # Ideal time between taking pairs
    pair_time = 22.
    times_needed = [pair_time, pair_time*2]
    detailer = detailers.Camera_rot_detailer(min_rot=-87., max_rot=87.)
    for filtername, filtername2 in zip(filter1s, filter2s):
        bfs = []
        bfs.append(bf.M5_diff_basis_function(filtername=filtername, nside=nside))
        if filtername2 is not None:
            bfs.append(bf.M5_diff_basis_function(filtername=filtername2, nside=nside))
        target_list = [tm[filtername] for tm in target_maps]
        bfs.append(bf.Target_map_modulo_basis_function(filtername=filtername,
                                                       target_maps=target_list,
                                                       season_modulo=mod_year, day_offset=day_offset,
                                                       out_of_bounds_val=np.nan, nside=nside,
                                                       norm_factor=norm_factor,
                                                       max_season=max_season))

        if filtername2 is not None:
            target_list = [tm[filtername2] for tm in target_maps]
            bfs.append(bf.Target_map_modulo_basis_function(filtername=filtername2,
                                                           target_maps=target_list,
                                                           season_modulo=mod_year, day_offset=day_offset,
                                                           out_of_bounds_val=np.nan, nside=nside,
                                                           norm_factor=norm_factor,
                                                           max_season=max_season))

        bfs.append(bf.Slewtime_basis_function(filtername=filtername, nside=nside))
        bfs.append(bf.Strict_filter_basis_function(filtername=filtername))
        bfs.append(bf.N_obs_per_year_basis_function(filtername=filtername, nside=nside,
                                                    footprint=target_map[filtername],
                                                    HA_limit=1., n_obs=3, season=250.))
        if filtername2 is not None:
            bfs.append(bf.N_obs_per_year_basis_function(filtername=filtername2, nside=nside,
                                                        footprint=target_map[filtername2],
                                                        HA_limit=1., n_obs=3, season=250.))
        # Masks, give these 0 weight
        bfs.append(bf.Zenith_shadow_mask_basis_function(nside=nside, shadow_minutes=60., max_alt=76.))
        bfs.append(bf.Moon_avoidance_basis_function(nside=nside, moon_distance=30.))
        filternames = [fn for fn in [filtername, filtername2] if fn is not None]
        bfs.append(bf.Filter_loaded_basis_function(filternames=filternames))
        if filtername2 is None:
            time_needed = times_needed[0]
        else:
            time_needed = times_needed[1]
        bfs.append(bf.Time_to_twilight_basis_function(time_needed=time_needed))
        bfs.append(bf.Not_twilight_basis_function())
        bfs.append(bf.Planet_mask_basis_function(nside=nside))
        weights = np.array([3.0, 3.0, .3, .3, 3., 3., template_weight, template_weight, 0., 0., 0., 0., 0., 0.])
        if filtername2 is None:
            # Need to scale weights up so filter balancing still works properly.
            weights = np.array([6.0, 0.6, 3., 3., template_weight*2, 0., 0., 0., 0., 0., 0.])
        if filtername2 is None:
            survey_name = 'blob, %s' % filtername
        else:
            survey_name = 'blob, %s%s' % (filtername, filtername2)
        surveys.append(Blob_survey(bfs, weights, filtername1=filtername, filtername2=filtername2,
                                   ideal_pair_time=pair_time, nside=nside,
                                   survey_note=survey_name, ignore_obs='DD', dither=True,
                                   nexp=nexp, detailers=[detailer]))

    return surveys


def run_sched(surveys, survey_length=365.25, nside=32, fileroot='baseline_', verbose=False,
              extra_info=None):
    years = np.round(survey_length/365.25)
    scheduler = Core_scheduler(surveys, nside=nside)
    n_visit_limit = None
    observatory = Model_observatory(nside=nside)
    observatory, scheduler, observations = sim_runner(observatory, scheduler,
                                                      survey_length=survey_length,
                                                      filename=fileroot+'%iyrs.db' % years,
                                                      delete_past=True, n_visit_limit=n_visit_limit,
                                                      verbose=verbose, extra_info=extra_info)


def slice_wfd_area(nslice, target_map, scale_down_factor=0.2):
    """
    Slice the WFD area into even dec bands
    """
    # Make it so things still sum to one.
    scale_up_factor = nslice - scale_down_factor*(nslice-1)

    wfd = target_map['r'] * 0
    wfd_indices = np.where(target_map['r'] == 1)[0]
    wfd[wfd_indices] = 1
    wfd_accum = np.cumsum(wfd)
    split_wfd_indices = np.floor(np.max(wfd_accum)/nslice*(np.arange(nslice)+1)).astype(int)
    split_wfd_indices = split_wfd_indices.tolist()
    split_wfd_indices = [0] + split_wfd_indices

    all_scaled_down = {}
    for filtername in target_map:
        all_scaled_down[filtername] = target_map[filtername]+0
        all_scaled_down[filtername][wfd_indices] *= scale_down_factor

    scaled_maps = []
    for i in range(len(split_wfd_indices)-1):
        new_map = {}
        indices = wfd_indices[split_wfd_indices[i]:split_wfd_indices[i+1]]
        for filtername in all_scaled_down:
            new_map[filtername] = all_scaled_down[filtername] + 0
            new_map[filtername][indices] = target_map[filtername][indices]*scale_up_factor
        scaled_maps.append(new_map)

    return scaled_maps


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nexp", type=int, default=1, help="Number of exposures per visit")
    parser.add_argument("--Pairs", dest='pairs', action='store_true')
    parser.add_argument("--noPairs", dest='pairs', action='store_false')
    parser.set_defaults(pairs=True)
    parser.add_argument("--mixedPairs", dest='mixedPairs', action='store_true')
    parser.add_argument("--nomixedPairs", dest='mixedPairs', action='store_false')
    parser.set_defaults(mixedPairs=True)
    parser.add_argument("--verbose", dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument("--survey_length", type=float, default=365.25*10)
    parser.add_argument("--outDir", type=str, default="")
    parser.add_argument("--perNight", dest='perNight', action='store_true')
    parser.add_argument("--maxDither", type=float, default=0.7, help="Dither size for DDFs (deg)")
    parser.add_argument("--splits", type=int, default=2)
    parser.add_argument("--scale_down_factor", type=float, default=0.2)

    args = parser.parse_args()
    nexp = args.nexp
    Pairs = args.pairs
    mixedPairs = args.mixedPairs
    survey_length = args.survey_length  # Days
    outDir = args.outDir
    verbose = args.verbose
    per_night = args.perNight
    max_dither = args.maxDither
    mod_year = args.splits
    scale_down_factor = args.scale_down_factor

    nside = 32

    extra_info = {}
    exec_command = ''
    for arg in sys.argv:
        exec_command += ' ' + arg
    extra_info['exec command'] = exec_command
    extra_info['git hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    extra_info['file executed'] = os.path.realpath(__file__)

    fileroot = 'simplerolling_mod%i_sdf_%.2f_' % (mod_year, scale_down_factor)
    file_end = 'v1.3_'

    observatory = Model_observatory(nside=nside)
    conditions = observatory.return_conditions()

    # Mark position of the sun at the start of the survey. Usefull for rolling cadence.
    sun_ra_0 = conditions.sunRA  # radians
    offset = create_season_offset(nside, sun_ra_0) + 365.25
    # Set up the DDF surveys to dither
    dither_detailer = detailers.Dither_detailer(per_night=per_night, max_dither=max_dither)
    details = [detailers.Camera_rot_detailer(min_rot=-87., max_rot=87.), dither_detailer]
    ddfs = generate_dd_surveys(nside=nside, nexp=nexp, detailers=details)

    sg = standard_goals()
    norm_factor = calc_norm_factor(sg)
    roll_maps = slice_wfd_area(mod_year, sg, scale_down_factor=scale_down_factor)
    target_maps = roll_maps + [sg]

    if Pairs:
        if mixedPairs:
            greedy = gen_greedy_surveys(nside, nexp=nexp, target_maps=target_maps, mod_year=mod_year, day_offset=None,
                                        norm_factor=norm_factor, max_season=None)
            blobs = generate_blobs(nside, nexp=nexp, mixed_pairs=True, offset=offset, target_maps=target_maps,
                                   norm_factor=norm_factor, mod_year=mod_year, max_season=None, day_offset=None)
            surveys = [ddfs, blobs, greedy]
            run_sched(surveys, survey_length=survey_length, verbose=verbose,
                      fileroot=os.path.join(outDir, fileroot+file_end), extra_info=extra_info,
                      nside=nside)
