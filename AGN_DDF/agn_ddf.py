import numpy as np
import matplotlib.pylab as plt
import healpy as hp
from lsst.sims.featureScheduler.modelObservatory import Model_observatory
from lsst.sims.featureScheduler.schedulers import Core_scheduler, simple_filter_sched
from lsst.sims.featureScheduler.utils import standard_goals, calc_norm_factor, create_season_offset
import lsst.sims.featureScheduler.basis_functions as bf
from lsst.sims.featureScheduler.surveys import (Greedy_survey,
                                                Blob_survey)
from lsst.sims.featureScheduler import sim_runner
import lsst.sims.featureScheduler.detailers as detailers
import sys
import subprocess
import os
import argparse
from lsst.sims.featureScheduler.surveys import Deep_drilling_survey
import lsst.sims.featureScheduler.basis_functions as basis_functions


def dd_bfs(RA, dec, survey_name, ha_limits, frac_total=0.007, aggressive_frac=0.005):
    """
    Convienence function to generate all the feasibility basis functions
    """
    sun_alt_limit = -18.
    time_needed = 9.
    fractions = [0.00, aggressive_frac, frac_total]
    bfs = []
    bfs.append(basis_functions.Filter_loaded_basis_function(filternames=['r', 'g', 'i', 'z', 'y']))
    bfs.append(basis_functions.Not_twilight_basis_function(sun_alt_limit=sun_alt_limit))
    bfs.append(basis_functions.Time_to_twilight_basis_function(time_needed=time_needed))
    bfs.append(basis_functions.Hour_Angle_limit_basis_function(RA=RA, ha_limits=ha_limits))
    bfs.append(basis_functions.Fraction_of_obs_basis_function(frac_total=frac_total, survey_name=survey_name))
    bfs.append(basis_functions.Look_ahead_ddf_basis_function(frac_total, aggressive_frac,
                                                             sun_alt_limit=sun_alt_limit, time_needed=time_needed,
                                                             RA=RA, survey_name=survey_name,
                                                             ha_limits=ha_limits))
    bfs.append(basis_functions.Soft_delay_basis_function(fractions=fractions, delays=[0., 0.04, 1.5],
                                                         survey_name=survey_name))

    return bfs


def dd_u_bfs(RA, dec, survey_name, ha_limits, frac_total=0.0019, aggressive_frac=0.0014):
    """Convienence function to generate all the feasibility basis functions for u-band DDFs
    """
    bfs = []
    sun_alt_limit = -18.
    time_needed = 6.
    fractions = [0.00, aggressive_frac, frac_total]
    bfs.append(basis_functions.Filter_loaded_basis_function(filternames='u'))
    bfs.append(basis_functions.Not_twilight_basis_function(sun_alt_limit=sun_alt_limit))
    bfs.append(basis_functions.Time_to_twilight_basis_function(time_needed=time_needed))
    bfs.append(basis_functions.Hour_Angle_limit_basis_function(RA=RA, ha_limits=ha_limits))
    bfs.append(basis_functions.Moon_down_basis_function())
    bfs.append(basis_functions.Fraction_of_obs_basis_function(frac_total=frac_total, survey_name=survey_name))
    bfs.append(basis_functions.Look_ahead_ddf_basis_function(frac_total, aggressive_frac,
                                                             sun_alt_limit=sun_alt_limit, time_needed=time_needed,
                                                             RA=RA, survey_name=survey_name,
                                                             ha_limits=ha_limits))
    bfs.append(basis_functions.Soft_delay_basis_function(fractions=fractions, delays=[0., 0.2, 0.5],
                                                         survey_name=survey_name))

    return bfs


def generate_dd_surveys(nside=None, nexp=2, detailers=None):
    """Utility to return a list of standard deep drilling field surveys.

    XXX-Someone double check that I got the coordinates right!

    """

    surveys = []

    # ELAIS S1
    RA = 9.45
    dec = -44.
    survey_name = 'DD:ELAISS1'
    ha_limits = ([0., 1.5], [21.5, 24.])
    bfs = dd_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='grizy',
                                        nvis=[1, 1, 3, 5, 4],
                                        survey_name=survey_name, reward_value=100,
                                        nside=nside, nexp=nexp, detailers=detailers))

    survey_name = 'DD:u,ELAISS1'
    bfs = dd_u_bfs(RA, dec, survey_name, ha_limits)

    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='u',
                                        nvis=[8], survey_name=survey_name, reward_value=100, nside=nside,
                                        nexp=nexp, detailers=detailers))

    # XMM-LSS
    survey_name = 'DD:XMM-LSS'
    RA = 35.708333
    dec = -4-45/60.
    ha_limits = ([0., 1.5], [21.5, 24.])
    bfs = dd_bfs(RA, dec, survey_name, ha_limits)

    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='grizy',
                                        nvis=[1, 1, 3, 5, 4], survey_name=survey_name, reward_value=100,
                                        nside=nside, nexp=nexp, detailers=detailers))
    survey_name = 'DD:u,XMM-LSS'
    bfs = dd_u_bfs(RA, dec, survey_name, ha_limits)

    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='u',
                                        nvis=[8], survey_name=survey_name, reward_value=100, nside=nside,
                                        nexp=nexp, detailers=detailers))

    # Extended Chandra Deep Field South
    RA = 53.125
    dec = -28.-6/60.
    survey_name = 'DD:ECDFS'
    ha_limits = [[0.5, 3.0], [20., 22.5]]
    bfs = dd_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='grizy',
                                        nvis=[1, 1, 3, 5, 4],
                                        survey_name=survey_name, reward_value=100, nside=nside,
                                        nexp=nexp, detailers=detailers))

    survey_name = 'DD:u,ECDFS'
    bfs = dd_u_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='u',
                                        nvis=[8], survey_name=survey_name, reward_value=100, nside=nside,
                                        nexp=nexp, detailers=detailers))
    # COSMOS
    RA = 150.1
    dec = 2.+10./60.+55/3600.
    survey_name = 'DD:COSMOS'
    ha_limits = ([0., 2.5], [21.5, 24.])
    bfs = dd_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='grizy',
                                        nvis=[1, 1, 3, 5, 4],
                                        survey_name=survey_name, reward_value=100, nside=nside,
                                        nexp=nexp, detailers=detailers))
    survey_name = 'DD:u,COSMOS'
    bfs = dd_u_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='u',
                                        nvis=[8], survey_name=survey_name, reward_value=100, nside=nside,
                                        nexp=nexp, detailers=detailers))

    # Extra DD Field, just to get to 5. Still not closed on this one
    survey_name = 'DD:290'
    RA = 349.386443
    dec = -63.321004
    ha_limits = ([0., 1.5], [21.5, 24.])
    bfs = dd_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='grizy',
                                        nvis=[1, 1, 3, 5, 4],
                                        survey_name=survey_name, reward_value=100, nside=nside,
                                        nexp=nexp, detailers=detailers))

    survey_name = 'DD:u,290'
    bfs = dd_u_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='u', nvis=[8],
                                        survey_name=survey_name, reward_value=100, nside=nside,
                                        nexp=nexp, detailers=detailers))

    return surveys


def gen_greedy_surveys(nside, nexp=1):
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
        bfs.append(bf.Target_map_basis_function(filtername=filtername,
                                                target_map=target_map[filtername],
                                                out_of_bounds_val=np.nan, nside=nside,
                                                norm_factor=norm_factor))
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


def generate_blobs(nside, mixed_pairs=False, nexp=1, no_pairs=False, offset=None, template_weight=0.):
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
        bfs.append(bf.Target_map_basis_function(filtername=filtername,
                                                target_map=target_map[filtername],
                                                out_of_bounds_val=np.nan, nside=nside,
                                                norm_factor=norm_factor))

        if filtername2 is not None:
            bfs.append(bf.Target_map_basis_function(filtername=filtername2,
                                                    target_map=target_map[filtername2],
                                                    out_of_bounds_val=np.nan, nside=nside,
                                                    norm_factor=norm_factor))

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
              extra_info=None, illum_limit=60.):
    years = np.round(survey_length/365.25)
    scheduler = Core_scheduler(surveys, nside=nside)
    n_visit_limit = None
    observatory = Model_observatory(nside=nside)
    filter_sched = simple_filter_sched(illum_limit=illum_limit)
    observatory, scheduler, observations = sim_runner(observatory, scheduler,
                                                      survey_length=survey_length,
                                                      filename=fileroot+'%iyrs.db' % years,
                                                      delete_past=True, n_visit_limit=n_visit_limit,
                                                      verbose=verbose, extra_info=extra_info,
                                                      filter_scheduler=filter_sched)


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
    parser.add_argument("--illum_limit", type=float, default=60.)

    args = parser.parse_args()
    nexp = args.nexp
    Pairs = args.pairs
    mixedPairs = args.mixedPairs
    survey_length = args.survey_length  # Days
    outDir = args.outDir
    verbose = args.verbose
    per_night = args.perNight
    max_dither = args.maxDither
    illum_limit = args.illum_limit

    nside = 32

    extra_info = {}
    exec_command = ''
    for arg in sys.argv:
        exec_command += ' ' + arg
    extra_info['exec command'] = exec_command
    extra_info['git hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    extra_info['file executed'] = os.path.realpath(__file__)

    fileroot = 'agnddf_'+'illum%i_' % illum_limit
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

    if Pairs:
        if mixedPairs:
            greedy = gen_greedy_surveys(nside, nexp=nexp)
            blobs = generate_blobs(nside, nexp=nexp, mixed_pairs=True, offset=offset)
            surveys = [ddfs, blobs, greedy]
            run_sched(surveys, survey_length=survey_length, verbose=verbose,
                      fileroot=os.path.join(outDir, fileroot+file_end), extra_info=extra_info,
                      nside=nside, illum_limit=illum_limit)
