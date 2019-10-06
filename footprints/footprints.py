import numpy as np
import healpy as hp
import lsst.sims.featureScheduler.utils as utils
from lsst.sims.featureScheduler.utils import generate_goal_map
from astropy.coordinates import SkyCoord
from astropy import units as u
from lsst.sims.featureScheduler.utils import standard_goals, calc_norm_factor
import os
from lsst.utils import getPackageDir
from lsst.sims.utils import _angularSeparation

# OK, what are the footprints we'd like to try?

def bluer_footprint(nside=32):
    """Try a bluer filter dist. May want to turn this into a larger parameter search.
    """

    result = {}
    result['u'] = generate_goal_map(nside=nside, NES_fraction=0.,
                                    WFD_fraction=0.31, SCP_fraction=0.15,
                                    GP_fraction=0.15, WFD_upper_edge_fraction=0.)
    # Turn up the g WFD
    result['g'] = generate_goal_map(nside=nside, NES_fraction=0.2,
                                    WFD_fraction=0.9, SCP_fraction=0.15,
                                    GP_fraction=0.15, WFD_upper_edge_fraction=0.)
    result['r'] = generate_goal_map(nside=nside, NES_fraction=0.46,
                                    WFD_fraction=1.0, SCP_fraction=0.15,
                                    GP_fraction=0.15, WFD_upper_edge_fraction=0.)
    result['i'] = generate_goal_map(nside=nside, NES_fraction=0.46,
                                    WFD_fraction=1.0, SCP_fraction=0.15,
                                    GP_fraction=0.15, WFD_upper_edge_fraction=0.)
    # Turn down the z and y WFD
    result['z'] = generate_goal_map(nside=nside, NES_fraction=0.4,
                                    WFD_fraction=0.7, SCP_fraction=0.15,
                                    GP_fraction=0.15, WFD_upper_edge_fraction=0.)
    result['y'] = generate_goal_map(nside=nside, NES_fraction=0.,
                                    WFD_fraction=0.7, SCP_fraction=0.15,
                                    GP_fraction=0.15, WFD_upper_edge_fraction=0.)


    return result


def gp_smooth(nside=32):
    # Treat the galactic plane as just part of the WFD
    result = {}
    result['u'] = generate_goal_map(nside=nside, NES_fraction=0.,
                                    WFD_fraction=0.31, SCP_fraction=0.15,
                                    GP_fraction=0.31, WFD_upper_edge_fraction=0.)
    result['g'] = generate_goal_map(nside=nside, NES_fraction=0.2,
                                    WFD_fraction=0.44, SCP_fraction=0.15,
                                    GP_fraction=0.44, WFD_upper_edge_fraction=0.)
    result['r'] = generate_goal_map(nside=nside, NES_fraction=0.46,
                                    WFD_fraction=1.0, SCP_fraction=0.15,
                                    GP_fraction=1.0, WFD_upper_edge_fraction=0.)
    result['i'] = generate_goal_map(nside=nside, NES_fraction=0.46,
                                    WFD_fraction=1.0, SCP_fraction=0.15,
                                    GP_fraction=1.0, WFD_upper_edge_fraction=0.)
    result['z'] = generate_goal_map(nside=nside, NES_fraction=0.4,
                                    WFD_fraction=0.9, SCP_fraction=0.15,
                                    GP_fraction=0.9, WFD_upper_edge_fraction=0.)
    result['y'] = generate_goal_map(nside=nside, NES_fraction=0.,
                                    WFD_fraction=0.9, SCP_fraction=0.15,
                                    GP_fraction=0.9, WFD_upper_edge_fraction=0.)

    return result


def no_gp_north(nside=32):
    result = {}
    gl1 = 40.
    gl2 = 290.
    result['u'] = generate_goal_map(nside=nside, NES_fraction=0.,
                                    WFD_fraction=0.31, SCP_fraction=0.15,
                                    GP_fraction=0.15, WFD_upper_edge_fraction=0.,
                                    gp_long1=gl1, gp_long2=gl2)

    result['g'] = generate_goal_map(nside=nside, NES_fraction=0.2,
                                    WFD_fraction=0.44, SCP_fraction=0.15,
                                    GP_fraction=0.15, WFD_upper_edge_fraction=0.,
                                    gp_long1=gl1, gp_long2=gl2)
    result['r'] = generate_goal_map(nside=nside, NES_fraction=0.46,
                                    WFD_fraction=1.0, SCP_fraction=0.15,
                                    GP_fraction=0.15, WFD_upper_edge_fraction=0.,
                                    gp_long1=gl1, gp_long2=gl2)
    result['i'] = generate_goal_map(nside=nside, NES_fraction=0.46,
                                    WFD_fraction=1.0, SCP_fraction=0.15,
                                    GP_fraction=0.15, WFD_upper_edge_fraction=0.,
                                    gp_long1=gl1, gp_long2=gl2)
    result['z'] = generate_goal_map(nside=nside, NES_fraction=0.4,
                                    WFD_fraction=0.9, SCP_fraction=0.15,
                                    GP_fraction=0.15, WFD_upper_edge_fraction=0.,
                                    gp_long1=gl1, gp_long2=gl2)
    result['y'] = generate_goal_map(nside=nside, NES_fraction=0.,
                                    WFD_fraction=0.9, SCP_fraction=0.15,
                                    GP_fraction=0.15, WFD_upper_edge_fraction=0.,
                                    gp_long1=gl1, gp_long2=gl2)
    return result


def add_mag_clouds(inmap=None, nside=32):
    if inmap is None:
        result = standard_goals(nside=nside)
    else:
        result = inmap
    ra, dec = utils.ra_dec_hp_map(nside=nside)

    lmc_ra = np.radians(80.893860)
    lmc_dec = np.radians(-69.756126)
    lmc_radius = np.radians(10.)

    smc_ra = np.radians(13.186588)
    smc_dec = np.radians(-72.828599)
    smc_radius = np.radians(5.)

    dist_to_lmc = _angularSeparation(lmc_ra, lmc_dec, ra, dec)
    lmc_pix = np.where(dist_to_lmc < lmc_radius)

    dist_to_smc = _angularSeparation(smc_ra, lmc_dec, ra, dec)
    smc_pix = np.where(dist_to_smc < smc_radius)

    for key in result:
        result[key][lmc_pix] = np.max(result[key])
        result[key][smc_pix] = np.max(result[key])
    return result


def big_sky(nside=32, weights={'u': [0.31, 0.15, False], 'g': [0.44, 0.15],
                               'r': [1., 0.3], 'i': [1., 0.3], 'z': [0.9, 0.3],
                               'y': [0.9, 0.3, False]}):
    """
    Based on the Olsen et al Cadence White Paper
    """

    # wfd covers -72.25 < dec < 12.4. Avoid galactic plane |b| > 15 deg
    wfd_north = np.radians(12.4)
    wfd_south = np.radians(-72.25)
    full_north = np.radians(30.)
    g_lat_limit = np.radians(15.)

    ra, dec = utils.ra_dec_hp_map(nside=nside)
    total_map = np.zeros(ra.size)
    coord = SkyCoord(ra=ra*u.rad, dec=dec*u.rad)
    g_long, g_lat = coord.galactic.l.radian, coord.galactic.b.radian

    # let's make a first pass here

    total_map[np.where(dec < full_north)] = 1e-6
    total_map[np.where((dec > wfd_south) &
                       (dec < wfd_north) &
                       (np.abs(g_lat) > g_lat_limit))] = 1.

    # Now let's break it down by filter
    result = {}

    for key in weights:
        result[key] = total_map + 0.
        result[key][np.where(result[key] == 1)] = weights[key][0]
        result[key][np.where(result[key] == 1e-6)] = weights[key][1]
        if len(weights[key]) == 3:
            result[key][np.where(dec > wfd_north)] = 0.

    return result


def big_sky_nouiy(nside=32, weights={'u': [0.31, 0., False], 'g': [0.44, 0.15],
                                     'r': [1., 0.3], 'i': [1., 0.0, False], 'z': [0.9, 0.3],
                                     'y': [0.9, 0.0, False]}):
    result = big_sky(nside=nside, weights=weights)
    return result


def big_sky_dust(nside=32, weights={'u': [0.31, 0.15, False], 'g': [0.44, 0.15],
                 'r': [1., 0.3], 'i': [1., 0.3], 'z': [0.9, 0.3],
                 'y': [0.9, 0.3, False]}, dust_limit=0.19):
    """
    Based on the Olsen et al Cadence White Paper
    """

    ebvDataDir = getPackageDir('sims_maps')
    filename = 'DustMaps/dust_nside_%i.npz' % nside
    dustmap = np.load(os.path.join(ebvDataDir, filename))['ebvMap']

    # wfd covers -72.25 < dec < 12.4. Avoid galactic plane |b| > 15 deg
    wfd_north = np.radians(12.4)
    wfd_south = np.radians(-72.25)
    full_north = np.radians(30.)

    ra, dec = utils.ra_dec_hp_map(nside=nside)
    total_map = np.zeros(ra.size)

    # let's make a first pass here

    total_map[np.where(dec < full_north)] = 1e-6
    total_map[np.where((dec > wfd_south) &
                       (dec < wfd_north) &
                       (dustmap < dust_limit))] = 1.

    # Now let's break it down by filter
    result = {}

    for key in weights:
        result[key] = total_map + 0.
        result[key][np.where(result[key] == 1)] = weights[key][0]
        result[key][np.where(result[key] == 1e-6)] = weights[key][1]
        if len(weights[key]) == 3:
            result[key][np.where(dec > wfd_north)] = 0.

    return result


def new_regions(nside=32, north_limit=2.25):
    ra, dec = utils.ra_dec_hp_map(nside=nside)
    coord = SkyCoord(ra=ra*u.rad, dec=dec*u.rad)
    g_long, g_lat = coord.galactic.l.radian, coord.galactic.b.radian

    # OK, let's just define the regions
    north = np.where((dec > np.radians(north_limit)) & (dec < np.radians(30.)))[0]
    wfd = np.where(utils.WFD_healpixels(dec_min=-72.25, dec_max=north_limit, nside=nside) > 0)[0]
    nes = np.where(utils.NES_healpixels(dec_min=north_limit, min_EB=-30., max_EB=10.) > 0)[0]
    scp = np.where(utils.SCP_healpixels(nside=nside, dec_max=-72.25) > 0)[0]

    new_gp = np.where((dec < np.radians(north_limit)) & (dec > np.radians(-72.25)) & (np.abs(g_lat) < np.radians(15.)) &
                      ((g_long < np.radians(90.)) | (g_long > np.radians(360.-70.))))[0]

    anti_gp = np.where((dec < np.radians(north_limit)) & (dec > np.radians(-72.25)) & (np.abs(g_lat) < np.radians(15.)) &
                       (g_long < np.radians(360.-70.)) & (g_long > np.radians(90.)))[0]

    footprints = {'north': north, 'wfd': wfd, 'nes': nes, 'scp': scp, 'gp': new_gp, 'gp_anti': anti_gp}

    return footprints


def newA(nside=32):
    """
    From https://github.com/rhiannonlynne/notebooks/blob/master/New%20Footprints.ipynb

    XXX--this seems to have very strange u-band distributions
    """
    zeros = np.zeros(hp.nside2npix(nside), dtype=float)
    footprints = new_regions(north_limit=12.25)

    # Define how many visits per field we want
    obs_region = {'gp': 750, 'wfd': 839, 'nes': 255, 'scp': 200, 'gp_anti': 825, 'north': 138}

    wfd_ratio = {'u': 0.06796116504854369, 'g': 0.0970873786407767,
                 'r': 0.22330097087378642, 'i': 0.22330097087378642, 'z': 0.1941747572815534, 'y': 0.1941747572815534}
    uniform_ratio = {'u': 0.16666666666666666, 'g': 0.16666666666666666,
                     'r': 0.16666666666666666, 'i': 0.16666666666666666, 'z': 0.16666666666666666, 'y': 0.16666666666666666}

    filter_ratios = {'gp': wfd_ratio,
                     'gp_anti': wfd_ratio,
                     'nes': {'u': 0.0, 'g': 0.2, 'r': 0.4, 'i': 0.4, 'z': 0.0, 'y': 0.0},
                     'north': uniform_ratio,
                     'scp': uniform_ratio,
                     'wfd': wfd_ratio}

    results = {}
    norm_val = obs_region['wfd']*filter_ratios['wfd']['r']
    for filtername in filter_ratios['wfd']:
        results[filtername] = zeros + 0
        for region in footprints:
            if np.max(filter_ratios[region][filtername]) > 0:
                results[filtername][footprints[region]] = filter_ratios[region][filtername]*obs_region[region]/norm_val

    return results


def newB(nside=32):
    """
    From https://github.com/rhiannonlynne/notebooks/blob/master/New%20Footprints.ipynb

    XXX--this seems to have very strange u-band distributions
    """
    zeros = np.zeros(hp.nside2npix(nside), dtype=float)
    footprints = new_regions(north_limit=12.25)

    # Define how many visits per field we want
    obs_region = {'gp': 650, 'wfd': 830, 'nes': 255, 'scp': 200, 'gp_anti': 100, 'north': 207}

    wfd_ratio = {'u': 0.06796116504854369, 'g': 0.0970873786407767,
                 'r': 0.22330097087378642, 'i': 0.22330097087378642, 'z': 0.1941747572815534, 'y': 0.1941747572815534}
    uniform_ratio = {'u': 0.16666666666666666, 'g': 0.16666666666666666,
                     'r': 0.16666666666666666, 'i': 0.16666666666666666, 'z': 0.16666666666666666, 'y': 0.16666666666666666}

    filter_ratios = {'gp': wfd_ratio,
                     'gp_anti': wfd_ratio,
                     'nes': {'u': 0.0, 'g': 0.2, 'r': 0.4, 'i': 0.4, 'z': 0.0, 'y': 0.0},
                     'north': uniform_ratio,
                     'scp': uniform_ratio,
                     'wfd': wfd_ratio}

    results = {}
    norm_val = obs_region['wfd']*filter_ratios['wfd']['r']
    for filtername in filter_ratios['wfd']:
        results[filtername] = zeros + 0
        for region in footprints:
            if np.max(filter_ratios[region][filtername]) > 0:
                results[filtername][footprints[region]] = filter_ratios[region][filtername]*obs_region[region]/norm_val

    return results


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


def stuck_rolling(nside=32, scale_down_factor=0.2):
    """A bit of a trolling footprint. See what happens if we use a rolling footprint, but don't roll it. 
    """
    sg = standard_goals()
    footprints = slice_wfd_area(2, sg, scale_down_factor=scale_down_factor)
    # Only take the first set
    footprints = footprints[0]
    return footprints


def wfd_only(nside=32):
    sg = standard_goals()
    not_wfd = np.where(sg['r'] != 1.)
    for key in sg:
        sg[key][not_wfd] = 0
    return sg





