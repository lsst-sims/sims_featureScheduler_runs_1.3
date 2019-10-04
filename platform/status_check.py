import numpy as np
from lsst.sims.featureScheduler.utils import schema_converter
import argparse
from baselines import gen_greedy_surveys, generate_blobs
import lsst.sims.featureScheduler.detailers as detailers
from lsst.sims.featureScheduler.surveys import (generate_dd_surveys)
from lsst.sims.featureScheduler.schedulers import Core_scheduler
import matplotlib.pylab as plt
import healpy
from lsst.sims.featureScheduler.modelObservatory import Model_observatory


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--max_indx", type=int, default=100)
    args = parser.parse_args()

    filename = args.db
    max_indx = args.max_indx

    sc = schema_converter()

    observations = sc.opsim2obs(filename)

    nside = 32
    nexp = 1
    max_dither = 0.7
    per_night = True

    dither_detailer = detailers.Dither_detailer(per_night=per_night, max_dither=max_dither)
    details = [detailers.Camera_rot_detailer(min_rot=-87., max_rot=87.), dither_detailer]
    ddfs = generate_dd_surveys(nside=nside, nexp=nexp, detailers=details)
    greedy = gen_greedy_surveys(nside, nexp=nexp)
    blobs = generate_blobs(nside, nexp=nexp, mixed_pairs=True)
    surveys = [ddfs, blobs, greedy]

    scheduler = Core_scheduler(surveys, nside=nside)

    for obs in observations[0:max_indx]:
        scheduler.add_observation(obs)

    observatory = Model_observatory(nside=nside)
    observatory.mjd = obs['mjd']
    _ = observatory.observe(obs)
    observatory.mjd = obs['mjd']
    _ = observatory.observe(obs)

    conditions = observatory.return_conditions()
    scheduler.update_conditions(conditions)

    rewards = []
    for survey_list in scheduler.survey_lists:
        rw = []
        for survey in survey_list:
            reward = survey.calc_reward_function(conditions)
            if np.size(reward) > 1:
                reward = reward[np.isfinite(reward)]
                if np.size(reward) > 0:
                    reward = np.sum(reward)
                else:
                    reward = np.inf
            rw.append(reward)
        rewards.append(rw)

    print(rewards)
