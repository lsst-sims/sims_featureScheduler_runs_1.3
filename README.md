# sims_featureScheduler_runs_1.3
Scheduler simulations


# Run list

## AGN_DDF

Runs where deep drilling fields are executed as specified in the AGN DDF white paper. We vary the amount of time the u filter is loaded. These are much shorted DDF sequences compared to our standard DDFs.

## DCR

Experiments where some observations in u and g are intentionally taken at high airmass. We vary the number of high-airmass observations that are desired per year.

## DDF_reward

Experiment to try and get the deep drilling fields more balanced.

## DESC_DDF

The deep drilling strategy presented in the DESC DDF white paper, varying how long the u filter is loaded. 

## alt_detail

Minor experiment to keep slews small especially in altitude.

## alt_sched

Simulation similar to the altSched strategy where the entire survey footprint it observed uniformly, alternating observing north and south each day. 

## baseline

The baseline attempts to observe pairs in different filters (g+r, r+i, or i+z) with 1x30s visits. We also test observing with 2x15s visits and pairs taken in the same filter.

## bulge

Observing the bulge and galactic with a variety of footprints and driving certain cadences

## euclid_DDF

Adding 2 pointings as a DDF to cover the Euclid DDF area.

## filter_load

Vary how long the u filter stays loaded around new moon.

## footprints

A variety of potential survey footprints

## maf

No runs, just a place to put MAF results

## notebooks

No runs, just a place for quick notebooks looking at things

## platform

Directory for testing that the code is returning identical results cross-platform. 

## presto

Taking pairs in widely separated filters, and testing observing a third observation.

## roll_alt

An experiment where we combine rolling cadence with altSched. 

## roll_alt_dust

Same as roll_alt, but with the survey footprint set to avoid high extinction areas.

## rolling_cadence

A variety of rolling cadence strategies.

## slurm

No runs, just a place to put slurm scripts to run on cluster

## templates

Testing how much weight to place on collecting full-footprint imaging in all filters every year.

## twilight_neo

Looking at executing a NEO survey in twilight time. 

## wfd_depth

Baseline-like surveys without any deep drilling fields, looking at what fraction of time needs to be given to the WFD area to still meet SRD requirements. 

## wfd_only

Running a footprint that is just the WFD area to test how much contingency the system has.
