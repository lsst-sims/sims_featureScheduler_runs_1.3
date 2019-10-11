Test that the code gives the same results accross different platforms


Putting rounds in the kdtree made things come out differently


1st check:  Let's make sure we can make identical runs on the same hardware (e.g., there's not an unseeded random somewhere)

run on master with:
python baselines.py --survey_length 366 --verbose

after putting in a bunch of int_rounding and turning the 


oh, I could just read observations back into the scheduler and do the checksum that way.

