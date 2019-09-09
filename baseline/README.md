
New things in v1.3 compared to the v1.2 baseline

* Deep Drilling Fields constraints not as tight, so they should execute more often
* Including a basis function to drive 3 observations per year per filter
* Rotational dithering (a rotTelpos value between -87 and 87 degrees is selected nightly)
* Spatial dithering (0.7 degrees) and rotational dithering included for DDFs
* New explicit planet masking basis function (Venus, Mars, Jupiter masked)
* New detailer to help further minimize slewtimes between observations
* Decrease the number of nights the u-band is loaded per lunation

The limited u-band availability does mean there are a few years where the entire sky does not get observed in u.
