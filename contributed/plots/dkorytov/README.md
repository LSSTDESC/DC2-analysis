## A Color-Redshift DESCQA Test

# Description 

This is a simple test that plots the population density
of galaxies by redshift and color. This test verifies that the
redshift interpolation of galaxy properties from snapshots galaxy
library (Galacticus) is done a smooth manner and very readily displays
any color discreteness in redshift space. I've been using this test in
production of cosmoDC2 for and I've been wanting to implement it in
the DESCQA framework so other people can see the results, use it and
to have easier time managing the outputs.

The yaml file specifies which color to plot (bands, sdds/lsst, rest or
observer frame) and any cuts to apply to the data. Data can be cut on:
faintest rest and observer frame r-band, minimum stellar mass, minimum
host halo mass, central or not, synthetic or not, and red sequence or
not.

# DESCQA link:

https://portal.nersc.gov/project/lsst/descqa/v2/?run=2019-01-07_39&catalog=cosmoDC2_v1.1.4_small
