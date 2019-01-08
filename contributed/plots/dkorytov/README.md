## A Color-Redshift DESCQA Test

# Description 

This is a simple test that plots the population density of galaxies by
redshift and color. This test verifies that the redshift interpolation
of galaxy properties from the snapshots galaxy library (Galacticus) is
done a smooth manner and very readily displays any color discreteness
in redshift space. I've been using this test in production of cosmoDC2
for and I've been wanting to implement in the DESCQA framework so
other people can see the results, use it and, for myself, to have an
easier time managing/sharing the plots.

The yaml file specifies which color to plot (bands, sdss/lsst, rest or
observer frame) and any cuts to apply to the data. Data can be cut on:
faintest rest and observer frame r-band, minimum stellar mass, minimum
host halo mass, central or not, synthetic or not, and red sequence or
not (as specified in baseDC2).

# Figs

Figure 1 is restframe colors. Fig 2 & 3 are observer colors. Figure 4
is applying a multitude of cuts. 


# DESCQA link:

https://portal.nersc.gov/project/lsst/descqa/v2/?run=2019-01-07_39&catalog=cosmoDC2_v1.1.4_small
