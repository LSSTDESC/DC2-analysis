## A Color-Redshift DESCQA Test

# Description 

This HackUr DC2 project demonstrates a new DESCQA test that plots the
population density of galaxies by redshift and color. The color-z test
verifies that the redshift interpolation of galaxy properties from the
snapshots galaxy library (Galacticus) is done a smooth manner and
readily displays any color discreteness in redshift space. I've been
using this test in production of cosmoDC2 for a while and I wanted to
implement in the DESCQA framework so other people can see the results
more easily, apply the test themselves and, for myself, to have an
easier time managing/sharing the plots.

The generic instructions on how to install and run your own DESCQA
tests can be found here:
https://github.com/LSSTDESC/descqa/blob/master/CONTRIBUTING.md As of
writing this note, the color-z test has not been incorporated into
master branch of DESCQA, so you will need to check out the github pull
request containing the new test with `git pull origin pull/164/head`. 

The color-z test comes with default test configurations labeled as
"color_z", "color_z_red_sequence" and "color_z_restframe". They can be
run similarly as other test by calling `./run_master.sh -t [test_name]`.
"color_z"/"color_z_restframe" plots observer/restframe
colors for all galaxies. "color_z_red_sequence" plots the observed colors
for galaxies classified as red sequence galaxies in clusters. 

You can create your own configuration by modifying or copying yaml
files found in "../descqa/descqa/configs/color_z*.yaml". The yaml
files specifies which test to run (color-z in our case) and how to configure
the test. Options for the color-z test include which color to plot
(bands, sdss/lsst, rest or observer frame) and which cuts to apply to
the data. Data can be cut on: highest (i.e. dimmest) rest and observer
frame r-band magnitudes, minimum stellar mass, minimum host halo mass,
central or not, synthetic or not, and red sequence or not (as
specified in baseDC2). If you created a new yaml file called
`new_color_z_config.yaml`, you can run it with `./run_master.sh -t
new_color_z_config`. 

# Figs

Figure 1 is restframe colors. Fig 2 & 3 are observer colors. Figure 4
is applying a multitude of cuts. If any cuts are applied, the cuts are
displayed as text in the plot itself.


# DESCQA link:

https://portal.nersc.gov/project/lsst/descqa/v2/?run=2019-01-07_39&catalog=cosmoDC2_v1.1.4_small
