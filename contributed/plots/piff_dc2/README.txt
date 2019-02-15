Submitted by Mike Jarvis

The plots in this directory were made by Piff running on
DC2 1.2i field 00261962-i using the yaml configuration file
piff_single_1.2i.yaml, also in this directory.

The plots include:

1. Rho statistics (rho.png).  The most notable is rho_1, which is the
   auto-correlation of the residual shapes between the model and the data.
   This is one of the key metrics of PSF fidelity for weak lensing studies.
   The results look quite good, as do the other 4 rhos stats, which involve
   different auto- and cross-correlations that are relevant to shear
   measurement errors.

   Note that rho_1 here is significantly noisier than the plots from large data
   surveys (e.g. DES http://adsabs.harvard.edu/abs/2016MNRAS.460.2245J).
   This is expected as this is just a single exposure.  The main takeaway
   at this point is that the measurement is broadly consistent with no signal.
   We will be doing a larger scale test of this using many more DC2 images,
   at which point we will be able to make more significant claims about the
   quality of the PSF reconstruction on these data.

2. Spatial distribution of size and shape errors (twod.png).  These used to
   show significant outliers that were too numerous for Piff to do a good job
   of removing them.  But switching to the DM PSF stars for the input list
   works well, and there don't seem to be any big outliers present.

3. Histogram of the size and shape measurements and residuals (shape.png).
   These probably aren't so informative, but it is the other statistic we have
   currently built into Piff.  They seem to match up fairly well, and the
   residuals are pretty well centered at zero, so that's good.

The Piff configuration file used is also submitted in the same directory
(`contributed/plots/piff_dc2`). This was run at nersc using the command
`piffify piff_single_1.2i.yaml`.

cf. http://rmjarvis.github.io/Piff/html/piffify.html for details about how
to run the `piffify` command.

Also, see http://rmjarvis.github.io/Piff/html/stats.html for more details
about the statistics we plotted and included in this directory.

The Piff version used was a development branch, called dc2. It has since
been merged to master, and will be included in Piff version >= 0.4.

It turned out that there were a number of changes required to Piff to get it
to properly read in the LSST DM data products. These are in Piff PR #79.

cf. https://github.com/rmjarvis/Piff/pull/79

The main change that we may want to push back on DM about is that their
variance plane images include the signal as part of the variance. This is
almost never the right thing to use for model fitting.

Piff expects the variance to just include noise from the sky, read noise, dark
current, etc. But specifically, **not** the signal. So I had to add code that
subtracts off the signal from the variance to get it in the format that Piff is
expecting.

@esheldon ran into the same issue, and it will probably keep cropping up.
The easiest long term solution, I think, would be for DM to add another hdu to
the calexp files with a normal weight image = 1/variance, where the variance
does not include the poisson shot noise from the signal.


