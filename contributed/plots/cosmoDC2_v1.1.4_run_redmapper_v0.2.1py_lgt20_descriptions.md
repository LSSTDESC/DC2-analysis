Basic QA Plots from redMaPPer v0.2.1py Run on cosmoDC2 v1.1.4
-------------------------------------------------------------

These plots are derived from the redMaPPer v0.2.1py (see
https://github.com/erykoff/redmapper for the redMaPPer python code) run on
cosmoDC2 v1.1.4.  The raw data from cosmoDC2 v1.1.4 was retrieved with the
following code:

```python
import GCRCatalogs
from astropy.table import Table

gc = GCRCatalogs.load_catalog('cosmoDC2_v1.1.4')
quantities_wanted = ['galaxy_id', 'ra', 'dec', 'redshift',
                     'mag_true_g_lsst', 'mag_true_r_lsst', 'mag_true_i_lsst',
                     'mag_true_z_lsst', 'mag_true_y_lsst', 'halo_id',
                     'halo_mass', 'is_central']
query = GCRCatalogs.GCRQuery('mag_true_z_lsst < 25.5')
data = Table(gc.get_quantities(quantities_wanted, [query]))
data.write('cosmoDC2_v1.1.4_raw.fits', format='fits')
```

Errors were applied to the truth magnitudes, and a sample of centrals and their
true redshifts were used for training.  These redshifts were the only truth
information used in the construction of the catalog.  The area of the catalog
is 500 deg^2, and there are 4768 clusters with lambda>20 and 0.2<z<1.2 in the
catalog.

### Caption for [cosmoDC2_v1.1.4_run_redmapper_v0.2.1py_lgt20_zspec.png]

This plot shows the cluster photometric redshift (z_lambda) performance.  The
x axis is the cluster photo-z, which is derived by fitting all the members to
the red-sequence model simultaneously.  The `z_spec` value is derived by taking
the median true redshift of all the members.  In the top panel the gray shaded
region shows the 1/3sigma contours, and red stars show 4-sigma outliers,
which make up 0.4% of the population.

The bottom panel shows three things.  The purple dot-dashed is the photo-z
bias.  The red dotted line is the measured redshift scatter (comparing z_lambda
to z_spec).  The cyan dashed line is the internally estimated redshift
scatter.  Both of these are very low (thanks to no systematics), and consistent
with each other.

### Caption for [cosmoDC2_v1.1.4_run_redmapper_v0.2.1py_lgt20_nz.png]

This plot shows the cluster comoving density as a function of photometric
redshift, for clusters with lambda>20.  At low redshift (z<\~0.5) the density is
consistent with that seen in SDSS and DES.  At moderately high redshift
(0.5<z<0.9) the density falls off at a slightly larger rate than I would
expect from other data, but there are uncertainties in the red-fraction
evolution, and this is what's in the cosmoDC2 data which is not unreasonable.
At the highest redshifts (z>\~0.9) I don't have anything good to compare to, so
it must look fine.
