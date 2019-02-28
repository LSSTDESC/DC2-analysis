# The functions in this file create a set of three catalogs: galaxies and stars from the DM stack measurement, and
# truth values from the extragalactic catalog.  The catalogs are matched to the same area (though not matched
# object-to-object), and the galaxy and star catalogs have shapes measured from properly rotated moments.
# The catalogs are given as the returned objects from the function get_catalogs.
# Code to cut down to regions from Jim Chiang's stack matching notebook: 
# https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/matching_stack.nbconvert.ipynb
# Code to limit to lensing-quality galaxies from Francois Lanusse and Javier Sanchez:
# https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/object_gcr_2_lensing_cuts.nbconvert.ipynb
# Code to convert I_xx, etc (second moments) to star and PSF ellipticities is from the Stile HSC module, most code written
# by Melanie Simet and Hironao Miyatake with help from Jim Bosch.

import re, os
import warnings
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import lsst.afw.geom as afw_geom
import lsst.afw.table as afw_table
import lsst.daf.persistence as dp
import lsst.afw.image as afwImage
from lsst.sims.utils import rotationMatrixFromVectors
from lsst.sims.utils import cartesianFromSpherical, sphericalFromCartesian
import GCRCatalogs
from GCR import GCRQuery
from desc_dc2_dm_data import REPOS
from .fieldRotator import FieldRotator

class RegionSelector:
    """
    Class to rotate the protoDC2 galaxies to the Run1.1p sky location and downselect those galaxies
    based on a magnitude limit and on the coordinates of the subregion (i.e., patch or CCD) being
    considered.
    """
    protoDC2_ra = 55.064
    protoDC2_dec = -29.783
    field_rotator = FieldRotator(0, 0, protoDC2_ra, protoDC2_dec)

    def __init__(self):
        pass
    
    def _set_coord_range(self, bbox, wcs):
        """
        Set the coordinate range of the region.
        
        Notes
        -----
        This method is used by the RegionSelector's subclasses.
        
        Parameters
        ----------
        bbox: Calexp.BBox
            Defines corners of region's bounding box
        wcs: Calexp.Wcs
            Defines pixel to world (sky) coordinate transformation
        """
        region_box = afw_geom.Box2D(bbox)
        corners = region_box.getCorners()
        ra_values, dec_values = [], []
        for corner in corners:
            ra, dec = wcs.pixelToSky(corner)
            ra_values.append(ra.asDegrees())
            dec_values.append(dec.asDegrees())
        self.ra_range = min(ra_values), max(ra_values)
        self.dec_range = min(dec_values), max(dec_values)
        
    def __call__(self, gc):
        """
        Return a Boolean mask indicating which items from the given catalog are in the region defined by this object.

        Parameters
        ----------
        gc: dict
            A catalog of objects (such as galaxies or stars) in dict form, including the keys 'ra_true' and 'dec_true'
            or 'ra' and 'dec'. Ideally from a get_quantities() call from a GCRCatalog, but that's not required.

        Returns
        -------
        mask: a boolean mask of the same length as gc['ra_true'] and gc['dec_true'] (or 'ra' and 'dec)')
        """
        # Rotate to the Run1.2 field if an EGC; otherwise don't.
        if 'ra_true' in gc:
            gc_ra_rot, gc_dec_rot \
                = self.field_rotator.transform(gc['ra_true'],
                                               gc['dec_true'])
        else:
            gc_ra_rot, gc_dec_rot = gc['ra'], gc['dec']

        # Select the galaxies within the region.
        mask = ((gc_ra_rot > self.ra_range[0]) &
                 (gc_ra_rot < self.ra_range[1]) &
                 (gc_dec_rot > self.dec_range[0]) &
                 (gc_dec_rot < self.dec_range[1]))
        # Return mask, and handle it in the calling function
        return mask

class PatchSelector(RegionSelector):
    """RegionSelector to use with skyMap patches, i.e., coadd data."""
    def __init__(self, butler, tract, patch):
        super(PatchSelector, self).__init__()
        # Get the patch boundaries.
        skymap = butler.get('deepCoadd_skyMap')
        tractInfo = skymap[tract]
        patchInfo = tractInfo.getPatchInfo(eval(patch))
        self._set_coord_range(patchInfo.getOuterBBox(), tractInfo.getWcs())
        
class ShearRotator:
    """ A class to compute distortions from moments, and optionally to rotate the results.  The outputs
        of the DM stack are in the local coordinate frame used for coaddition, but we want sky coordinates
        relative to ra and dec).  This class will store the info needed to rotate the moments (Ixx, Iyy, 
        Ixy, or IxxPSf, IyyPSF, IxyPSF) and additionally compute errors, if the covariance matrices are 
        also provided (Cxx etc).  Each desired quantity has a different function (g1, g2, sigma, and their
        errors) but heavy computations are stored to facilitate faster returns.
        
        If you want to use a ShearRotator for multiple catalogs (that is, multiple positions),
        then you MUST call ShearRotator.clear() between catalogs to remove the saved versions of the 
        rotated moments.
        
        Note that sigma is given in units of arcsec^2!
        
        Parameters
        ----------
        butler: lsst.daf.persistence.Butler
            A butler from which to retrieve the calibrated exposures and WCS
        tractId: int
            The number of the desired tract
        patchId: str
            "%i,%i"-formatted string indicating the desired patch
        filter_: str
            The filter to analyze
        convert_to_sky_coords: bool
            Whether to rotate to sky coordinates or leave in native coordinates (default: True)
        cache_size: int
            How many results to store in memory. Default is 8, meaning the results for 8 object
            catalogs are stored.
       """
    def __init__(self, butler, tractId, patchId, filter_, convert_to_sky_coords=True, cache_size=8):
            
        if convert_to_sky_coords:
            dataId = dict(tract=tractId, patch=patchId, filter=filter_)
            skymap = butler.get('deepCoadd_skyMap')
            self.wcs = skymap[tractId].getWcs()
        else:
            self.wcs = None
        self.clear()
        
    def clear(self):
        self.saved_transforms = None
        self.saved_moments = None
        self.saved_covariances = None        

    def getTransform(self, ra, dec):
        if self.saved_transforms:
            return self.saved_transforms
        centroids = [afw_geom.Point2D(tra, tdec) for tra, tdec in zip(ra, dec)]
        self.saved_transforms = [self.wcs.linearizePixelToSky(centroid, afw_geom.degrees).getLinear()
                                           for centroid in centroids]
        return self.saved_transforms

    def getRotatedMoments(self, ra, dec, ixx, iyy, ixy):
        if self.saved_moments:
            return self.saved_moments
        if self.wcs:
            localLinearTransform = self.getTransform(ra, dec)
            ellipses = [afw_geom.ellipses.Quadrupole(tixx, tiyy, tixy) 
                        for tixx, tiyy, tixy in zip(ixx, iyy, ixy)]
            moments = [ellipse.transform(lt) for ellipse, lt in zip(ellipses, localLinearTransform)]
            ixx = np.array([mom.getIxx() for mom in moments])
            ixy = np.array([mom.getIxy() for mom in moments])
            iyy = np.array([mom.getIyy() for mom in moments])
        self.saved_moments = (ixx, iyy, ixy)
        return ixx, iyy, ixy
    
    def getRotatedCovariance(self, ra, dec, cxx, cyy, cxy):
        if self.saved_covariances:
            return self.saved_covariances
        if self.wcs:
            localLinearTransform = self.getTransform(ra, dec)
            cov_ixx = np.zeros(cxx.shape)
            cov_iyy = np.zeros(cxx.shape)
            cov_ixy = np.zeros(cxx.shape)
            for i, (tcxx, tcxy, tcyy, lt) in enumerate(zip(cxx, cxy, cyy,localLinearTransform)):
                cov_ixx[i] = (lt[0,0]**4*tcxx +
                              (2.*lt[0,0]*lt[0,1])**2*tcyy + lt[0,1]**4*tcxy)
                cov_iyy[i] = (lt[1,0]**4*tcxx +
                              (2.*lt[1,0]*lt[1,1])**2*tcyy + lt[1,1]**4*tcxy)
                cov_ixy[i] = ((lt[0,0]*lt[1,0])**2*tcxx +
                              (lt[0,0]*lt[1,1]+lt[0,1]*lt[1,0])**2*tcyy +
                              (lt[0,1]*lt[1,1])**2*tcxy)
            self.saved_covariances = (cov_ixx, cov_iyy, cov_ixy)
            return cov_ixx, cov_iyy, cov_ixy
        self.saved_covariances = (cxx, cyy, cxy)
        return cxx, cyy, cxy
    
    def g1(self, ra, dec, ixx, iyy, ixy):
        ixx, iyy, ixy = self.getRotatedMoments(ra, dec, ixx, iyy, ixy)
        return (ixx-iyy)/(ixx+iyy)

    def g2(self, ra, dec, ixx, iyy, ixy):
        ixx, iyy, ixy = self.getRotatedMoments(ra, dec, ixx, iyy, ixy)
        return 2.*ixy/(ixx+iyy)
    
    def sigma(self, ra, dec, ixx, iyy, ixy):
        ixx, iyy, ixy = self.getRotatedMoments(ra, dec, ixx, iyy, ixy)
        return 3600*(ixx*iyy - ixy**2)**0.25 #3600: in arcsec
    
    def g1_err(self, ra, dec, ixx, iyy, ixy, cov_ixx, cov_iyy, cov_ixy):
        ixx, iyy, ixy = self.getRotatedMoments(ra, dec, ixx, iyy, ixy)
        cov_ixx, cov_iyy, cov_ixy = self.getRotatedCovariance(ra, dec, cov_ixx, cov_iyy, cov_ixy)
        dg1_dixx = 2.*iyy/(ixx+iyy)**2
        dg1_diyy = -2.*ixx/(ixx+iyy)**2
        return np.sqrt(dg1_dixx**2 * cov_ixx + dg1_diyy**2 * cov_iyy)

    def g2_err(self, ra, dec, ixx, iyy, ixy, cov_ixx, cov_iyy, cov_ixy):
        ixx, iyy, ixy = self.getRotatedMoments(ra, dec, ixx, iyy, ixy)
        cov_ixx, cov_iyy, cov_ixy = self.getRotatedCovariance(ra, dec, cov_ixx, cov_iyy, cov_ixy)
        dg2_dixx = -2.*ixy/(ixx+iyy)**2
        dg2_diyy = -2.*ixy/(ixx+iyy)**2
        dg2_dixy = 2./(ixx+iyy)
        return np.sqrt(dg2_dixx**2 * cov_ixx + dg2_diyy**2 * cov_iyy +
                            dg2_dixy**2 * cov_ixy)

    def sigma_err(self, ra, dec, ixx, iyy, ixy, cov_ixx, cov_iyy, cov_ixy):
        ixx, iyy, ixy = self.getRotatedMoments(ra, dec, ixx, iyy, ixy)
        cov_ixx, cov_iyy, cov_ixy = self.getRotatedCovariance(ra, dec, cov_ixx, cov_iyy, cov_ixy)
        sigma = self.sigma(ra, dec, ixx, iyy, ixy)
        dsigma_dixx = 0.25/sigma**3*iyy
        dsigma_diyy = 0.25/sigma**3*ixx
        dsigma_dixy = -0.5/sigma**3*ixy
        return 3600*np.sqrt(dsigma_dixx**2 * cov_ixx + dsigma_diyy**2 * cov_iyy +
                               dsigma_dixy**2 * cov_ixy)

def add_empty_columns(d, col_names):
    """ Add empty columns with names col_names to dict d, using the length of whatever is the first
        element of d.values(). """
    lend = len(list(d.values())[0])
    for col in col_names:
        d[col] = np.full(lend, np.nan)
    return d

def make_masked_column(cat, col, mask, func, input_cols):
    """ Apply the function func to some columns input_cols from dict cat, then store the results in
        column col; do this only for the items indicated by mask.  A convenience function to save
        typing.
        
        Parameters
        ----------
        cat: dict
        col: str
        mask: iterable
            Something that can mask cat[col], either a Boolean mask or a list of indices
        func: callable
        input_cols: iterable of strs
            The inputs to func should be cat[input_col[0]], cat[input_col[1]], ...
            
        Returns
        -------
        cat: dict
            cat is also altered in place.
        """
    cat[col][mask] = func(*[cat[c][mask] for c in input_cols])
    return cat

def get_butler(repo = '/global/projecta/projectdirs/lsst/global/in2p3/Run1.1/output'):
    # Create a data butler for the given repo.
    butler = dp.Butler(repo)
    return butler

def get_catalogs(tract=4851, filter_='i', mag_max=24.5,
                 repo_version='1.2p',
                 object_catalog='dc2_object_run1.2p_all_columns',
                 truth_catalog='dc2_truth_run1.2_static',
                 egc_catalog='proto-dc2_v3.0',
                 convert_to_sky_coords=True):
    """ Get a set of four catalogs: a galaxy catalog, a star catalog, an EGC catalog, and a truth catalog.  
        The galaxies and stars are from a DM run stored in `repo`, while the EGC and truth catalogs are 
        from an extragalactic catalog run.  The catalogs are only given for the sky area of the tract or 
        tracts given by the kwarg.  No object matching is performed--these are area cuts only.
        
        If convert_to_sky_coords is true, then it is assumed that the DM output shape moments are in the
        coadd reference frame, and the returned catalogs will have their shapes properly rotated to sky
        coordinates.
        
        Parameters
        ----------
        tract: int
            The tract to retrieve for this analysis.  
        filter_: str
            Use measurements from this filter (default "i")
        mag_max: float
            The maximum magnitude for objects in any catalog (default: 24.5)
        repo_version: str
            The version of the DM object catalogs you would like to access; should match object_catalog,
            truth_catalog, and egc_catalog, but we don't check this.
        object_catalog: str
            The name of the object catalog, passed to GCRCatalogs; should match repo_version, truth_catalog,
            and egc_catalog. This needs to be the "all columns" version.
        truth_catalog: str
            The name of the truth catalog, passed to GCRCatalogs; should match repo_version, object_catalog,
            and egc_catalog.
        egc_catalog: str
            The name of the EGC catalog, passed to GCRCatalogs; should match repo_version, object_catalog,
            and truth_catalog.
        convert_to_sky_coords: bool
            If true, rotate moments from the DM stack to sky coordinates before computing shapes.  (This is
            needed at least through repo_version=1.2p.)
        
        Returns
        -------
            galaxy_catalog
                A dict of measurements for galaxies in the desired region with the desired cuts
            star_catalog
                The same, but for stars
            egc_catalog
                The same, but for galaxies in the EGC
            truth_catalog
                The same, but for objects in the truth catalog
        """

    
    # First, get catalog from DM stack
    catalog = GCRCatalogs.load_catalog(object_catalog)
    catalog.add_quantity_modifier('shape_hsm_regauss_etot', 
                                  (np.hypot, 'ext_shapeHSM_HsmShapeRegauss_e1', 'ext_shapeHSM_HsmShapeRegauss_e2'), 
                                  overwrite=True)
    # Galaxy cuts are from the tutorial by F. Lanusse and J. Sanchez referenced at the top of this file
    galaxy_cuts = [
        GCRQuery('extendedness > 0'),     # Extended objects
        GCRQuery((np.isfinite, 'mag_i')), # Select objects that have i-band magnitudes
        GCRQuery('clean'), # The source has no flagged pixels (interpolated, saturated, edge, clipped...) 
                           # and was not skipped by the deblender
        GCRQuery('xy_flag == 0'),                                      # Flag for bad centroid measurement
        GCRQuery('ext_shapeHSM_HsmShapeRegauss_flag == 0'),            # Error code returned by shape measurement code
        GCRQuery((np.isfinite, 'ext_shapeHSM_HsmShapeRegauss_sigma')), # Shape measurement uncertainty should not be NaN
        GCRQuery('snr_{}_cModel > 10'.format(filter_)),                              # SNR > 10
        GCRQuery('mag_{}_cModel < {}'.format(filter_, mag_max)),                     # cModel imag brighter than 24.5
        GCRQuery('ext_shapeHSM_HsmShapeRegauss_resolution >= 0.3'), # Sufficiently resolved galaxies compared to PSF
        GCRQuery('shape_hsm_regauss_etot < 2'),                     # Total distortion in reasonable range
        GCRQuery('ext_shapeHSM_HsmShapeRegauss_sigma <= 0.4'),      # Shape measurement errors reasonable
    ]    
    
    star_cuts = [
        GCRQuery('extendedness == 0'),                            # Not extended objects!
        GCRQuery('mag_{}_cModel < {}'.format(filter_, mag_max)),  # Above flux limit (probably will be for stars)
        GCRQuery('mag_{}_cModel > 0'.format(filter_))             # A reasonable mag measurement
    ]
    tract_filter = 'tract == {}'.format(tract)
    
    # Grab the butler and the coadd skymap so we can figure out all our tracts & their patches
    butler = get_butler(REPOS[repo_version])
    skymap = butler.get('deepCoadd_skyMap')

    # Shape and location columns
    galaxy_quantities = ['ra', 'dec', 
                         'ext_shapeHSM_HsmShapeRegauss_e1', 'ext_shapeHSM_HsmShapeRegauss_e2', 'ext_shapeHSM_HsmShapeRegauss_sigma',
                         'IxxPSF_{}'.format(filter_), 'IxyPSF_{}'.format(filter_), 'IyyPSF_{}'.format(filter_),
                         'ext_shapeHSM_HsmShapeRegauss_resolution']
    star_quantities = ['ra', 'dec',
                       'Ixx_{}'.format(filter_), 'Ixy_{}'.format(filter_), 'Iyy_{}'.format(filter_),
                       'IxxPSF_{}'.format(filter_), 'IxyPSF_{}'.format(filter_), 'IyyPSF_{}'.format(filter_)]
    
    galaxy_catalog = catalog.get_quantities(galaxy_quantities, 
                                            filters=galaxy_cuts,
                                            native_filters = tract_filter)
    star_catalog = catalog.get_quantities(star_quantities, 
                                          filters=star_cuts,
                                          native_filters = tract_filter)
    galaxy_catalog = add_empty_columns(galaxy_catalog, ['psf_e1', 'psf_e2', 'psf_sigma'])
    star_catalog = add_empty_columns(star_catalog, ['e1', 'e2', 'sigma', 'psf_e1', 'psf_e2', 'psf_sigma'])

    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        egc = GCRCatalogs.load_catalog(egc_catalog)
        tc = GCRCatalogs.load_catalog(truth_catalog)
    bandname = 'mag_true_{}_lsst'.format(filter_)
    egc_cuts = ['{} < {}'.format(bandname, mag_max)]
    egc_catalog = egc.get_quantities(['galaxy_id', 'ra_true', 'dec_true', 
                                      'size_true', 'ellipticity_1_true', 'ellipticity_2_true'], 
                                      filters=egc_cuts)
    bandname = 'mag_true_{}'.format(filter_)
    tc_cuts = ['{} < {}'.format(bandname, mag_max)]
    truth_catalog = tc.get_quantities(['ra', 'dec', 'object_id', 'star', 'sprinkled'],
                                   filters=tc_cuts)
    egc_masks = []
    truth_masks = []
    this_tract = skymap[tract]
    for patch in this_tract:
        patchId = '%d,%d' % patch.getIndex()
        region_selector = PatchSelector(butler, this_tract.getId(), patchId)
        shear_rotator = ShearRotator(butler, this_tract.getId(), patchId, filter_, convert_to_sky_coords=convert_to_sky_coords)
        mask = region_selector(galaxy_catalog)
        make_masked_column(galaxy_catalog, 'psf_e1', mask, shear_rotator.g1,
                           ['ra', 'dec', 'IxxPSF_{}'.format(filter_), 'IyyPSF_{}'.format(filter_), 'IxyPSF_{}'.format(filter_)])
        make_masked_column(galaxy_catalog, 'psf_e2', mask, shear_rotator.g2,
                           ['ra', 'dec', 'IxxPSF_{}'.format(filter_), 'IyyPSF_{}'.format(filter_), 'IxyPSF_{}'.format(filter_)])
        make_masked_column(galaxy_catalog, 'psf_sigma', mask, shear_rotator.sigma,
                           ['ra', 'dec', 'IxxPSF_{}'.format(filter_), 'IyyPSF_{}'.format(filter_), 'IxyPSF_{}'.format(filter_)])
        shear_rotator.clear()
        mask = region_selector(star_catalog)
        make_masked_column(star_catalog, 'psf_e1', mask, shear_rotator.g1,
                           ['ra', 'dec', 'IxxPSF_{}'.format(filter_), 'IyyPSF_{}'.format(filter_), 'IxyPSF_{}'.format(filter_)])
        make_masked_column(star_catalog, 'psf_e2', mask, shear_rotator.g2,
                           ['ra', 'dec', 'IxxPSF_{}'.format(filter_), 'IyyPSF_{}'.format(filter_), 'IxyPSF_{}'.format(filter_)])
        make_masked_column(star_catalog, 'psf_sigma', mask, shear_rotator.sigma,
                           ['ra', 'dec', 'IxxPSF_{}'.format(filter_), 'IyyPSF_{}'.format(filter_), 'IxyPSF_{}'.format(filter_)])
        shear_rotator.clear()
        make_masked_column(star_catalog, 'e1', mask, shear_rotator.g1,
                           ['ra', 'dec', 'Ixx_{}'.format(filter_), 'Iyy_{}'.format(filter_), 'Ixy_{}'.format(filter_)])
        make_masked_column(star_catalog, 'e2', mask, shear_rotator.g2,
                           ['ra', 'dec', 'Ixx_{}'.format(filter_), 'Iyy_{}'.format(filter_), 'Ixy_{}'.format(filter_)])
        make_masked_column(star_catalog, 'sigma', mask, shear_rotator.sigma,
                           ['ra', 'dec', 'Ixx_{}'.format(filter_), 'Iyy_{}'.format(filter_), 'Ixy_{}'.format(filter_)])
        mask = region_selector(truth_catalog)
        truth_masks.append(mask)
        mask = region_selector(egc_catalog)
        egc_masks.append(mask)
    # This does a one-line "or" for all those truth masks, in a memory-expensive kind of way
    truth_mask = np.any(np.array(truth_masks), axis=0)
    truth_catalog = {k: truth_catalog[k][truth_mask] for k in truth_catalog}
    egc_mask = np.any(np.array(egc_masks), axis=0)
    egc_catalog = {k: egc_catalog[k][egc_mask] for k in egc_catalog}
    
    galaxy_mask = np.logical_not(np.any(np.array([np.isnan(galaxy_catalog['psf_e1']), 
                                        np.isnan(galaxy_catalog['psf_e2']), 
                                        np.isnan(galaxy_catalog['psf_sigma'])]), axis=0))
    galaxy_catalog = {k: galaxy_catalog[k][galaxy_mask] for k in galaxy_catalog}

    star_mask = np.logical_not(np.any(np.array([np.isnan(star_catalog['psf_e1']), 
                                      np.isnan(star_catalog['psf_e2']), 
                                      np.isnan(star_catalog['psf_sigma']),
                                      np.isnan(star_catalog['e1']), 
                                      np.isnan(star_catalog['e2']), 
                                      np.isnan(star_catalog['sigma'])]), axis=0))
    star_catalog = {k: star_catalog[k][star_mask] for k in star_catalog}

    return galaxy_catalog, star_catalog, egc_catalog, truth_catalog

