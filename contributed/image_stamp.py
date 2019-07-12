import lsst.afw.display as afwDisplay
from lsst.afw.math import Warper
import lsst.afw.geom as afwGeom
from lsst.daf.persistence import Butler
from lsst.geom import SpherePoint
import lsst.geom

import matplotlib.pyplot as plt

def cutout_ra_dec(butler, data_id, ra, dec, dataset_type='deepDiff_differenceExp',
                  cutout_size=75, warp_to_exposure=None, **kwargs):
    """
    Produce a cutout from dataset_type from the given butler at the given ra, dec
    
    Notes
    -----
    Trivial wrapper around 'cutout_spherepoint'
    
    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Loaded DM Butler providing access to a data repository
    data_id: Butler data ID
        E.g., {'visit': 1181556, 'detector': 45, 'filter': 'r'}
    ra: float
        Right ascension of the center of the cutout, degrees
    dec: float
        Declination of the center of the cutout, degrees
    cutout_size: int [optional] 
        Side of the cutout region in pixels.  Region will be cutout_size x cutout_size.
    warp_to_exposure: optional
        Warp coadd to system of specified 'exposure', e.g., the visit image, to warp the coadd to
        before making the cutout.  The goal is to that a cut out of a coadd image
        and a cutout of a visit image should line up.
        'warp_to_exposure' overrides setting of 'cutout_size'.
         
    Returns
    -------
    MaskedImage
    """
    cutout_extent = afwGeom.ExtentI(cutout_size, cutout_size)
    radec = SpherePoint(ra, dec, afwGeom.degrees)
   
    image = butler.get(dataset_type, dataId=data_id)

    xy = afwGeom.PointI(image.getWcs().skyToPixel(radec))
    bbox = afwGeom.BoxI(xy - cutout_extent//2, cutout_extent)
    
    if warp_to_exposure is not None:
        warper = Warper(warpingKernelName='lanczos4')
        cutout_image = warper.warpExposure(warp_to_exposure.getWcs(), image,
                                           destBBox=warp_to_exposure.getBBox())
    else:
        cutout_image = image.getCutout(radec, cutout_extent)
    
    return cutout_image


def make_cutout_image(butler, data_id, ra, dec,
                      title=None,
                      frame=None, display=None, backend='matplotlib',
                      show=True, saveplot=False, savefits=False,
                      zscale=None,
                      dataset_type='deepCoadd',
                      **kwargs):
    """
    Generate and optionally display and save a postage stamp for a given RA, Dec.
    
    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Servant providing access to a data repository
    data_id:
        DM Butler Data Id
    ra: float
        Right ascension of the center of the cutout, degrees
    dec: float
        Declination of the center of the cutout, degrees
    filter: string 
        Filter of the image to load
    Returns
    -------
    MaskedImage

    Notes
    -----
    Uses lsst.afw.display with matplotlib to generate stamps.  Saves FITS file if requested.
    """
    cutout_image = cutout_ra_dec(butler, data_id, ra, dec, dataset_type=dataset_type, **kwargs)
    
    if savefits:
        if isinstance(savefits, str):
            filename = savefits
        else:
            filename = 'postage-stamp.fits'
        cutout_image.writeFits(filename)
    
    radec = SpherePoint(ra, dec, afwGeom.degrees)
    xy = cutout_image.getWcs().skyToPixel(radec)
    
    if display is None:
        display = afwDisplay.Display(frame=frame, backend=backend)

    display.mtv(cutout_image)
    display.scale("linear", "zscale")
    display.dot('o', xy.getX(), xy.getY(), ctype='red')
    display.show_colorbar()

    plt.xlabel('x')
    plt.ylabel('y')
    if title is not None:
        plt.title(title)

    if saveplot:
        if isinstance(saveplot, str):
            filename = saveplot
        else:
            filename = 'postage-stamp.png'
        plt.savefig(filename)
    if show:
        plt.show()

    return cutout_image
