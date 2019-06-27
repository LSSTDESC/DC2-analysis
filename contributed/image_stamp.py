import lsst.afw.display as afwDisplay
import lsst.afw.geom as afwGeom
from lsst.daf.persistence import Butler
from lsst.geom import SpherePoint
import lsst.geom

import matplotlib.pyplot as plt


def cutout_visit_ra_dec(butler, ra, dec, dataId=None, datasetType='deepDiff_differenceExp', cutoutSideLength=51, **kwargs):
    """
    Produce a cutout from datasetType from the given butler at the given ra, dec
    
    Notes
    -----
    Trivial wrapper around 'cutout_spherepoint'
    
    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Servant providing access to a data repository
    ra: float
        Right ascension of the center of the cutout, degrees
    dec: float
        Declination of the center of the cutout, degrees
    dataId: Butler data ID.  E.g., {'visit': 1181556, 'detector': 45, 'filter': 'r'}
    cutoutSideLength: float [optional] 
        Side of the cutout region in pixels.
         
    Returns
    -------
    MaskedImage
    """
    cutoutSize = afwGeom.ExtentI(cutoutSideLength, cutoutSideLength)
    radec = SpherePoint(ra, dec, afwGeom.degrees)

    image_wcs = butler.get(datasetType + '_wcs', dataId=dataId)
    xy = afwGeom.PointI(image_wcs.skyToPixel(radec))
    bbox = afwGeom.BoxI(xy - cutoutSize//2, cutoutSize)
    
    cutout_image = butler.get(datasetType+'_sub', bbox=bbox, dataId=dataId)
    
    return cutout_image
    
    
def cutout_coadd_ra_dec(butler, ra, dec, filter='r', datasetType='deepCoadd', **kwargs):
    """
    Produce a cutout from datasetType from the given butler at the given RA, Dec in decimal degrees.
    
    Notes
    -----
    Trivial wrapper around 'cutout_coadd_spherepoint'
    
    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Servant providing access to a data repository
    ra: float
        Right ascension of the center of the cutout, degrees
    dec: float
        Declination of the center of the cutout, degrees
    filter: string
        Filter of the image to load
        
    Returns
    -------
    MaskedImage
    """
    radec = SpherePoint(ra, dec, afwGeom.degrees)
    return cutout_coadd_spherepoint(butler, radec, filter=filter, datasetType=datasetType)
    

def cutout_coadd_spherepoint(butler, radec, filter='r', datasetType='deepCoadd',
                       skymap=None, cutoutSideLength=51, **kwargs):
    """
    Produce a cutout from datasetType at the given afw SpherePoint radec position.
    
    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Servant providing access to a data repository
    radec: lsst.afw.geom.SpherePoint 
        Coordinates of the center of the cutout.
    filter: string 
        Filter of the image to load
    datasetType: string ['deepCoadd']  
        Which type of coadd to load.  Doesn't support 'calexp'
    skymap: lsst.afw.skyMap.SkyMap [optional] 
        Pass in to avoid the Butler read.  Useful if you have lots of them.
    cutoutSideLength: float [optional] 
        Side of the cutout region in pixels.
    
    Returns
    -------
    MaskedImage
    """
    cutoutSize = afwGeom.ExtentI(cutoutSideLength, cutoutSideLength)

    if skymap is None:
        skymap = butler.get("%s_skyMap" % datasetType)
    
    # Look up the tract, patch for the RA, Dec
    tractInfo = skymap.findTract(radec)
    patchInfo = tractInfo.findPatch(radec)
    xy = afwGeom.PointI(tractInfo.getWcs().skyToPixel(radec))
    bbox = afwGeom.BoxI(xy - cutoutSize//2, cutoutSize)

    coaddId = {'tract': tractInfo.getId(), 'patch': "%d,%d" % patchInfo.getIndex(), 'filter': filter}
    
    cutout_image = butler.get(datasetType+'_sub', bbox=bbox, immediate=True, dataId=coaddId)

    return cutout_image


def make_cutout_image(butler, ra, dec, dataId=None,
                      title=None,
                      frame=None, display=None, backend='matplotlib',
                      show=True, saveplot=False, savefits=False,
                      zscale=None,
                      datasetType='deepCoadd'):
    """
    Generate and optionally display and save a postage stamp for a given RA, Dec.
    
    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Servant providing access to a data repository
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
    
    if datasetType == 'deepCoadd':
        cutout_image = cutout_coadd_ra_dec(butler, ra, dec, dataId=dataId, datasetType=datasetType)
    else:
        cutout_image = cutout_visit_ra_dec(butler, ra, dec, dataId=dataId, datasetType=datasetType)

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