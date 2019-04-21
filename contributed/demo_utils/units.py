from __future__ import absolute_import, division, print_function
import numpy as np

def flux_to_mag(flux, zeropoint_mag=0.0, from_unit=None):
    if from_unit=='nMgy':
        zeropoint_mag = 22.5
    return zeropoint_mag - 2.5*np.log10(flux)

def mag_to_flux(mag, zeropoint_mag=0.0, to_unit=None):
    if to_unit=='nMgy':
        zeropoint_mag = 22.5
    return np.power(10.0, -0.4*(mag - zeropoint_mag))

def fwhm_to_sigma(fwhm):
    return fwhm/np.sqrt(8.0*np.log(2.0))

def deg_to_arcsec(deg):
    return 3600.0*deg

def arcsec_to_deg(arcsec):
    return arcsec/3600.0

def e1e2_to_phi(e1, e2):
    phi = 0.5*np.arctan(e2/e1)
    return phi

def e1e2_to_ephi(e1, e2):
    e = np.power(np.power(e1, 2.0) + np.power(e2, 2.0), 0.5)
    phi = 0.5*np.arctan(e2/e1)
    return e, phi