# Solution for Challenge 2 of DC2 Coadd Run1.1p GCR tutorial Part III: Guided Challenges
import numpy as np
import matplotlib.pyplot as plt
import GCRCatalogs
from GCR import GCRQuery

catalog = GCRCatalogs.load_catalog('dc2_coadd_run1.1p')

filters=[
         GCRQuery('extendedness == 0'),
         GCRQuery('clean'), 
         GCRQuery('blendedness < 10**(-0.375)'),
        ~GCRQuery('I_flag'),
         GCRQuery('i_SNR > 21')
]

g1_modif = lambda ixx,iyy,ixy: (ixx-iyy)/(ixx+iyy)
g2_modif = lambda ixx,iyy,ixy:  2.*ixy/(ixx+iyy)
sigma_modif = lambda ixx,iyy,ixy: (ixx*iyy - ixy**2)**0.25


catalog.add_derived_quantity('g1', g1_modif, 'Ixx', 'Iyy', 'Ixy')
catalog.add_derived_quantity('g2', g2_modif, 'Ixx', 'Iyy', 'Ixy')
catalog.add_derived_quantity('sigma', sigma_modif, 'Ixx', 'Iyy', 'Ixy')

catalog.add_derived_quantity('psf_g1', g1_modif, 'IxxPSF', 'IyyPSF', 'IxyPSF')
catalog.add_derived_quantity('psf_g2', g2_modif, 'IxxPSF', 'IyyPSF', 'IxyPSF')
catalog.add_derived_quantity('psf_sigma', sigma_modif, 'IxxPSF', 'IyyPSF', 'IxyPSF')

quantities = ['ra', 'dec', 
              'mag_i', 'i_SNR', 'psf_fwhm_i',
              'g1', 'g2', 'sigma',
              'psf_g1', 'psf_g2', 'psf_sigma']

# Would be hidden
data = catalog.get_quantities(quantities, 
                              native_filters=[(lambda x: x==4850, 'tract')],
                              filters=filters)

plt.figure(figsize=(10,5))

plt.subplot(121)
plt.hist2d(data['mag_i'], (data['sigma'] - data['psf_sigma'])/data['psf_sigma'], 100, range=[[15,23],[-0.02,0.02]]);
plt.xlabel('i mag')
plt.ylabel('$f \delta_\sigma$')
plt.colorbar(label='Number of objects')
plt.subplot(122)
plt.hist2d(data['psf_fwhm_i'], (data['sigma'] - data['psf_sigma'])/data['psf_sigma'], 100, range=[[0.4,1.0],[-0.02,0.02]]);
plt.xlabel('seeing FWHM (arcsec)')
plt.colorbar(label='Number of objects');

plt.savefig('plot1.png')


plt.figure(figsize=(15,10))

plt.subplot(221)
plt.hist2d(data['mag_i'], (data['g1'] - data['psf_g1']), 100, range=[[15,23],[-0.02,0.02]]);
plt.xlabel('i mag')
plt.ylabel('$g_1 - g_1^{PSF}$')
plt.colorbar(label='Number of objects')
plt.subplot(222)
plt.hist2d(data['psf_fwhm_i'], (data['g1'] - data['psf_g1']), 100, range=[[0.4,1.0],[-0.02,0.02]]);
plt.xlabel('seeing FWHM (arcsec)')
plt.ylabel('$g_1 - g_1^{PSF}$')
plt.colorbar(label='Number of objects')
plt.subplot(223)
plt.hist((data['g1'] - data['psf_g1']), 100, range=[-0.04,0.04]);
plt.xlabel('$g_1 - g_1^{PSF}$')
plt.axvline(0)
plt.subplot(224)
plt.hist((data['g2'] - data['psf_g2']), 100, range=[-0.04,0.04]);
plt.xlabel('$g_2 - g_2^{PSF}$')
plt.axvline(0)

plt.savefig('plot2.png')


