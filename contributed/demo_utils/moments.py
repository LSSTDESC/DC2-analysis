from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import units

def calculate_total_flux(df):
    df = df.set_index(['objectId', 'ccdVisitId'])
    df['apFlux'] = df.groupby(['objectId', 'ccdVisitId'])['flux'].sum()
    df = df.reset_index()
    return df

def calculate_1st_moments(df):
    flux_ratio = df['flux'].values/df['apFlux'].values
    df['Ix_contrib'] = df['ra'].values*flux_ratio
    df['Iy_contrib'] = df['dec'].values*flux_ratio
    df = df.set_index(['objectId', 'ccdVisitId'])
    df['Ix'] = df.groupby(['objectId', 'ccdVisitId'])['Ix_contrib'].sum()
    df['Iy'] = df.groupby(['objectId', 'ccdVisitId'])['Iy_contrib'].sum()
    df = df.reset_index()
    return df

def calculate_centered_2nd_moments(e, phi, sigma):
    sqrt_q = ((1.0 - e)/(1.0 + e))**0.5
    lam1 = sigma**2.0/sqrt_q
    lam2 = sigma**2.0*sqrt_q
    cos = np.cos(phi)
    sin = np.sin(phi)
    Ixx = lam1*cos**2.0 + lam2*sin**2.0
    Iyy = lam1*sin**2.0 + lam2*cos**2.0
    Ixy = (lam1 - lam2)*cos*sin
    return Ixx, Iyy, Ixy

def calculate_2nd_moments(df):
    e = df['e']
    phi = df['phi']
    gauss_sigma = df['gauss_sigma']
    flux_ratio = df['flux']/df['apFlux']
    ra = df['ra']
    dec = df['dec']
    reference_Ix = df['Ix']
    reference_Iy = df['Iy']
    Ixx, Iyy, Ixy = calculate_centered_2nd_moments(e=e, phi=phi, sigma=gauss_sigma)
    df['Ixx_contrib'] = flux_ratio*(Ixx + (ra - reference_Ix)**2.0)
    df['Iyy_contrib'] = flux_ratio*(Iyy + (dec - reference_Iy)**2.0)
    df['Ixy_contrib'] = flux_ratio*(Ixy + (ra - reference_Ix)*(dec - reference_Iy))
    df = df.set_index(['objectId', 'ccdVisitId'])
    for mom in ['Ixx', 'Iyy', 'Ixy']:
        df[mom] = df.groupby(['objectId', 'ccdVisitId'])['%s_contrib' %mom].sum()
    df = df.reset_index()
    return df

def apply_environment(df, add_flux_noise=True):
    df['Ixx'] += df['Ixx_PSF']
    df['Iyy'] += df['Ixx_PSF']
    df['apFlux'] += np.random.normal(0.0, df['apFluxErr'].values)
    return df

def collapse_mog(mog_df):
    collapsed = mog_df.groupby(['objectId', 'ccdVisitId',])['apFlux', 'Ix', 'Iy', 'Ixx', 'Iyy', 'Ixy', 
                                                           'apFluxErr', 'sky', 'Ixx_PSF', 'expMJD',
                                                           'num_gal_neighbors',
                                                           'num_star_neighbors', 'num_agn_neighbors', 'num_sprinkled_neighbors'].mean()
    collapsed[['Ix', 'Iy']] = units.arcsec_to_deg(collapsed[['Ix', 'Iy']])
    collapsed = collapsed.reset_index()
    return collapsed