from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from itertools import product
import units
import moments

def _format_obs_history(obs_history, field, save_to_disk=None):
    """
    Parameters
    ----------
    obs_history : Pandas.DataFrame
    field : Pandas.DataFrame
    save_to_disk : str
    
    Note
    ----
    We use the dithered RA, Dec and express all positions in arcsec.
    
    Returns
    -------
    DataFrame obs_history, formatted with new column conventions and units
    """
    # Join with Field table
    obs_history = pd.merge(obs_history, field, left_on='Field_fieldID', right_on='fieldID')
    # Some unit conversion and column renaming
    # NOTE: OpSim DB defines dithered positions as offset from the field center.
    obs_history['ditheredRA'] = units.deg_to_arcsec(obs_history['ditheredRA'].values + obs_history['fieldRA'].values)
    obs_history['ditheredDec'] = units.deg_to_arcsec(obs_history['ditheredDec'].values  + obs_history['fieldDec'].values)
    obs_history['Ixx_PSF'] = units.fwhm_to_sigma(obs_history['finSeeing'].values)**2.0
    obs_history['apFluxErr'] = units.mag_to_flux(obs_history['fiveSigmaDepth'].values-22.5)/5.0
    obs_history = obs_history.rename({'filtSkyBrightness': 'sky', 'obsHistID': 'ccdVisitId'}, axis=1)
    # Only keep columns we'll need
    obs_keep_cols = ['ccdVisitId', 'Field_fieldID', 'expMJD', 
                     'ditheredRA', 'ditheredDec', 'Ixx_PSF', 'apFluxErr', 'sky', 'filter',]
    obs_history = obs_history[obs_keep_cols]
    obs_history['ccdVisitId'] = obs_history['ccdVisitId'].astype(np.int64)
    if save_to_disk is not None:
        obs_history.to_csv(save_to_disk, index=False)
    return obs_history

'''
def _format_field(field, save_to_disk=None):
    # NOTE: OpSim DB documentation says RA, Dec are in radians but they seem to be in degrees.
    # field[['fieldRA', 'fieldDec']] = units.deg_to_arcsec(np.rad2deg(field[['fieldRA', 'fieldDec']]))
    if save_to_disk is not None:
        field.to_csv(save_to_disk, index=False)
    return field
'''

def _format_extragal_catalog(galaxies, save_to_disk=None):
    # Unit conversion and column renaming
    filters = list('ugrizy')
    galaxies.columns = map(str.lower, galaxies.columns)
    galaxies[['ra', 'dec']] = units.deg_to_arcsec(galaxies[['ra_true', 'dec_true']])
    galaxies['phi_bulge'] = units.e1e2_to_phi(e1=galaxies['ellipticity_1_bulge_true'].values,
                                              e2=galaxies['ellipticity_2_bulge_true'].values)
    galaxies['phi_disk'] = units.e1e2_to_phi(e1=galaxies['ellipticity_1_disk_true'].values,
                                             e2=galaxies['ellipticity_2_disk_true'].values)
    galaxies['disk_to_total_ratio'] = 1.0 - galaxies['bulge_to_total_ratio_i']
    for bp in 'ugrizy':
        galaxies['flux_%s' %bp] = units.mag_to_flux(galaxies['mag_true_%s_lsst' %bp].values, to_unit='nMgy')
        galaxies['flux_disk_%s' %bp] = galaxies['flux_%s' %bp].values*galaxies['disk_to_total_ratio'].values
        galaxies['flux_bulge_%s' %bp] = galaxies['flux_%s' %bp].values*galaxies['bulge_to_total_ratio_i'].values
    for component in ['disk', 'bulge']:
        galaxies['ra_%s' %component] = galaxies['ra'].values
        galaxies['dec_%s' %component] = galaxies['dec'].values
        galaxies['size_circular_%s' %component] = (galaxies['size_minor_%s_true' %component].values*galaxies['size_%s_true' %component].values)**0.5
        galaxies['e_%s' %component] = galaxies['ellipticity_%s_true' %component]
    # Only keep columns we'll use
    galaxies_cols_to_keep = ['galaxy_id', 'ra', 'dec'] # 'agn', 'sprinkled', 'star']
    galaxies_cols_to_keep += ['ra_bulge', 'dec_bulge', 'size_circular_bulge', 'e_bulge', 'phi_bulge']
    galaxies_cols_to_keep += ['ra_disk', 'dec_disk', 'size_circular_disk', 'e_disk', 'phi_disk'] 
    galaxies_cols_to_keep += [prop + '_' + bp for prop, bp in product(['flux_bulge', 'flux_disk'], filters)]
    galaxies = galaxies[galaxies_cols_to_keep]
    if save_to_disk is not None:
        galaxies.to_csv(save_to_disk, index=False)
    return galaxies

def _format_truth_catalog(point_neighbors, save_to_disk=None):
    # Point-source neighbors
    point_neighbors[['ra', 'dec']] = units.deg_to_arcsec(point_neighbors[['ra', 'dec']])
    for bp in 'ugrizy':
        point_neighbors['flux_%s' %bp] = units.mag_to_flux(point_neighbors[bp].values, to_unit='nMgy')
    if save_to_disk is not None:
        point_neighbors.to_csv(save_to_disk, index=False)
    return point_neighbors

def get_neighbors(candidate_df, reference_ra, reference_dec, radius, ra_colname='ra', dec_colname='dec'):
    from scipy import spatial
    positions = np.c_[candidate_df[[ra_colname, dec_colname]].values]
    tree = spatial.cKDTree(positions)
    target_objects_idx = tree.query_ball_point(x=[reference_ra, reference_dec], r=radius, p=2)
    target_objects = candidate_df.iloc[target_objects_idx].copy()
    return target_objects, target_objects_idx

def separate_bulge_disk(extragal_df):
    # Rename for convenience
    df = extragal_df
    # Separate df into bulge-related and disk-related
    bulge_df = df.filter(like='bulge', axis=1).copy()
    disk_df = df.filter(like='disk', axis=1).copy()
    # Make column schema the same across bulge and disk DataFrames (not sure if necessary)
    bulge_df.columns = [col.strip().replace('_bulge', '') for col in bulge_df.columns]
    disk_df.columns = [col.strip().replace('_disk', '') for col in disk_df.columns]
    return bulge_df, disk_df, df

def point_to_mog(point_df):
    point_df['gauss_sigma'] = 0.0
    point_df['e'] = 0.0
    point_df['phi'] = 0.0
    # Column sanity check
    output_cols = ['ra', 'dec', 'e', 'phi', 'gauss_sigma',]
    output_cols += ['flux_%s' %bp for bp in 'ugrizy']
    mog_df = point_df[output_cols]
    return mog_df

def sersic_to_mog(sersic_df, bulge_or_disk):
    from scipy.special import gammaincinv
    if bulge_or_disk=='bulge':
        # Mixture of gaussian parameters for de Vaucouleurs profile from Hogg and Lang #2013)
        weights = [0.00139, 0.00941, 0.04441, 0.16162, 0.48121, 1.20357, 2.54182, 4.46441, 6.22821, 6.15393]
        stdevs = [0.00087, 0.00296, 0.00792, 0.01902, 0.04289, 0.09351, 0.20168, 0.44126, 1.01833, 2.74555]
        mog_params = {'weight': weights, 'stdev': stdevs}
        sersic_norm = gammaincinv(8, 0.5) # for deVaucouleurs
        gauss_norm = 40320.0*np.pi*np.exp(sersic_norm)/sersic_norm**8.0
    elif bulge_or_disk=='disk':
        # Mixture of gaussian parameters for exponential profile from Hogg and Lang #2013)
        weights = [0.00077, 0.01017, 0.07313, 0.37188, 1.39727, 3.56054, 4.74340, 1.78732]
        stdevs = [0.02393, 0.06490, 0.13580, 0.25096, 0.42942, 0.69672, 1.08879, 1.67294]
        mog_params = {'weight': weights, 'stdev': stdevs}
        sersic_norm = gammaincinv(2, 0.5) # for exponential
        gauss_norm = 2.0*np.pi*np.exp(sersic_norm)/sersic_norm**2.0
    else:
        raise ValueError("Component is either bulge or disk.")
    
    mog_params_df = pd.DataFrame.from_dict(mog_params)
    # Join bulge_df and mog_params_df
    sersic_df = sersic_df.reset_index()
    sersic_df['key'] = 0
    mog_params_df['key'] = 0
    mog_df = sersic_df.merge(mog_params_df, how='left', on='key')
    mog_df = mog_df.drop('key', 1)
    mog_df['gauss_sigma'] = mog_df['size_circular']*mog_df['stdev']
    for bp in 'ugrizy':
        mog_df['flux_%s' %bp] = mog_df['flux_%s' %bp]*mog_df['weight']/gauss_norm
    # Column sanity check
    output_cols = ['ra', 'dec', 'e', 'phi', 'gauss_sigma',]
    output_cols += ['flux_%s' %bp for bp in 'ugrizy']
    mog_df = mog_df[output_cols]
    return mog_df

def join_with_observation(before_observed_df, observation_df):
    observation_df = observation_df.sort_index(axis=1)
    before_observed_df['key'] = 0
    observation_df['key'] = 0
    joined = before_observed_df.merge(observation_df, how='left', on='key')
    joined = joined.drop('key', 1)
    return joined

def collapse_unobserved_fluxes(multi_filter_df):
    all_filters = list('ugrizy')
    for observed_bp in 'ugrizy':
        all_but_observed = all_filters[:]
        all_but_observed.remove(observed_bp)
        set_zero_cols = ['flux_%s' %bp for bp in all_but_observed]
        multi_filter_df.loc[multi_filter_df['filter']==observed_bp, set_zero_cols] = 0.0
    # Sum across filters
    all_flux_cols = ['flux_%s' %bp for bp in all_filters]
    for bp in 'ugrizy':
        multi_filter_df['flux'] = multi_filter_df[all_flux_cols].sum(axis=1)
    # Delete filter-specific fluxes
    single_filter_df = multi_filter_df.drop(all_flux_cols, axis=1)
    return single_filter_df

def typecast_source(source_df):
    source_df['ccdVisitId'] = source_df['ccdVisitId'].astype(np.int64)
    source_df['objectId'] = source_df['objectId'].astype(np.int64)
    source_df['num_gal_neighbors'] = source_df['num_gal_neighbors'].astype(np.int8)
    return source_df

def realize_all_visits_single_object(target_object_id, is_galaxy, fov, deblending_scale,
                                    galaxies_df, points_df, obs_history, add_flux_noise=False):
    source_cols = ['objectId', 'ccdVisitId', 
               'apFlux', 'Ix', 'Iy', 'Ixx', 'Iyy', 'Ixy', 
               'Ixx_PSF', 'sky', 'apFluxErr', 'expMJD',
               'num_star_neighbors', 'num_agn_neighbors', 'num_sprinkled_neighbors']
    # Initialize DataFrame to populate before joining with obs_history_in_field
    target_source_rows = pd.DataFrame(columns=source_cols)
    
    # Target galaxy
    this_galaxy = galaxies_df.query('galaxy_id == @target_object_id')
    ra_center, dec_center = this_galaxy['ra'].item(), this_galaxy['dec'].item() # pos of central galaxy
    
    #################
    # Sersic to MoG #
    #################
    if is_galaxy:
        # Separate galaxy catalog into bulge and disk
        bulge, disk, all_gal = separate_bulge_disk(this_galaxy)
        # Deconstruct bulge/disk into MoG
        bulge_mog = sersic_to_mog(sersic_df=bulge, bulge_or_disk='bulge')
        disk_mog = sersic_to_mog(sersic_df=disk, bulge_or_disk='disk')
        full_mog = pd.concat([bulge_mog, disk_mog,], axis=0)
        full_mog['objectId'] = target_object_id
    else:
        # Query truth catalog for stars/AGNs within blending radius
        this_star, _ = get_neighbors(points_df, ra_center, dec_center, deblending_scale)
        point_mog = point_to_mog(point_df=this_star)
        full_mog = point_mog.copy()
        full_mog['objectId'] = this_star['object_id']

    # Add some metadata
    full_mog['num_gal_neighbors'] = 0
    full_mog['num_star_neighbors'] = 0
    full_mog['num_agn_neighbors'] = 0
    full_mog['num_sprinkled_neighbors'] = 0

    # Get visits at the system's position
    obs_sys_center, _ = get_neighbors(obs_history, ra_center, dec_center, fov, 'ditheredRA', 'ditheredDec')
    # Join with observations
    mog_observed = join_with_observation(full_mog, obs_sys_center)
    # Remove unobserved fluxes (only keep flux of observed filter)
    mog_observed = collapse_unobserved_fluxes(mog_observed)
    # Calculate moment contribution of each Gaussian of blended system
    mog_observed = moments.calculate_total_flux(mog_observed)
    mog_observed = moments.calculate_1st_moments(mog_observed)
    mog_observed = moments.calculate_2nd_moments(mog_observed)
    # Collapse MoGs into one blended system
    target_source_rows = moments.collapse_mog(mog_observed)
    target_source_rows = moments.apply_environment(target_source_rows, add_flux_noise)

    return target_source_rows