from __future__ import absolute_import, division, print_function
import os, sys
import numpy as np
import pandas as pd
from itertools import product
import units
import matplotlib.pyplot as plt
import re
import GCRCatalogs
import scipy.spatial as scipy_spatial
from lsst.utils import getPackageDir
from lsst.sims.utils import defaultSpecMap
from lsst.sims.photUtils import BandpassDict, Bandpass, Sed, CosmologyObject
import time
import multiprocessing

def plot_bic(param_range,bics,lowest_comp):
    from astroML.plotting import setup_text_plots
    
    plt.clf()
    setup_text_plots(fontsize=16, usetex=False)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(param_range,bics,color='blue',lw=2, marker='o')
    plt.text(lowest_comp, bics.min() * 0.97 + .03 * bics.max(), '*',
             fontsize=14, ha='center')

    plt.xticks(param_range)
    plt.ylim(bics.min() - 0.05 * (bics.max() - bics.min()),
             bics.max() + 0.05 * (bics.max() - bics.min()))
    plt.xlim(param_range.min() - 1, param_range.max() + 1)

    plt.xticks(param_range,fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel('Number of components',fontsize=18)
    plt.ylabel('BIC score',fontsize=18)

    plt.show()
    
def get_sl2s_data(filename):
    """
    Note
    ----
    Edited from original version in `MatchingLensGalaxies_utilities.py`
    Returns
    -------
    Formatted SL2S data (Sonnenfeld et al. 2013) where the schema for X is:
    column index : value
    0 : z
    1 : v_disp
    2 : r_eff --> removed
    3 : log_m --> 2: log_m
    
    # TODO avoid number indexing
    """
    z = np.array([])
    z_err = np.array([])
    v_disp = np.array([])
    v_disp_err = np.array([])
    #r_eff = np.array([])
    #r_eff_err = np.array([])
    log_m = np.array([])
    log_m_err = np.array([])
    
    infile = open(filename, 'r')
    inlines = infile.readlines()
    
    for line1 in inlines:
        if line1[0] == '#': continue
        line = line1.split(',')
        #Params
        z = np.append(z, float(line[1]))
        v_disp = np.append(v_disp, float(line[2]))
        #r_eff = np.append(r_eff, float(line[3]))
        log_m = np.append(log_m, float(line[4]))
        #Errors
        z_err = np.append(z_err, float(line[5]))
        v_disp_err = np.append(v_disp_err, float(line[6]))
        #r_eff_err = np.append(r_eff_err, float(line[7]))
        log_m_err = np.append(log_m_err, float(line[8]))
    
    #Build final arrays
    X = np.vstack([z, v_disp, 
                   #r_eff,
                   log_m]).T # shape (N, 4) --> (N, 3) without radius
    Xerr = np.zeros(X.shape + X.shape[-1:]) # shape (N, 4, 4) --> (N, 3, 3) without radius
    diag = np.arange(X.shape[-1]) # range(4) --> range(3)
    
    Xerr[:, diag, diag] = np.vstack([z_err**2, v_disp_err**2,
                                     #r_eff_err**2,
                                     log_m_err**2]).T
    
    return X, Xerr

def get_log_m(cond_indices, m_index, X, empiricist, model_file, Xerr=None):
    
    """
    Uses a subset of parameters in the given data to condition the
    model and return a sample value for log(M/M_sun).

    Parameters
    ----------
    cond_indices: array_like
        Array of indices indicating which parameters to use to
        condition the model. 
    m_index: int
        Index of log(M/M_sun) in the list of parameters that were used
        to fit the model.
    X: array_like, shape = (n < n_features,)
        Input data.
    Xerr: array_like, shape = (X.shape,) (optional)
        Error on input data. If none, no error used to condition.

    Returns
    -------
    log_m: float
        Sample value of log(M/M_sun) taken from the conditioned model.

    Notes
    -----
    The fit_params array specifies a list of indices to use to
    condition the model. The model will be conditioned and then
    a mass will be drawn from the conditioned model.

    This is so that the mass can be used to find cosmoDC2 galaxies
    to act as hosts for OM10 systems.

    This does not make assumptions about what parameters are being
    used in the model, but does assume that the model has been
    fit already.
    """

    if m_index in cond_indices:
        raise ValueError("Cannot condition model on log(M/M_sun).")

    cond_data = np.array([])
    if Xerr is not None: cond_err = np.array([])
    m_cond_idx = m_index
    n_features = empiricist.XDGMM.mu.shape[1]
    j = 0

    for i in range(n_features):
        if i in cond_indices:
            cond_data = np.append(cond_data,X[j])
            if Xerr is not None: cond_err = np.append(cond_err, Xerr[j])
            j += 1
            if i < m_index: m_cond_idx -= 1
        else:
            cond_data = np.append(cond_data,np.nan)
            if Xerr is not None: cond_err = np.append(cond_err, 0.0)

    if Xerr is not None:
        cond_XDGMM = empiricist.XDGMM.condition(cond_data, cond_err)
    else: cond_XDGMM = empiricist.XDGMM.condition(cond_data)

    sample = cond_XDGMM.sample()
    log_m = sample[0][m_cond_idx]
    return log_m

_galaxy_sed_dir = os.path.join(getPackageDir('sims_sed_library'), 'galaxySED')

disk_re = re.compile(r'sed_(\d+)_(\d+)_disk$')
bulge_re = re.compile(r'sed_(\d+)_(\d+)_bulge$')

def sed_filter_names_from_catalog(catalog):
    """
    Takes an already-loaded GCR catalog and returns the names, wavelengths,
    and widths of the SED-defining bandpasses
    Parameters
    ----------
    catalog -- is a catalog loaded with GCR.load_catalog()
    Returns
    -------
    A dict keyed to 'bulge' and 'disk'.  The values in this dict will
    be dicts keyed to 'filter_name', 'wav_min', 'wav_width'.  The
    corresponding values are:
    filter_name -- list of the names of the columns defining the SED
    wav_min -- list of the minimum wavelengths of SED-defining bandpasses (in nm)
    wav_width -- list of the widths of the SED-defining bandpasses (in nm)
    All outputs will be returned in order of increasing wav_min
    """

    all_quantities = catalog.list_all_quantities()

    bulge_names = []
    bulge_wav_min = []
    bulge_wav_width = []

    disk_names = []
    disk_wav_min = []
    disk_wav_width = []

    for qty_name in all_quantities:
        disk_match = disk_re.match(qty_name)
        if disk_match is not None:
            disk_names.append(qty_name)
            disk_wav_min.append(0.1*float(disk_match[1]))  # 0.1 converts to nm
            disk_wav_width.append(0.1*float(disk_match[2]))

        bulge_match = bulge_re.match(qty_name)
        if bulge_match is not None:
            bulge_names.append(qty_name)
            bulge_wav_min.append(0.1*float(bulge_match[1]))
            bulge_wav_width.append(0.1*float(bulge_match[2]))

    disk_wav_min = np.array(disk_wav_min)
    disk_wav_width = np.array(disk_wav_width)
    disk_names = np.array(disk_names)
    sorted_dex = np.argsort(disk_wav_min)
    disk_wav_width = disk_wav_width[sorted_dex]
    disk_names = disk_names[sorted_dex]
    disk_wav_min = disk_wav_min[sorted_dex]

    bulge_wav_min = np.array(bulge_wav_min)
    bulge_wav_width = np.array(bulge_wav_width)
    bulge_names = np.array(bulge_names)
    sorted_dex = np.argsort(bulge_wav_min)
    bulge_wav_width = bulge_wav_width[sorted_dex]
    bulge_names = bulge_names[sorted_dex]
    bulge_wav_min = bulge_wav_min[sorted_dex]

    return {'disk':{'filter_name': disk_names,
                    'wav_min': disk_wav_min,
                    'wav_width': disk_wav_width},
            'bulge':{'filter_name': bulge_names,
                     'wav_min': bulge_wav_min,
                     'wav_width': bulge_wav_width}}

def _create_library_one_sed(_galaxy_sed_dir, sed_file_name_list,
                            av_grid, rv_grid, bandpass_dict,
                            out_dict):

    n_obj = len(av_grid)*len(rv_grid)

    imsim_bp = Bandpass()
    imsim_bp.imsimBandpass()

    t_start = time.time()
    for i_sed, sed_file_name in enumerate(sed_file_name_list):
        if i_sed>0 and i_sed%10 ==0:
            duration = (time.time()-t_start)/3600.0
            pred = len(sed_file_name_list)*duration/i_sed
            print('%d of %d; dur %.2e pred %.2e' %
            (i_sed, len(sed_file_name_list), duration, pred))

        base_spec = Sed()
        base_spec.readSED_flambda(os.path.join(_galaxy_sed_dir, sed_file_name))
        ax, bx = base_spec.setupCCMab()

        mag_norm = base_spec.calcMag(imsim_bp)

        sed_names = np.array([defaultSpecMap[sed_file_name]]*n_obj)
        rv_out_list = np.zeros(n_obj, dtype=float)
        av_out_list = np.zeros(n_obj, dtype=float)
        sed_mag_norm = mag_norm*np.ones(n_obj, dtype=float)
        sed_mag_list = []

        i_obj = 0
        for av in av_grid:
            for rv in rv_grid:
                spec = Sed(wavelen=base_spec.wavelen, flambda=base_spec.flambda)
                spec.addCCMDust(ax, bx, A_v=av, R_v=rv)
                av_out_list[i_obj] = av
                rv_out_list[i_obj] = rv
                sed_mag_list.append(tuple(bandpass_dict.magListForSed(spec)))
                i_obj += 1

        out_dict[sed_file_name] = (sed_names, sed_mag_norm, sed_mag_list,
                                   av_out_list, rv_out_list)


def _create_sed_library_mags(wav_min, wav_width):
    """
    Calculate the magnitudes of the SEDs in sims_sed_library dir in the
    tophat filters specified by wav_min, wav_width
    Parameters
    ----------
    wav_min is a numpy array of the minimum wavelengths of the tophat
    filters (in nm)
    wav_width is a numpy array of the widths of the tophat filters (in nm)
    Returns
    -------
    sed_names is an array containing the names of the SED files repeated over
    combinations of dust parameters (sorry; that wording is awkward)
    sed_mag_list is MxN float array, with M = number of SED file, dust parameter
    combinations in the library, and N = number of top hat filters in the catalog
    sed_mag_norm is 1d float array, with length = number of SED file, dust parameter
    combinations in the library
    av_out_list is a 1d float array of Av
    rv_out_list is a 1d float array of Rv
    """

    av_grid = np.arange(0.0, 3.0, 0.1)
    rv_grid = np.arange(2.0, 4.1, 0.1)

    wav_max = max((wav0+width
                  for wav0, width in zip(wav_min, wav_width)))
    wav_grid = np.arange(wav_min.min(), wav_max, 0.1)

    bp_name_list = list()
    bp_list = list()
    for wav0, width in zip(wav_min, wav_width):
        sb_grid = ((wav_grid >= wav0) & (wav_grid <= (wav0+width))).astype(float)
        bp_list.append(Bandpass(wavelen=wav_grid, sb=sb_grid))
        bp_name_list.append('%d_%d' % (wav0, width))

    bandpass_dict = BandpassDict(bp_list, bp_name_list)

    sed_names = list()
    sed_mag_list = list()
    sed_mag_norm = list()
    av_out_list = list()
    rv_out_list = list()

    list_of_files = os.listdir(_galaxy_sed_dir)
    n_tot = len(list_of_files)*len(av_grid)*len(rv_grid)
    t_start = time.time()

    sed_names = np.empty(n_tot, dtype=(str, 200))
    sed_mag_list = np.zeros((n_tot, len(bp_list)), dtype=float)
    sed_mag_norm = np.zeros(n_tot, dtype=float)
    av_out_list = np.zeros(n_tot, dtype=float)
    rv_out_list = np.zeros(n_tot, dtype=float)

    print('\n\ncreating library')
    p_list = []
    n_proc = 24
    mgr = multiprocessing.Manager()
    out_dict = mgr.dict()
    i_stored = 0
    d_start = len(list_of_files)//n_proc
    i_start_list = range(0, len(list_of_files), d_start)
    for i_meta, i_start in enumerate(i_start_list):
        i_end = i_start+d_start
        if i_meta == len(i_start_list)-1:
            i_end = len(list_of_files)

        p = multiprocessing.Process(target=_create_library_one_sed,
                                    args=(_galaxy_sed_dir,
                                          list_of_files[i_start:i_end],
                                          av_grid, rv_grid,
                                          bandpass_dict, out_dict))

        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()

    print('done calculating')
    t_start = time.time()
    n_kk = len(list(out_dict.keys()))
    for i_kk, kk in enumerate(out_dict.keys()):
        n_out = len(out_dict[kk][1])
        sed_names[i_stored:i_stored+n_out] = out_dict[kk][0]
        sed_mag_norm[i_stored:i_stored+n_out] = out_dict[kk][1]
        sed_mag_list[i_stored:i_stored+n_out][:] = out_dict[kk][2]
        av_out_list[i_stored:i_stored+n_out] = out_dict[kk][3]
        rv_out_list[i_stored:i_stored+n_out] = out_dict[kk][4]
        i_stored += n_out
        if i_kk>0 and i_kk%10==0:
            d = (time.time()-t_start)/3600.0
            p = n_kk*d/i_kk
            print('%d in %.2e; pred %.2e' % (i_kk, d, p))

    print('made library')
    print('%d' % (len(np.where(av_out_list<1.0e-10)[0])))
    assert len(np.where(av_out_list<1.0e-10)[0]) == len(list_of_files)*len(rv_grid)

    return (sed_names, np.array(sed_mag_list), sed_mag_norm,
            av_out_list, rv_out_list)


def sed_from_galacticus_mags(galacticus_mags, redshift, H0, Om0,
                             wav_min, wav_width):
    """
    Fit SEDs from sims_sed_library to Galacticus galaxies based on the
    magnitudes in tophat filters.
    Parameters
    ----------
    galacticus_mags is a numpy array such that
    galacticus_mags[i][j] is the magnitude of the jth star in the ith bandpass,
    where the bandpasses are ordered in ascending order of minimum wavelength.
    redshift is an array of redshifts for the galaxies being fit
    H0 is the Hubbleparameter in units of km/s/Mpc
    Om0 is the critical density parameter for matter
    wav_min is a numpy array of the minimum wavelengths of the tophat
    filters (in nm)
    wav_grid is a numpy array of the widths of the tophat filters
    (in nm)
    Returns
    -------
    a numpy array of SED names and a numpy array of magNorms.
    """

    if (not hasattr(sed_from_galacticus_mags, '_color_tree') or
        not np.allclose(wav_min, sed_from_galacticus_mags._wav_min,
                        atol=1.0e-10, rtol=0.0) or
        not np.allclose(wav_width, sed_from_galacticus_mags._wav_width,
                        atol=1.0e-10, rtol=0.0)):

        (sed_names,
         sed_mag_list,
         sed_mag_norm,
         av_grid, rv_grid) = _create_sed_library_mags(wav_min, wav_width)


        sed_colors = sed_mag_list[:,1:] - sed_mag_list[:,:-1]
        sed_from_galacticus_mags._sed_names = sed_names
        sed_from_galacticus_mags._mag_norm = sed_mag_norm # N_sed
        sed_from_galacticus_mags._av_grid = av_grid
        sed_from_galacticus_mags._rv_grid = rv_grid
        sed_from_galacticus_mags._sed_mags = sed_mag_list # N_sed by N_mag
        sed_from_galacticus_mags._color_tree = scipy_spatial.cKDTree(sed_colors)
        sed_from_galacticus_mags._wav_min = wav_min
        sed_from_galacticus_mags._wav_width = wav_width

    if (not hasattr(sed_from_galacticus_mags, '_cosmo') or
        np.abs(sed_from_galacticus_mags._cosmo.H()-H0)>1.0e-6 or
        np.abs(sed_from_galacticus_mags._cosmo.OmegaMatter()-Om0)>1.0e-6):

        sed_from_galacticus_mags._cosmo = CosmologyObject(H0=H0, Om0=Om0)

    print("done initializing")

    galacticus_mags_t = np.asarray(galacticus_mags).T # N_star by N_mag
    assert galacticus_mags_t.shape == (len(redshift), sed_from_galacticus_mags._sed_mags.shape[1])

    with np.errstate(invalid='ignore', divide='ignore'):
        galacticus_colors = galacticus_mags_t[:,1:] - galacticus_mags_t[:,:-1] # N_star by (N_mag - 1)

    print("querying")
    t_start = time.time()
    (sed_dist,
     sed_idx) = sed_from_galacticus_mags._color_tree.query(galacticus_colors, k=1)
    print("querying took %e" % ((time.time()-t_start)/3600.0))

    # cKDTree returns an invalid index (==len(tree_data)) in cases
    # where the distance is not finite
    sed_idx = np.where(sed_idx<len(sed_from_galacticus_mags._sed_names),
                       sed_idx, 0)

    distance_modulus = sed_from_galacticus_mags._cosmo.distanceModulus(redshift=redshift)
    output_names = sed_from_galacticus_mags._sed_names[sed_idx]
    d_mag = (galacticus_mags_t - sed_from_galacticus_mags._sed_mags[sed_idx]).mean(axis=1)
    output_mag_norm = sed_from_galacticus_mags._mag_norm[sed_idx] + d_mag + distance_modulus

    return output_names, output_mag_norm, sed_from_galacticus_mags._av_grid[sed_idx], sed_from_galacticus_mags._rv_grid[sed_idx]