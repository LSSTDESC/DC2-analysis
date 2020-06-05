#!/usr/bin/env python
# coding: utf-8

"""
# Inspection of DC2 Run 2.2i DR3 Object Table
### Michael Wood-Vasey (@wmwv)
### Last Verified to Run: 2020-06-04 by MWV

Produce qualitative validation plots for Run 2.2i DR6 Object Table

1. 2D density plots (e.g., `hexbin`, `hist2d`, `datashader`) of
    - ra, dec
    - u-g, g-r
    - r-i, g-r
    - i-z, g-r
    - z-y, g-r
2. 1D density plots (e.g., `hist`, kernel-density-estimation)
    - N({ugrizy})
    - Shape parameters

#### Run 2.2i DR6 as of 2020-06-04 includes
  * 78 tracts
  * 52 million objects
  * 34 million objects with i-band SNR > 5

Loading two columns of entire catalog from parquet file takes a few minutes,
depending on memory load on the JupyterHub node.
It thus is often most useful to develop ones codes looking at only subsamples of the full DC2 datasets, whether that's considering just one tract, or taking a subsample of the full catalog.

#### Quick Data Size estimates

Loading 1 column stored as 64-bit floats on 64 million objects takes 512 MB of memory:

8 bytes/column/row * 1 column * 64 million rows = 2^3 * 2^0 * 2^6 million bytes (MB) = 2^9 MB = 512 MB

We're using the DPDD Parquet file directly:
/global/cfs/cdirs/lsst/production/DC2_ImSim/Run2.2i/dpdd/dc2_object_run2.2i_dr6.parquet

It's 42 GB.  Needs to be used in significantly-restricted column mode
on a dedicated node, or with some Spark/DASK approach.
"""

# Import Needed Modules

import os

import numpy as np
from numpy.lib import scimath as SM

import astropy.units as u
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import seaborn as sns

cmap = "viridis"


def select_good_detections(df):
    # Select good detections:
    #  1. Marked as 'good' in catalog flags.
    #  2. SNR in given band > threshold
    #  3. In defined simulation range
    snr_threshold = 5
    snr_filter = "i"

    # We want to do a SNR cut, but magerr is the thing already calculated
    # So we'll redefine our SNR in terms of magerr
    magerr_cut = (2.5 / np.log(10)) / snr_threshold
    magerr_col = f"magerr_{snr_filter}"

    good_idx = df["good"] & (df[magerr_col] < magerr_cut)

    return df.loc[good_idx]


def define_columns_to_use(filters):
    columns = ["ra", "dec"]
    columns += [f"mag_{f}" for f in filters]
    columns += [f"magerr_{f}" for f in filters]
    columns += [f"mag_{f}_cModel" for f in filters]
    columns += [f"magerr_{f}_cModel" for f in filters]
    columns += [f"Ixx_{f}" for f in filters]
    columns += [f"Ixy_{f}" for f in filters]
    columns += [f"Iyy_{f}" for f in filters]
    columns += [f"psf_fwhm_{f}" for f in filters]
    columns += ["good", "extendedness", "blendedness"]

    return columns


def print_expected_memory_usage(sampling_factor, columns):
    N = 52000000
    MB_per_column = 512 * (N / 64000000)  # MB / column / 64 million rows
    print(f"We are going to load {len(columns)} columns.")
    print(
        f"For {N//sampling_factor} million rows that should take {(len(columns)/sampling_factor)*MB_per_column/1024:0.2f} GB of memory"
    )


def plot_ra_dec_object_density(
    cat, show_dc2_region=True, bins=100, cmin=100, plotname=None,
):
    """
    DC2 Run 2.x WFD and DDF regions
    https://docs.google.com/document/d/18nNVImxGioQ3tcLFMRr67G_jpOzCIOdar9bjqChueQg/view
    https://github.com/LSSTDESC/DC2_visitList/blob/master/DC2visitGen/notebooks/DC2_Run2_regionCoords_WFD.ipynb

    | Location          | RA (degrees) | Dec (degrees) | RA (degrees) | Dec (degrees) |
    |:----------------- |:------------ |:------------- |:------------ |:------------- |
    | Region            | WFD          | WFD           | DDF          | DDF           |
    | Center            | 61.856114    | -35.79        | 53.125       | -28.100       |
    | North-East Corner | 71.462228    | -27.25        | 53.764       | -27.533       |
    | North-West Corner | 52.250000    | -27.25        | 52.486       | -27.533       |
    | South-West Corner | 49.917517    | -44.33        | 52.479       | -28.667       |
    | South-East Corner | 73.794710    | -44.33        | 53.771       | -28.667       |

    (Note that the order of the rows above is different than in the DC2 papers.
    The order of the rows above goes around the perimeter in order.)

    We're just doing this on a rectilinear grid.
    We should do a projection, of course, but that distortion is tolerable in this space.
    """

    dc2_run2x_wfd = [
        [71.462228, -27.25],
        [52.250000, -27.25],
        [49.917517, -44.33],
        [73.794710, -44.33],
    ]
    dc2_run2x_ddf = [
        [53.764, -27.533],
        [52.486, -27.533],
        [52.479, -28.667],
        [53.771, -28.667],
    ]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_aspect(1)

    plt.hist2d(cat["ra"], cat["dec"], bins=bins, cmin=cmin)
    ax.invert_xaxis()  # Flip to East left
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.colorbar(shrink=0.5, label="objects / bin")

    if show_dc2_region:
        # This region isn't quite a polygon.  The sides should be curved.
        wfd_region = Polygon(dc2_run2x_wfd, color="red", fill=False)
        ddf_region = Polygon(dc2_run2x_ddf, color="orange", fill=False)
        ax.add_patch(wfd_region)
        ax.add_patch(ddf_region)

        max_delta_ra = dc2_run2x_wfd[3][0] - dc2_run2x_wfd[2][0]
        delta_dec = dc2_run2x_wfd[1][1] - dc2_run2x_wfd[3][1]
        grow_buffer = 0.05
        ax.set_xlim(
            dc2_run2x_wfd[3][0] + max_delta_ra * grow_buffer,
            dc2_run2x_wfd[2][0] - max_delta_ra * grow_buffer,
        )
        ax.set_ylim(
            dc2_run2x_wfd[3][1] - delta_dec * grow_buffer,
            dc2_run2x_wfd[1][1] + delta_dec * grow_buffer,
        )

    if plotname is not None:
        plt.savefig(plotname)


def ellipticity(I_xx, I_xy, I_yy):
    """Calculate ellipticity from second moments.

    Parameters
    ----------
    I_xx : float or numpy.array
    I_xy : float or numpy.array
    I_yy : float or numpy.array

    Returns
    -------
    e, e1, e2 : (float, float, float) or (numpy.array, numpy.array, numpy.array)
        Complex ellipticity, real component, imaginary component

    Copied from https://github.com/lsst/validate_drp/python/lsst/validate/drp/util.py
    """
    e = (I_xx - I_yy + 2j * I_xy) / (I_xx + I_yy + 2 * SM.sqrt(I_xx * I_yy - I_xy * 2))
    e1 = np.real(e)
    e2 = np.imag(e)
    return e, e1, e2


# We refer to a file over in `tutorials/assets' for the stellar locus
datafile_davenport = "../tutorials/assets/Davenport_2014_MNRAS_440_3430_table1.txt"


def get_stellar_locus_davenport(
    color1="gmr", color2="rmi", datafile=datafile_davenport
):
    data = pd.read_table(datafile, sep="\s+", header=1)
    return data[color1], data[color2]


def plot_stellar_locus(
    color1="gmr", color2="rmi", color="blue", linestyle="--", linewidth=2.5, ax=None
):
    model_gmr, model_rmi = get_stellar_locus_davenport(color1, color2)
    plot_kwargs = {
        "linestyle": linestyle,
        "linewidth": linewidth,
        "color": color,
        "scalex": False,
        "scaley": False,
    }
    if not ax:
        ax = fig.gca()

    ax.plot(model_gmr, model_rmi, **plot_kwargs)


def plot_color_color(
    z,
    color1,
    color2,
    range1=(-1, +2),
    range2=(-1, +2),
    bins=101,
    cmin=10,
    cmap="gist_heat_r",
    vmin=None,
    vmax=None,
    ax=None,
    figsize=(4, 4),
    plotname=None,
):
    """Plot a color-color diagram.  Overlay stellar locus"""
    band1, band2 = color1[0], color1[-1]
    band3, band4 = color2[0], color2[-1]
    H, xedges, yedges = np.histogram2d(
        z[f"mag_{band1}"] - z[f"mag_{band2}"],
        z[f"mag_{band3}"] - z[f"mag_{band4}"],
        range=(range1, range2),
        bins=bins,
    )

    zi = H.T
    xi = (xedges[1:] + xedges[:-1]) / 2
    yi = (yedges[1:] + yedges[:-1]) / 2

    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

    # Only take elements with a minimum number of entries
    zi = np.where(zi >= cmin, zi, np.nan)

    im = ax.pcolormesh(xi, yi, zi, cmap=cmap, vmin=vmin, vmax=vmax)
    cf = ax.contour(xi, yi, zi)
    ax.set_xlabel(f"{band1}-{band2}")
    ax.set_ylabel(f"{band3}-{band4}")

    try:
        plot_stellar_locus(color1, color2, ax=ax)
    except KeyError as e:
        print(f"Couldn't plot Stellar Locus model for {color1}, {color2}")

    if plotname is not None:
        plt.savefig(plotname)

    return im


def plot_four_color_color(cat, vmin=0, vmax=50000, plotname=None):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    colors = ["umg", "rmi", "imz", "zmy"]
    ref_color = "gmr"
    for ax, color in zip(axes.flat, colors):
        try:
            im = plot_color_color(cat, ref_color, color, ax=ax, vmin=vmin, vmax=vmax)
            ax.set_ylim(-1, +2)
        except KeyError:
            continue

    fig.colorbar(im)
    plt.tight_layout()

    if plotname is not None:
        plt.savefig(plotname)


def plot_density_mag(
    good, stars, galaxies, filt, log=True, range=(16, 28), ax=None, plotname=None,
):
    if ax is None:
        ax = fig.gca()
    mag = f"mag_{filt}"
    ax.hist(
        [good[mag], stars[mag], galaxies[mag]],
        label=["all", "star", "galaxy"],
        log=log,
        range=range,
        bins=np.linspace(*range, 100),
        histtype="step",
    )
    ax.set_xlabel(filt)
    ax.set_ylabel("objects / bin")
    ax.set_xlim(range)
    ax.set_ylim(bottom=10)
    ax.legend(loc="upper left")

    if plotname is not None:
        plt.savefig(plotname)


def plot_density_mag_filters(good, stars, galaxies, filters, plotname=None):

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, filt in zip(axes.flat, filters):
        plot_density_mag(good, stars, galaxies, filt, ax=ax, range=(16, 32))

    plt.tight_layout()
    if plotname is not None:
        plt.savefig(plotname)


def calculate_area(cat, threshold=0.25, nside=1024, verbose=False):
    """Calculate the area covered by a catalog with 'ra', 'dec'

    Parameters:
    --
    cat: DataFrame, dict-like with 'ra', 'dec', keys
    threshold:  float
        Fraction of median value required to count a pixel.
    nside:  int
        Healpix NSIDE.  NSIDE=1024 is ~12 sq arcmin/pixel, NSIDE=4096 is 0.74 sq. arcmin/pixel
        Increasing nside will decrease calculated area as holes become better resolved
        and relative Poisson fluctuations in number counts become more significant.
    verbose:  bool
        Print details on nside, number of significant pixels, and area/pixel.

    Returns:
    --
    area:  Astropy Quantity.
    """
    import healpy as hp

    indices = hp.ang2pix(nside, cat["ra"], cat["dec"], lonlat=True)
    idx, counts = np.unique(indices, return_counts=True)

    # Take the 25% of the median value of the non-zero counts/pixel
    threshold_counts = threshold * np.median(counts)

    if verbose:
        print(f"Median {np.median(counts)} objects/pixel")
        print(f"Only count pixels with more than {threshold_counts} objects")

    (significant_pixels,) = np.where(counts > threshold_counts)
    area_pixel = hp.nside2pixarea(nside, degrees=True) * u.deg ** 2

    if verbose:
        print(f"Pixel size ~ {hp.nside2resol(nside, arcmin=True) * u.arcmin:0.2g}")
        print(
            f"nside: {nside}, area/pixel: {area_pixel:0.4g}, num significant pixels: {len(significant_pixels)}"
        )

    area = len(significant_pixels) * area_pixel

    if verbose:
        print(f"Total area: {area:0.7g}")

    return area


def plot_normalize_mag_density(galaxies, plotname=None, figsize=(8, 8)):
    """
    Now we plot the *normalized* i-band magnitude distributions in Run 2.2i.
    They are normalized so we can focus on the shape of the distribution.
    However, the legend indicates the total number density of galaxies selected with our magnitude cut,
    which lets us find issues with the overall number density matching (or not).
    """

    max_mag_i = 26
    plt.figure(figsize=figsize)
    nbins = 50
    mag_range = [20, max_mag_i]
    data_to_plot = [galaxies["mag_i"]]
    labels_to_plot = [
        f"Run 2.2i object catalog: {num_den_dc2.value:.1f} {num_den_dc2.unit:fits}",
    ]
    plt.hist(
        data_to_plot,
        nbins,
        range=mag_range,
        histtype="step",
        label=labels_to_plot,
        linewidth=2.0,
        density=True,
    )

    plt.legend(loc="upper left")
    plt.xlabel("i-band magnitude")
    plt.ylabel("normalized distribution")
    plt.yscale("log")
    if plotname is not None:
        plt.savefig(plotname)


def plot_mag_magerr(
    df, band, ax, range=(16, 28), magerr_limit=0.25, cmin=100, plotname=None
):
    """
    Magnitude Error vs. Magnitude

    The magnitude uncertainties come directly from the poisson estimates of the flux measurements.  By construction they will follow smooth curves.  We here confirm that they do.
    """

    # Restrict to reasonable range
    mag_col, magerr_col = f"mag_{band}", f"magerr_{band}"
    good = df[df[magerr_col] < magerr_limit]

    ax.hexbin(good[mag_col], good[magerr_col], cmin=cmin)
    ax.set_xlabel(band)
    ax.set_ylabel(f"{band} err")
    ax.set_ylim(0, magerr_limit)

    if plotname is not None:
        plt.savefig(plotname)


def plot_mag_magerr_filters(df, filters=("u", "g", "r", "i", "z", "y"), plotname=None):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, filt in zip(axes.flat, filters):
        plot_mag_magerr(df, filt, ax=ax)
    plt.tight_layout()

    if plotname is not None:
        plt.savefig(plotname)


def plot_blendedness(df, plotname=None):
    df_blendedness = df.loc[np.isfinite(df["blendedness"])]
    plt.hexbin(
        df_blendedness["mag_i"], df_blendedness["blendedness"], bins="log", vmin=10
    )
    plt.xlabel("i")
    plt.ylabel("blendedness")
    plt.colorbar(label="objects / bin")

    if plotname is not None:
        plt.savefig(plotname)


def plot_extendedness(df, plotname=None):
    # ### Extendedness
    #
    # Extendedness is essentially star/galaxy separation based purely on morphology in the main detected reference band (which is `i` for most Objects).
    #
    # Extendedness a binary property in the catalog, so it's either 0 or 1.

    # In[40]:

    plt.hexbin(
        df["mag_i"],
        df["extendedness"],
        extent=(14, 26, -0.1, +1.1),
        bins="log",
        vmin=10,
    )
    plt.xlabel("i")
    plt.ylabel("extendedness")
    plt.ylim(-0.1, 1.1)
    plt.text(19, 0.1, "STARS", fontdict={"fontsize": 24}, color="orange")
    plt.text(19, 0.8, "GALAXIES", fontdict={"fontsize": 24}, color="orange")
    plt.colorbar(label="objects / bin")

    if plotname is not None:
        plt.savefig(plotname)


def plot_psf_cmodel(good, stars, galaxies, plotname=None):
    plt.axvline(
        0.0164, 0.4, 1, color="red", linestyle="--", label=r"0.0164 $\Delta$mag cut"
    )  # psf-cModel mag cut from Bosch et al. 2018.

    plt.hist(
        [
            good["mag_i"] - good["mag_i_cModel"],
            stars["mag_i"] - stars["mag_i_cModel"],
            galaxies["mag_i"] - galaxies["mag_i_cModel"],
        ],
        label=["All", "Stars", "Galaxies"],
        bins=np.linspace(-0.1, 0.1, 201),
        histtype="step",
    )

    plt.legend()
    plt.xlabel("mag_i[_psf] - mag_i_CModel")
    plt.ylabel("objects / bin")

    plt.text(-0.080, 100, "STARS", fontdict={"fontsize": 24}, color="orange")
    plt.text(+0.025, 100, "GALAXIES", fontdict={"fontsize": 24}, color="orange")

    if plotname is not None:
        plt.savefig(plotname)


def plot_psf_cmodel_mag_hist2d(good, extent=(14, 26, -0.75, +2.4), plotname=None):
    plt.hexbin(
        good["mag_i"], good["mag_i"] - good["mag_i_cModel"], extent=extent, bins="log",
    )
    plt.xlabel("i")
    plt.ylabel("mag_i[_psf] - mag_i_CModel")
    plt.text(14.5, -0.5, "STARS", fontdict={"fontsize": 24}, color="orange")
    plt.text(18, 2, "GALAXIES", fontdict={"fontsize": 24}, color="orange")
    plt.colorbar(label="objects / bin")

    plt.axhline(0.0164, 0.92, 1.0, color="red", linestyle="--")
    plt.axhline(
        0.0164, 0, 0.1, color="red", linestyle="--", label=r"0.0164 $\Delta$mag cut"
    )
    # psf-cModel mag cut from Bosch et al. 2018.

    if plotname is not None:
        plt.savefig(plotname)


def plot_psf_cmodel_gmr_hist2d(good, extent=(14, 26, -0.75, +2.4), plotname=None):
    plt.hexbin(
        good["mag_g"] - good["mag_r"],
        good["mag_i"] - good["mag_i_cModel"],
        extent=extent,
        bins="log",
    )
    plt.xlabel("g-r")
    plt.ylabel("mag_i[_psf] - mag_i_CModel")
    plt.text(14.5, -0.5, "STARS", fontdict={"fontsize": 24}, color="orange")
    plt.text(18, 2, "GALAXIES", fontdict={"fontsize": 24}, color="orange")
    plt.colorbar(label="objects / bin")

    plt.axhline(0.0164, 0.92, 1.0, color="red", linestyle="--")
    plt.axhline(
        0.0164, 0, 0.1, color="red", linestyle="--", label=r"0.0164 $\Delta$mag cut"
    )
    # psf-cModel mag cut from Bosch et al. 2018.

    if plotname is not None:
        plt.savefig(plotname)


def plot_ellipticity(good, stars, galaxies, filt, ax=None, legend=True, plotname=None):
    if not ax:
        ax = fig.gca()

    names = ["all", "star", "galaxy"]
    colors = ["blue", "orange", "green"]
    hist_kwargs = {
        "color": colors,
        "log": True,
        "bins": np.logspace(-1, 1.5, 100),
        "range": (0, 5),
        "histtype": "step",
    }
    for prefix, ls in (("e", "-"), ("e1", "--"), ("e2", ":")):
        field = f"{prefix}_{filt}"
        labels = [f"{prefix} {name}" for name in names]
        ax.hist(
            [good[field], stars[field], galaxies[field]],
            label=labels,
            linestyle=ls,
            **hist_kwargs,
        )

    ax.set_xlim(0, 20)
    ax.set_ylim(10, ax.get_ylim()[1])

    ax.set_xlabel(f"{filt}-band ellipticity")
    ax.set_ylabel("objects / bin")
    if legend:
        ax.legend()


def plot_shape(filt, ax=None, legend=True, plotname=None):
    if not ax:
        ax = fig.gca()

    names = ["all", "star", "galaxy"]
    colors = ["blue", "orange", "green"]
    hist_kwargs = {
        "color": colors,
        "log": True,
        "bins": np.logspace(-1, 1.5, 100),
        "range": (0, 50),
        "histtype": "step",
    }
    for prefix, ls in (("Ixx", "-"), ("Iyy", "--"), ("Ixy", ":")):
        field = f"{prefix}_{filt}"
        labels = [f"{prefix} {name}" for name in names]
        ax.hist(
            [good[field], stars[field], galaxies[field]],
            label=labels,
            linestyle=ls,
            **hist_kwargs,
        )

    ax.set_ylim(100, ax.get_ylim()[1])

    ax.set_xlabel(f"{filt}-band Moments: Ixx, Iyy, Ixy [pixels^2]")
    ax.set_ylabel("objects / bin")
    if legend:
        ax.legend()

    plt.tight_laytout()
    if plotname is not None:
        plt.savefig(plotname)


def plot_psf_fwhm(
    good,
    filters,
    colors=("purple", "blue", "green", "orange", "red", "brown"),
    plotname=None,
):
    for filt, color in zip(filters, colors):
        psf_fwhm = np.array(good[f"psf_fwhm_{filt}"])
        (w,) = np.where(np.isfinite(psf_fwhm) & (psf_fwhm < 3))
        sns.distplot(psf_fwhm[w], label=filt, color=color)
    plt.xlabel("PSF FWHM [arcsec]")
    plt.ylabel("normalized object density")
    plt.legend()

    plt.tight_laytout()
    plotname = f"{data_release}_ellipticity.{suffix}"
    plt.savefig(plotname)


def run():
    suffix = "pdf"
    data_release = "DC2_Run2.2i_DR6"

    # Define Catalog and Subsampling

    catalog_dirname = "/global/cfs/cdirs/lsst/production/DC2_ImSim/Run2.2i/dpdd/"
    catalog_basename = "dc2_object_run2.2i_dr6.parquet"
    catalog_file = os.path.join(catalog_dirname, catalog_basename)

    # Load Data

    filters = ("u", "g", "r", "i", "z", "y")

    columns = define_columns_to_use(filters)
    sampling_factor = 1
    print_expected_memory_usage(sampling_factor, columns)


    print(f"Reading {catalog_file}")
    df = pd.read_parquet(catalog_file, columns=columns)
    good = select_good_detections(df)

    print(f"Loaded {len(df)} objects.")
    print(f"Loaded {len(good)} good objects.")

    for filt in filters:
        df[f"e_{filt}"], df[f"e1_{filt}"], df[f"e2_{filt}"] = ellipticity(
            df[f"Ixx_{filt}"], df[f"Ixy_{filt}"], df[f"Iyy_{filt}"]
        )

    plot_ra_dec(df, plotname=f"{data_release}_ra_dec.{suffix}")

    stars = df.loc[df["extendedness"] == 0]
    galaxies = df.loc[df["extendedness"] > 0]

    print(
        f"Total: {len(df)}, Good: {len(good)}, Stars: {len(stars)}, Galaxies: {len(galaxies)}"
    )
    print(f"For {catalog_file} with {sampling_factor}x subsample")

    # Color-Color Diagrams and the Stellar Locus
    im = plot_color_color(good, "gmr", "rmi")
    plt.colorbar(im)

    plotname = f"{data_release}_good_color_color.{suffix}"
    plot_four_color_color(good, vmax=50000, plotname=plotname)

    plotname = f"{data_release}_star_color_color.{suffix}"
    plot_four_color_color(stars, vmax=10000, plotname=plotname)

    plotname = f"{data_release}_galaxy_color_color.{suffix}"
    plot_four_color_color(galaxies, vmax=40000, plotname=plotname)

    area_dc2 = calculate_area(galaxies)
    print(f"DC2 Run 2.2i area: {area_dc2:0.2f}")

    num_den_dc2 = sampling_factor * len(galaxies) / area_dc2

    # Change default expression to 1/arcmin**2
    num_den_dc2 = num_den_dc2.to(1 / u.arcmin ** 2)

    plotname = f"{data_release}_galaxy_counts.pdf"
    plot_normalize_mag_density(galaxies, plotname=plotname)

    plot_mag_magerr_filters(galaxies, filters)
    plot_mag_magerr_filters(stars, filters)

    # ## Blendedness
    #
    # Blendedness is a measure of how much the identified flux from an object is affected by overlapping from other objects.
    #
    # See Bosch et al., 2018, Section 4.9.11.

    (w,) = np.where(np.isfinite(good["blendedness"]))

    print(
        f"{100 * len(w)/len(good):0.1f}% of objects have finite blendedness measurements."
    )
    plotname = f"{data_release}_psf_cmodel.{suffix}"
    plot_psf_cmodel(good, stars, galaxies, plotname=plotname)
    plotname = f"{data_release}_psf_cmodel_i.{suffix}"
    plot_psf_cmodel_mag_hist2d(good, plotname=plotname)
    plotname = f"{data_release}_psf_cmodel_i_zoom.{suffix}"
    plot_psf_cmodel_mag_hist2d(good, plotname=plotname, extent=(22, 25.5, -0.1, +0.5))

    plotname = f"{data_release}_psf_cmodel_g_r.{suffix}"
    plot_psf_cmodel_gmr_hist2d(good, plotname=plotname, extent=(-2, +3, -0.1, +0.5))

    ############
    plt.hist(
        [galaxies["mag_g"] - galaxies["mag_r"], stars["mag_g"] - stars["mag_r"]],
        label=["galaxies", "stars"],
        histtype="step",
        bins=np.linspace(-5, +5, 51),
    )
    plt.xlabel("g-r")
    plt.ylabel("objects / bin")

    # In[46]:

    plt.hexbin(
        stars["mag_g"] - stars["mag_r"],
        stars["mag_i"] - stars["mag_i_cModel"],
        extent=(-2, +3, -0.5, +5),
        bins="log",
    )
    plt.xlabel("g-r")
    plt.ylabel("mag_i[_psf] - mag_i_CModel")
    # plt.text(14.5, 0.3, "STARS", fontdict={'fontsize': 24}, color='orange')
    # plt.text(18, 2, "GALAXIES", fontdict={'fontsize': 24}, color='orange')
    plt.colorbar(label="objects / bin")

    # ## Shape Parameters
    #
    # Ixx, Iyy, Ixy

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    legend = True
    for ax, filt in zip(axes.flat, filters):
        plot_shape(filt, ax=ax, legend=legend)
        legend = False

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    legend = True
    for ax, filt in zip(axes.flat, filters):
        plot_ellipticity(good, stars, galaxies, filt, ax=ax, legend=legend)
        legend = False
    plt.tight_laytout()
    plotname = f"{data_release}_ellipticity.{suffix}"
    plt.savefig(plotname)

    plot_psf_fwhm()
    plt.tight_laytout()
    plotname = f"{data_release}_fwhm.{suffix}"
    plt.savefig(plotname)


if __name__ == "__main__":
    run()
