"""
galex-specific utilities.
some are vendored from gPhoton 2.
others require a gPhoton 2 installation in the environment.
"""

import random
import warnings
from pathlib import Path

import astropy
import fast_histogram as fh
import fitsio
import numpy as np
from cytoolz import first
from pyarrow import parquet

from s3_fuse.fits import extract_wcs_keywords, corners_of_a_square, \
    sky_box_to_image_box, make_bounding_wcs
from s3_fuse.utilz import Stopwatch, Netstat


def eclipse_to_paths(
    eclipse: int,
    data_directory="data",
    depth=None,
    compression="gzip"
) -> dict[str, dict[str, str]]:
    """
    generate canonical paths for files associated with a given eclipse,
    optionally including files at a specific depth
    """
    zpad = str(eclipse).zfill(5)
    eclipse_path = f"{data_directory}/e{zpad}/"
    eclipse_base = f"{eclipse_path}e{zpad}"
    bands = "NUV", "FUV"
    band_initials = "n", "f"
    file_dict = {}
    comp_suffix = {
        "gzip": ".fits.gz", "none": ".fits", "rice": "-rice.fits"
    }[compression]
    for band, initial in zip(bands, band_initials):
        prefix = f"{eclipse_base}-{initial}d"
        band_dict = {
            "raw6": f"{prefix}-raw6.fits.gz",
            "photonfile": f"{prefix}.parquet",
            "image": f"{prefix}-full{comp_suffix}",
        }
        if depth is not None:
            band_dict |= {
                "movie": f"{prefix}-{depth}s{comp_suffix}",
                # stem -- multiple aperture sizes possible
                "photomfile": f"{prefix}-{depth}s-photom-",
                "expfile": f"{prefix}-{depth}s-exptime.csv",
            }
        file_dict[band] = band_dict
    return file_dict


def get_galex_version_path(eclipse, band, depth, obj, version, data_root):
    """
    get file path for the GALEX data object with specified eclipse, band,
    depth, object type, and compression version, relative to data_root.
    this is a semistandardized convention loosely based on gPhoton conventions
    (which don't include multiple compression types!)
    """
    paths = eclipse_to_paths(eclipse, data_root, depth)
    return {
        "rice": paths[band][obj].replace(".fits.gz", "_rice.fits"),
        "none": paths[band][obj].replace(".gz", ""),
        "gz": paths[band][obj],
    }[version]


def pick_galex_eclipses(count=5, eclipse_type="mislike"):
    """randomly select a set of GALEX eclipses matching some criteria"""

    from gPhoton.aspect import TABLE_PATHS

    # this is the set of actually-available eclipses. it should be adjusted to
    # contain actually-available eclipses in your bucket of choice, or whatever
    # restricted subset you would like.
    with open("extant_eclipses.txt") as file:
        extant_eclipses = set(map(int, file.readlines()))

    # typically larger files, ~10000 x 10000 pixels per frame
    if eclipse_type == "complex":
        columns, predicates, refs = ["legs"], [">"], [1]
    # typically smaller files, ~3000x3000 pixels per frame
    elif eclipse_type == "mislike":
        columns, predicates, refs = (
            ["legs", "obstype"],
            ["=", "in"],
            [0, ["MIS", "DIS", "GII"]],
        )
    # anything!
    else:
        columns, predicates, refs = [], [], []
    eclipse_slice = parquet_generic_search(
        columns, predicates, refs, table_path=TABLE_PATHS["metadata"]
    )
    sliced_eclipses = set(eclipse_slice["eclipse"].to_pylist())
    eclipses_of_interest = extant_eclipses.intersection(sliced_eclipses)
    actual_count = min(count, len(eclipses_of_interest))
    if actual_count < count:
        warnings.warn("only {len(eclipses_of_interest)} available")
    return random.sample(tuple(eclipses_of_interest), k=actual_count)


def parquet_generic_search(columns, predicates, refs, table_path):
    filters = [
        (column, predicate, ref)
        for column, predicate, ref in zip(columns, predicates, refs)
    ]
    return parquet.read_table(table_path, filters=filters)


def project_slice_to_shared_wcs(
    values, individual_wcs, shared_wcs, ra_min, dec_min
):
    """
    Args:
        values: sliced values from source image
        individual_wcs: WCS object for full-frame source image
        shared_wcs: WCS object for coadd
        ra_min: minimum RA of pixels in values
        dec_min: minimum DEC of pixels in values
    """
    indices = np.indices((values.shape[0], values.shape[1]), dtype=np.int16)
    y_ix, x_ix = indices[0].ravel() + dec_min, indices[1].ravel() + ra_min
    ra_input, dec_input = individual_wcs.pixel_to_world_values(x_ix, y_ix)
    x_shared, y_shared = shared_wcs.wcs_world2pix(ra_input, dec_input, 1)
    return {
        "x": x_shared,
        "y": y_shared,
        "weight": values.ravel(),
    }


def bin_projected_weights(x, y, weights, imsz):
    binned = fh.histogram2d(
        y - 0.5,
        x - 0.5,
        bins=imsz,
        range=([[0, imsz[0]], [0, imsz[1]]]),
        weights=weights,
    )
    return binned


def zero_flag_and_edge(cnt, flag, edge):
    cnt[~np.isfinite(cnt)] = 0
    cnt[np.nonzero(flag)] = 0
    cnt[np.nonzero(edge)] = 0
    return cnt


def coadd_galex_rice_slices(
    image_paths, ra, dec, side_length, stat=None, watch=None
):
    """
    top-level handler function for coadding slices from rice-compressed GALEX
    images.
    TODO: not fully integrated yet.
    """
    if watch is None:
        watch = Stopwatch()
    if stat is None:
        stat = Netstat()
    print(f"... planning cuts on {len(image_paths)} galex image(s) ...")
    hduls = [fitsio.FITS(file) for file in image_paths]
    headers = [hdul[1].read_header() for hdul in hduls]
    systems = [
        astropy.wcs.WCS(extract_wcs_keywords(header))
        for header in headers
    ]
    corners = corners_of_a_square(ra, dec, side_length)
    cutout_coords = [
        sky_box_to_image_box(corners, system)
        for system in systems
    ]
    if len(image_paths) > 0:
        shared_wcs = make_bounding_wcs(
            np.array(
                [
                    [corners[2][0], corners[1][1]],
                    [corners[0][0], corners[0][1]]
                ]
            )
        )
    else:
        shared_wcs = None
    watch.click()
    stat.update()
    print(
        f"{round(first(stat.interval.values()) / 1024 ** 2)} MB transferred"
    )
    binned_images = []
    for header, hdul, coords, system in zip(
        headers, hduls, cutout_coords, systems
    ):
        print(f"slicing data from {Path(hdul._filename).name}")
        cnt, flag, edge = [
            hdul[ix][coords[2]:coords[3] + 1, coords[0]:coords[1] + 1]
            for ix in (1, 2, 3)
        ]
        cnt = zero_flag_and_edge(cnt, flag, edge)
        stat.update()
        print(
            f"{round(first(stat.interval.values()) / 1024 ** 2)} "
            f"MB transferred"
        )
        watch.click()
        if len(image_paths) > 0:
            projection = project_slice_to_shared_wcs(
                cnt, system, shared_wcs, coords[0], coords[2]
            )
            binned_images.append(
                bin_projected_weights(
                    projection['x'],
                    projection['y'],
                    projection['weight'] / header['EXPTIME'],
                    wcs_imsz(shared_wcs)
                )
            )
        else:
            return cnt / header['EXPTIME'], system
    return np.sum(binned_images, axis=0), shared_wcs
