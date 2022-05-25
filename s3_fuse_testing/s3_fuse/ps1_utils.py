"""
utilities for interacting with PS1 catalogs, files, and services.
"""
from pathlib import Path

import astropy
import astropy.wcs
import fitsio
import pyarrow as pa
import pyarrow.compute as pac
from cytoolz.curried import get
from gPhoton.coords.wcs import (
    extract_wcs_keywords, corners_of_a_square, sky_box_to_image_box
)
from more_itertools import all_equal

from s3_fuse.utilz import print_stats


def ps1_stack_path(proj_cell, sky_cell, band):
    """
    construct relative path for a PS1 stack image by projection cell,
    sky cell, and band.
    """
    proj_cell = str(proj_cell).zfill(4)
    sky_cell = str(sky_cell).zfill(3)
    return (
        f"/rings.v3.skycell/{proj_cell}/{sky_cell}/"
        f"rings.v3.skycell.{proj_cell}.{sky_cell}.stk.{band}.unconv.fits"
    )


def prune_ps1_catalog(catalog_table, test_table):
    """
    support function for testing subsets of the PS1 archive.
    filter a pyarrow table of PS1 objects, retaining only those
    objects that fall into the projection/sky cells defined in test_table.
    """
    # i chose this convoluted-looking technique because pyarrow won't join
    # if there's a list data type, and the equivalent operation is slow in
    # pandas
    proj_cell_tables = []
    for proj_cell in pac.unique(test_table['proj_cell']):
        allowed_skies = pac.unique(
            test_table.filter(
                pac.equal(test_table['proj_cell'], proj_cell)
            )['sky_cell']
        )
        catslice = catalog_table.filter(
            pac.equal(catalog_table['proj_cell'], proj_cell)
        )
        proj_cell_tables.append(
            catslice.filter(pac.is_in(catslice['sky_cell'], allowed_skies))
        )
    return pa.concat_tables(proj_cell_tables)


def get_ps1_cutouts(
    targets, bands, side_length, data_root, stat, watch, verbose=2
):
    # probably a mild waste of time, but might as well check
    assert all_equal(map(get(['proj_cell', 'sky_cell']), targets))
    if verbose > 0:
        print(
            f"...accessing PS1 stack image(s) w/proj cell, sky cell = "
            f"{targets[0]['proj_cell']}, {targets[0]['sky_cell']}..."
        )
    cutouts = {}
    ps1_wcs = None
    watch.click(), stat.update()
    for band in bands:
        # I exactly copied the canonical ps1 directory structure for this
        # test deployment; it lives under the ps1 prefix in the test bucket
        path = ps1_stack_path(
            targets[0]['proj_cell'], targets[0]['sky_cell'], band
        )
        path = f"{data_root}/ps1{path}"
        if verbose > 0:
            print(f"... initializing {Path(path).name} ... ", end="")
        hdul = fitsio.FITS(path)
        # all ps1 stack images associated with a sky cell / proj cell should
        # have the same wcs no matter their bands, so don't waste
        # time initializing the wcs multiple times (the projections themselves
        # should be very cheap because they're so small)
        if ps1_wcs is None:
            ps1_wcs = astropy.wcs.WCS(
                extract_wcs_keywords(hdul[1].read_header())
            )
        if verbose > 0:
            print_stats(watch, stat)
        if verbose == 1:
            print(f"... making {len(targets)} slices ...", end="")
        for target in targets:
            if verbose > 1:
                print(
                    f"... slicing objID={target['obj_id']}; "
                    f"ra={round(target['ra'], 3)}; "
                    f"dec={round(target['dec'], 3)} ... ",
                    end=""
                )
            corners = corners_of_a_square(
                target['ra'], target['dec'], side_length
            )
            coords = sky_box_to_image_box(corners, ps1_wcs)
            cutouts[f"{target['obj_id']}_{band}"] = hdul[1][
                 coords[2]:coords[3] + 1, coords[0]:coords[1] + 1
            ]
            if verbose > 1:
                print_stats(watch, stat)
        if verbose == 1:
            print_stats(watch, stat)
    return cutouts, ps1_wcs
