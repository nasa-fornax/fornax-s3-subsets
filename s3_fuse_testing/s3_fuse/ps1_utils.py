"""
utilities for interacting with PS1 catalogs, files, and services.
"""
from pathlib import Path

import astropy
import astropy.wcs
import fitsio
import pyarrow as pa
import pyarrow.compute as pac
from cytoolz import first
from gPhoton.coords.wcs import (
    extract_wcs_keywords, corners_of_a_square, sky_box_to_image_box
)


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
    target, bands, side_length, data_root, stat, watch
):
    band_cutouts = {}
    ps1_wcs, coords = None, None
    for band in bands:
        # note that I exactly copied the ps1 file tree structure for this
        # test s3 deployment
        path = ps1_stack_path(target['proj_cell'], target['sky_cell'], band)
        path = f"{data_root}/ps1{path}"
        hdul = fitsio.FITS(path)
        # all bands of a ps1 stack image should have the same wcs, don't
        # waste time projecting stuff twice
        if ps1_wcs is None:
            print("... planning cuts on PS1 images ...")
            ps1_wcs = astropy.wcs.WCS(
                extract_wcs_keywords(hdul[1].read_header())
            )
            corners = corners_of_a_square(
                target['ra'], target['dec'], side_length
            )
            coords = sky_box_to_image_box(corners, ps1_wcs)
            watch.click()
        print(f"slicing data from {Path(path).name}")
        band_cutouts[band] = hdul[1][
             coords[2]:coords[3] + 1, coords[0]:coords[1] + 1
        ]
        watch.click()
        stat.update()
        print(
            f"{round(first(stat.interval.values()) / 1024 ** 2)} "
            f"MB transferred"
        )
    return band_cutouts, ps1_wcs
