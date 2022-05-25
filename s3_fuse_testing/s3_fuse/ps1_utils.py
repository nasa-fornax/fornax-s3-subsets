"""
utilities for interacting with PS1 catalogs, files, and services.
"""
from pathlib import Path

from cytoolz.curried import get
from gPhoton.coadd import skybox_cuts_from_file, zero_flag_and_edge
from gPhoton.reference import eclipse_to_paths
from more_itertools import all_equal
import pyarrow as pa
import pyarrow.compute as pac


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


def get_ps1_cutouts(targets, loader, bands, side_length, data_root, verbose=2):
    # probably a mild waste of time, but might as well check
    assert all_equal(map(get(['proj_cell', 'sky_cell']), targets))
    if verbose > 0:
        print(
            f"... accessing PS1 stack image(s) w/proj cell, sky cell = "
            f"{targets[0]['proj_cell']}, {targets[0]['sky_cell']} ..."
        )
    cuts, log = {}, {}
    wcs_object = None
    for band in bands:
        # I exactly copied the canonical ps1 directory structure for this
        # test deployment; it lives under the ps1 prefix in the test bucket
        path = ps1_stack_path(
            targets[0]['proj_cell'], targets[0]['sky_cell'], band
        )
        path = f"{data_root}{path}"
        if verbose > 0:
            print(f"... initializing {Path(path).name} ... ")
        # all ps1 stack images associated with a sky cell / proj cell should
        # have the same wcs no matter their bands, so don't waste time
        # repeatedly initializing the WCS object (the projection operations
        # themselves are very cheap because we're only projecting 4 pixels,
        # so they can be harmlessly repeated)
        band_cuts, wcs_object, _, band_log = skybox_cuts_from_file(
            path, loader, targets, side_length, (1,), wcs_object, verbose
        )
        cuts[band] = band_cuts
        log |= band_log
    return cuts, wcs_object, log


def get_galex_cutouts(
    eclipse, targets, loader, side_length, data_root, verbose=0
):
    # our canonical GALEX path structure, except that these test files happen
    # not to have 'rice' in the name despite being RICE-compressed
    path = eclipse_to_paths(eclipse, data_root, None, "none")["NUV"]["image"]
    if verbose > 0:
        print(f"... initializing {Path(path).name} ... ")
    cuts, wcs_object, header, log = skybox_cuts_from_file(
        path, loader, targets, side_length, (1, 2, 3), verbose=verbose
    )
    for cut in cuts:
        cut['array'] = zero_flag_and_edge(
            cut['arrays'][0], cut['arrays'][1], cut['arrays'][2]
        ) / header['EXPTIME']
        cut.pop('arrays')
    return cuts, wcs_object, log

