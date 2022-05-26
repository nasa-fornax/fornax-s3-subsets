"""
utilities for interacting with PS1 catalogs, files, and services.
"""
from io import BytesIO
from pathlib import Path
from typing import Collection, Optional

import astropy.io.fits
from cytoolz.curried import get
from gPhoton.coadd import skybox_cuts_from_file
from more_itertools import all_equal
import pyarrow as pa
import pyarrow.compute as pac
import requests

PS1_FILTERS = ("g", "r", "i", "z", "y")
PS1_IMAGE_TYPES = (
    "stack",
    "warp",
    "stack.wt",
    "stack.mask",
    "stack.exp",
    "stack.num",
    "warp.wt",
    "warp.mask"
)
PS1_IMAGE_FORMATS = ("fits", "jpg", "png")
PS1_FILENAME_URL = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
PS1_CUTOUT_URL = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"


# see documentation at
# https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service
def request_ps1_filenames(
    ra: Collection[float],
    dec: Collection[float],
    filters: Collection[str] = PS1_FILTERS,
    image_types: Collection[str] = ("stack",),
    session: Optional[requests.Session] = None
):
    """
    using the STSCI ps1 filename service, fetch ps1 filenames for a given set
    of ra and dec positions, image filters, and image types.
    """
    image_type_str = ",".join(image_types)
    filters_str = "".join(filters)
    radec_str = "\n".join(
        map(lambda radec: f"{radec[0]} {radec[1]}", zip(ra, dec))
    )
    if session is None:
        session = requests.Session()
    response = session.post(
        PS1_FILENAME_URL,
        data={"filters": filters_str, "type": image_type_str},
        files={"file": radec_str}
    )
    response.raise_for_status()
    lines = tuple(map(lambda s: s.split(" "), response.text.split("\n")))
    fields = lines[0]
    records = [
        {field: value for field, value in zip(fields, line)}
        for line in lines[1:]
    ]
    return records


def request_ps1_cutout(filename, ra, dec, side_length, image_format):
    """
    using the STSCI PS1 image cutout service, fetch cutout of side_length
    (in arcseconds) from filename, centered at ra, dec, in image_format.
    """
    # approximate resolution of PS1 images is 4 pixels / asec
    side_length = round(side_length * 4)
    response = requests.get(
        PS1_CUTOUT_URL,
        params={
            "ra": ra,
            "dec": dec,
            "size": side_length,
            "format": image_format,
            "red": filename
        }
    )
    buffer = BytesIO(response.content)
    buffer.seek(0)
    return astropy.io.fits.open(buffer)


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


