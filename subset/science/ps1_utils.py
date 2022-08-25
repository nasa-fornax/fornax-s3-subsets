"""
utilities for interacting with PS1 catalogs, files, and services.
"""
from io import BytesIO
from types import MappingProxyType
from typing import Collection, Optional

import astropy.io.fits
import astropy.wcs
import numpy as np
import pyarrow as pa
import pyarrow.compute as pac
import requests
from cytoolz import groupby
from cytoolz.curried import get
from more_itertools import chunked

PS1_FILTERS = ("g", "r", "i", "z", "y")
PS1_IMAGE_TYPES = (
    "stack",
    "warp",
    "stack.wt",
    "stack.mask",
    "stack.exp",
    "stack.num",
    "warp.wt",
    "warp.mask",
)
PS1_IMAGE_FORMATS = ("fits", "jpg", "png")
PS1_FILENAME_URL = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
PS1_CUTOUT_URL = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
PS1_DOWNLOAD_ROOT = "http://ps1images.stsci.edu/rings.v3.skycell"

# see documentation at
# https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service
def request_ps1_filenames(
    ra: Collection[float],
    dec: Collection[float],
    filters: Collection[str] = PS1_FILTERS,
    image_types: Collection[str] = ("stack",),
    session: Optional[requests.Session] = None,
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
        files={"file": radec_str},
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
            "red": filename,
        },
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
    for proj_cell in pac.unique(test_table["proj_cell"]):
        allowed_skies = pac.unique(
            test_table.filter(pac.equal(test_table["proj_cell"], proj_cell))[
                "sky_cell"
            ]
        )
        catslice = catalog_table.filter(
            pac.equal(catalog_table["proj_cell"], proj_cell)
        )
        proj_cell_tables.append(
            catslice.filter(pac.is_in(catslice["sky_cell"], allowed_skies))
        )
    return pa.concat_tables(proj_cell_tables)


def ps1_chunker(stacks, targets, bands, image_chunksize=40):
    stack_chunks = chunked(stacks, int(image_chunksize / len(bands)))
    target_groups = {
        "_".join(map(str, k)): v
        for k, v in groupby(get(["proj_cell", "sky_cell"]), targets).items()
    }
    return stack_chunks, target_groups


def ps1_chunk_kwargs(stack, band, data_root, loader, bands):
    # all ps1 stack images associated with a sky cell / proj cell should
    # have the same wcs no matter their bands, so don't waste time
    # repeatedly initializing the WCS object (the projection operations
    # themselves are very cheap because we're only projecting 4 pixels)
    stack_init_params = {
        "path": f"{data_root}{ps1_stack_path(*stack.split('_'), band)}",
        "get_wcs": band == bands[0],
        "loader": loader,
        "hdu_indices": (1,),
        "band": band,
    }
    return stack_init_params


def twice_sinh(array):
    """
    calculate 2 * sinh(array) using np.sinh if AVX512F is available,
    and exponential functions if not (> 10x performance difference)
    """
    # noinspection PyProtectedMember
    from numpy.core._multiarray_umath import __cpu_features__

    if __cpu_features__.get("AVX512F") is True:
        return np.sinh(array) * 2
    return np.exp(array) - np.exp(-array)


# a scaling constant used in the PS1 stack image production pipeline,
# not included in the headers.
SCALING_CONSTANT_A = 2.5 / np.log(10)


def ps1_stack2flux(data, header) -> np.ndarray:
    """
    converts asinh-scaled data units in PS1 stack images to linear flux units
    """
    scaled = data / SCALING_CONSTANT_A
    return header["BOFFSET"] + header["BSOFTEN"] * twice_sinh(scaled)


def ps1_flux2mag(flux_sum, exptime):
    """convert PS1 summed linear flux units to AB Mag (exptime in seconds)"""
    return -2.5 * np.log10(flux_sum) + 25 + 2.5 * np.log10(exptime)


def ps1_stack_mask_path(proj_cell, sky_cell, band):
    """
    construct relative path for a PS1 stack mask image by projection cell,
    sky cell, and band.
    """
    proj_cell = str(proj_cell).zfill(4)
    sky_cell = str(sky_cell).zfill(3)
    return (
        f"/rings.v3.skycell/{proj_cell}/{sky_cell}/"
        f"rings.v3.skycell.{proj_cell}.{sky_cell}.stk.{band}.unconv.mask.fits"
    )


PS1_CUT_CONSTANTS = MappingProxyType(
    {
        'chunker': ps1_chunker,
        'kwarg_assembler': ps1_chunk_kwargs,
        'hdu_indices': (1,),
        'share_wcs': True
    }
)
