"""utility functions for occasional FITS manipulation"""
from functools import reduce
from operator import mul
import os
import re
from pathlib import Path
from typing import Sequence, Literal, Union

from cytoolz import keyfilter


def make_tiled_galex_object(
    eclipse: int,
    band: Literal["NUV", "FUV"],
    depth: int,
    tile_size: Sequence[int] = (1, 100, 100),
    quantize_level: int = 16,
    obj: str = Literal["image", "movie"],
    data_path: str = "test_data",
    return_obj: bool = True,
):
    """convert a gzipped galex data object to a RICE-compressed version"""
    import astropy.io.fits

    from gPhoton.coadd import pyfits_open_igzip
    from gPhoton.reference import eclipse_to_paths

    paths = eclipse_to_paths(eclipse, data_path, depth)
    hdul = pyfits_open_igzip(paths[band][obj])
    comp_hdus = [
        astropy.io.fits.CompImageHDU(
            hdul[ix].data,
            hdul[ix].header,
            tile_size=tile_size,
            compression_type="RICE_1",
            quantize_level=quantize_level,
        )
        for ix in range(3)
    ]
    # TODO: confirm astropy.io.fits -- and maybe the FITS standard itself? --
    #  do not permit compressed image HDUs as primary HDUs
    primary_hdu = astropy.io.fits.PrimaryHDU(None, hdul[0].header)
    hdul_comp = astropy.io.fits.HDUList([primary_hdu, *comp_hdus])
    path_comp = paths[band][obj].replace(".fits.gz", "_rice.fits")
    hdul_comp.writeto(path_comp, overwrite=True)
    print(
        f"compression ratio: "
        f"{os.path.getsize(path_comp) / os.path.getsize(paths[band][obj])}"
    )
    if return_obj:
        return path_comp, hdul
    return path_comp


def imsz_from_header(header):
    """
    get image size from either compressed or uncompressed FITS image headers.
    returns in 'reversed' order for numpy array indexing.
    """
    key_type = 'ZNAXIS' if "ZNAXIS" in header.keys() else "NAXIS"
    axis_entries = keyfilter(
        lambda k: False if k is None else re.match(rf"{key_type}\d", k),
        dict(header)
    )
    return tuple(reversed(axis_entries.values()))


def fitsstat(path: Union[str, Path]) -> dict:
    import astropy.io.fits

    info = {'filesize': os.stat(path).st_size}
    hdul = astropy.io.fits.open(path)
    hdulinfo = hdul.info(False)
    info['n_hdu'] = len(hdul)
    for hdu_ix, hdu in enumerate(hdul):
        hduinfo = hdu.fileinfo()
        hdu_info = {
            'size': hduinfo['datLoc'] - hduinfo['hdrLoc'] + hduinfo['datSpan'],
            'name': hdulinfo[hdu_ix][1],
            'hdutype': hdulinfo[hdu_ix][3],
            'dim': imsz_from_header(hdu.header),
        }
        if len(hdu_info['dim']) == 0:
            hdu_info['itemsize'] = None
            hdu_info['datasize'] = 0
        else:
            hdu_info['itemsize'] = abs(hdu.header['BITPIX'])
            hdu_info[
                'datasize'
            ] = abs(hdu.header['BITPIX'] * reduce(mul, hdu_info['dim']))
        info[hdu_ix] = hdu_info
    return info
