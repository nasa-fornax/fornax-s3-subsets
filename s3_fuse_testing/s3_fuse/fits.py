"""utility functions for occasional FITS manipulation"""
from functools import reduce
from operator import mul
import os
import re
from pathlib import Path
from typing import Sequence, Literal, Union, Callable

from cytoolz import keyfilter
from gPhoton.io.fits_utils import get_header
from gPhoton.reference import crudely_find_library
from gPhoton.types import Pathlike
from killscreen.monitors import make_monitors


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
            ] = int(
                abs(hdu.header['BITPIX'] * reduce(mul, hdu_info['dim'])) / 8
            )
        info[hdu_ix] = hdu_info
    return info


def logged_fits_initializer(
    path: Pathlike,
    loader: Callable,
    hdu_indices: Sequence[int],
    get_wcs: bool = False,
    get_handles: bool = False,
    verbose: int = 0,
    logged: bool = True,
    astropy_handle_attribute: str = "data",
    preload_hdus: bool = False
):
    """
    initialize a FITS object using a passed 'loader' -- probably
    astropy.io.fits.open, a constructor for fitsio.FITS, or a wrapped
    version of one of those. optionally also meticulously record time and
    network transfer involved at all stages of its initialization. At
    present, this function is primarily used for benchmarking.
    """
    # initialize fits HDU list object and read selected HDU's header
    stat, note = make_monitors(fake=not logged)
    hdul = loader(path)
    note(f"init fits object,{path},{stat()}", verbose > 0)
    library = crudely_find_library(loader)
    header = get_header(hdul, hdu_indices[0], library)
    note(f"got header,{path},{stat()}", verbose > 1)
    # TODO: this is a slightly weird hack to revert astropy's automatic
    #  translation of some FITS header values to astropy types. There might
    #  be a cleaner way to do this.
    if library == "astropy":
        output_header = {}
        for k, v in header.items():
            if isinstance(v, (str, float, int)):
                output_header[k] = v
            else:
                output_header[k] = str(v)
        header = output_header
    output = {'header': header, 'stat': stat}
    file_attr = next(filter(lambda attr: "filename" in attr, dir(hdul)))
    output['path'] = getattr(hdul, file_attr)
    if isinstance(output['path'], Callable):
        output['path'] = output['path']()
    if get_handles is True:
        # initialize selected HDU object and get its data 'handles'
        output["handles"] = [hdul[hdu_ix] for hdu_ix in hdu_indices]
        # fitsio exposes slices on HDU data by assigning a __getitem__ method
        # directly to its HDU objects. astropy instead assigns __getitem__
        # methods to attributes of HDU objects, so here we return an attribute
        # rather than the HDU itself as the "handle". by default this is
        # "data", but there are other attributes, notably "section", that also
        # offer data access
        if library == "astropy":
            output["handles"] = [
                getattr(h, astropy_handle_attribute) for h in output["handles"]
            ]
        note(f"got data handles,{path},{stat()}", loud=verbose > 1)
        if preload_hdus is True:
            if library == "astropy":
                [h.data[:].copy() for h in output['handles']]
            else:
                [h.read() for h in output['handles']]
            note(f"preloaded hdus,{path},{stat()}", loud=verbose > 1)
    if get_wcs is True:
        import astropy.wcs
        from gPhoton.coords.wcs import extract_wcs_keywords

        output['wcs'] = astropy.wcs.WCS(extract_wcs_keywords(header))
        note(f"initialized wcs,{path},{stat()}", loud=verbose > 1)
    output['log'] = note(None, eject=True)
    return output
