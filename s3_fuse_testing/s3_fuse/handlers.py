"""
top-level handling functions for s3-slicing testing and benchmarks
"""
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np

from s3_fuse.fits import get_header, imsz_from_header
from s3_fuse.random_generators import rectangular_slices
from s3_fuse.mount_s3 import mount_bucket
from s3_fuse.utilz import record_and_yell, crudely_find_library, Stopwatch


def perform_cut(
    array_handle: Union["fitsio.hdu.base.HDUBase", np.ndarray],
    cut_ix: int,
    imsz: Sequence[int],
    boxes: np.ndarray,
    bands: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    cut a specified 'box' from an array (like the data attribute of an
    astropy.io.fits HDU) or array accessor (like a fitsio HDU)
    """
    x0, x1, y0, y1 = boxes[:, cut_ix, :].ravel()
    if (len(imsz) == 3) and (bands is not None):
        # written in this weird way for fitsio, which will not automatically
        # broadcast slices
        return array_handle[
            y0:y1, x0:x1, bands[cut_ix]:bands[cut_ix] + 1
        ][:, :, 0]
    elif len(imsz) == 3:
        return array_handle[:, y0:y1, x0:x1]
    return array_handle[y0:y1, x0:x1]


def get_cuts_from_file(
    path: Path,
    loader: Callable,
    hdu_ix: int,
    cut_settings: Mapping,
    seed: Optional[int] = None,
    shallow: bool = False,
):
    """take random slices from a fits file; examine this process closely"""
    # initialize fits HDU list object and read selected HDU's header
    watch = Stopwatch(silent=True)
    watch.start()
    log = {}
    record_and_yell(f"init fits object,{path}", log)
    hdul = loader(path)
    library = crudely_find_library(loader)
    record_and_yell(f"get header,{path}", log)
    header = get_header(hdul, hdu_ix, library)
    # initialize selected HDU object and get its data 'handle'
    record_and_yell(f"get data handle,{path}", log)
    hdu = hdul[hdu_ix]
    array_handle = hdu if library == "fitsio" else hdu.data
    # pick some boxes to slice from the HDU
    imsz = imsz_from_header(header)
    rng = np.random.default_rng(seed)
    boxes = rectangular_slices(imsz, rng=rng, **cut_settings)
    # TODO: hacky
    if shallow and len(imsz) == 3:
        bands = rng.integers(0, imsz[0], boxes.shape[1])
    else:
        bands = None
    # and then slice them!
    cuts = {}
    for cut_ix in range(boxes.shape[1]):
        record_and_yell(
            f"planning cuts,{path},{boxes[:, cut_ix, :].ravel()}", log
        )
        cuts[cut_ix] = perform_cut(array_handle, cut_ix, imsz, boxes, bands)
    for cut_ix, cut in cuts.items():
        record_and_yell(
            f"retrieve data,{path},{boxes[:, cut_ix, :].ravel()}", log
        )
        cuts[cut_ix] = cut.copy()
#     print(f"{watch.peek()} total seconds")
    return cuts, log


def get_cuts_from_files(paths, s3_settings, return_cuts, **cut_settings):
    """
    top-level benchmarking function: optionally (re)mount an s3 bucket,
    then take slices from various FITS files
    """
    log = {}
    # (re)mount s3 bucket to avoid 'cheating'
    record_and_yell(f"mounting bucket", log, loud=True)
    mount_bucket(**s3_settings)
    watch = Stopwatch(silent=True)
    watch.start()
    cuts = []
    for path in paths:
        record_and_yell(f"getting cuts,{path}", log, loud=True)
        path_cuts, path_log = get_cuts_from_file(path, **cut_settings)
        if return_cuts is True:
            cuts.append(path_cuts)
        else:
            del path_cuts
        log |= path_log
    runtime = watch.peek()
    if len(paths) >= 2:
        print(f"{runtime} total seconds for entire file list")
    return cuts, runtime, log
