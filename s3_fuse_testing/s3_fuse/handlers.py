"""
top-level handling functions for s3-slicing testing and benchmarks
"""
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

from gPhoton.io.fits_utils import logged_fits_initializer
from gPhoton.pretty import print_stats, notary
from killscreen.monitors import Stopwatch, Netstat
import numpy as np

from s3_fuse.fits import imsz_from_header
from s3_fuse.mount_s3 import mount_bucket
from s3_fuse.random_generators import rectangular_slices


def random_cuts_from_file(
    path: Path,
    loader: Callable,
    hdu_ix: int,
    cut_settings: Mapping,
    seed: Optional[int] = None,
    verbose=0
):
    """take random slices from a fits file; examine this process closely"""
    hdu_struct = logged_fits_initializer(
        path, loader, [hdu_ix], False, True, verbose
    )
    array_handle = hdu_struct['handles'][0]
    log, header, stat = [hdu_struct[k] for k in ('log', 'header', 'stat')]
    note = notary(log)
    # pick some boxes to slice from the HDU
    imsz = imsz_from_header(header)
    rng = np.random.default_rng(seed)
    offsets = rectangular_slices(imsz, rng=rng, **cut_settings)
    # and then slice them!
    cuts = {}
    for cut_ix in range(offsets.shape[0]):
        slices = tuple(
            np.apply_along_axis(lambda row: slice(*row), 1, offsets[cut_ix])
        )
        cuts[cut_ix] = array_handle[slices]
        note(f"planned cuts,{path},{stat()}", loud=verbose > 1)
    for cut_ix, cut in cuts.items():
        cuts[cut_ix] = cut.copy()
        note(f"got data,{path},{stat()}", loud=verbose > 1)
    return cuts, log


def random_cuts_from_files(paths, s3_settings, return_cuts, **cut_settings):
    """
    top-level benchmarking function: optionally (re)mount an s3 bucket,
    then take slices from various FITS files
    """
    watch, netstat, log = Stopwatch(silent=True), Netstat(), {}
    stat, note = print_stats(watch, netstat), notary(log)
    # (re)mount s3 bucket to avoid 'cheating'
    mount_bucket(**s3_settings)
    note(f"mounted bucket,,{stat()}", loud=True)
    watch = Stopwatch(silent=True)
    watch.start()
    cuts = []
    for path in paths:
        # TODO: confusing to call these separate objects both cut_settings
        path_cuts, path_log = random_cuts_from_file(path, **cut_settings)
        note(f"got cuts,{path},{stat()}", loud=True)
        if return_cuts is True:
            cuts.append(path_cuts)
        else:
            del path_cuts
        log |= path_log
    runtime = watch.peek()
    if len(paths) >= 2:
        print(f"{runtime} total seconds for entire file list")
    return cuts, runtime, log


