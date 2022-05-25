"""
stream handling, generic utilities, etc.
"""
import os
import shutil
from functools import partial
from io import BytesIO
import random
from pathlib import Path
from typing import Callable


def preload_target(function, path, *args, **kwargs):
    """
    `function` should be able to accept both streams and paths.
    `preload_target` fully loads path into an in-memory object
    before passing it to `function`, hopefully preempting any clever
    lazy / random-read / memmap / etc. behavior on the part of function,
    OS-level filesystem handlers, etc.
    """
    buffer = BytesIO()
    with open(path, 'rb') as file:
        buffer.write(file.read())
    buffer.seek(0)
    return function(buffer, *args, **kwargs)


def preload_from_shm(function, path, *args, **kwargs):
    """
    preloader for functions that will not accept buffers -- thin wrappers
    to C extensions like fitsio.FITS, for instance. will only work on Linux
    and may fail in some Linux environments depending on security settings.
    """
    os.makedirs('/dev/shm/slicetemp', exist_ok=True)
    shutil.copy(path, '/dev/shm/slicetemp')
    return function(f"/dev/shm/slicetemp/{Path(path).name}", *args, **kwargs)


def strip_irrelevant_kwargs(func, *args, **kwargs):
    """
    call a function with subsets of kwargs until it agrees to be called.
    quick hacky way to enable a consistent interface for callables whose real
    signatures cannot be inspected because they live inside extensions or
    whatever.
    """
    while len(kwargs) > 0:
        try:
            return func(*args, **kwargs)
        except TypeError as error:
            if "unexpected keyword" not in str(error):
                raise error
            bad_kwarg = str(error).split("'")[1]
            kwargs.pop(bad_kwarg)
    return func(*args)


def sample_table(table, k=None):
    """
    select a sample of size k from rows of a pyarrow Table.
    """
    if k is None:
        return table
    return table.take(random.sample(range(len(table)), k=k))


def make_loaders(*loader_names: str) -> dict[str, Callable]:
    """
    produce a mapping from FITS-loader names to callable load methods.
    currently only three are defined by default.
    """
    loaders = {}
    for name in loader_names:
        if name == "astropy":
            import astropy.io.fits
            loaders[name] = astropy.io.fits.open
        elif name == "fitsio":
            import fitsio
            loaders[name] = fitsio.FITS
        # "greedy" version of astropy.io.fits.open, which fully loads a file
        # into memory before doing anything with it. a useful bench reference.
        # note that fitsio.FITS will not accept filelike objects and cannot be
        # wrapped in this way without modifying its C extensions.
        elif name == "greedy_astropy":
            import astropy.io.fits
            loaders[name] = partial(preload_target, astropy.io.fits.open)
        elif name == "greedy_fitsio":
            import fitsio
            # fitsio is a thinnish wrapper for cfitsio and cannot be
            # transparently initialized from python buffer objects.
            # this will probably fail on non-Linux systems and may fail
            # in some Linux environments, depending on security settings.
            loaders[name] = partial(preload_from_shm, fitsio.FITS)

    return loaders
