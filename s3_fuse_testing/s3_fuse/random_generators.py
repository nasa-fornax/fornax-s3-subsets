"""
random generators.

note 1: these could be used to build hypothesis strategies later if that
framework seems useful for testing.

note 2: very design phase-y. kludgey, too many options (or too few?),
inefficient.
"""
from functools import partial
import io
from typing import Sequence, Optional, Union, Callable

import numpy as np

from s3_fuse.utilz import strip_irrelevant_kwargs

RNG = np.random.default_rng()


def rectangular_slices(
    imsz: Sequence[int],
    cut_count: int,
    box_size: int,
    rng: Optional[np.random.Generator] = None,
    size_variance: int = 0,
) -> np.ndarray:
    """generate coordinates for random rectangular slices"""
    if rng is None:
        rng = np.random.default_rng()
    offsets = np.array([rng.integers(0, imsz[ix], cut_count) for ix in (0, 1)])
    offsets = np.dstack([offsets, offsets + box_size])
    if size_variance == 0:
        return offsets
    variances = rng.integers(-size_variance, size_variance, offsets.shape)
    offsets += variances
    return np.array([np.clip(offsets[ix], 0, imsz[ix]) for ix in (0, 1)])


def fits_file(
    size: Sequence[int] = (0, 0),
    dtype: Union[str, np.dtype] = np.uint8,
    element_generator: Union[str, Callable] = "poisson",
    generator_args: tuple = (),
    compression_type: Optional[str] = None,
    base_hdu_count: int = 1,
    return_bytes = True,
    **hdu_constructor_kwargs
) -> Union[io.BytesIO, "astropy.io.fits.HDUList"]:
    """
    construct a randomish FITS file containing the specified number of HDUs
    with the specified attributes. always prepends a primary HDU for
    HDU list length consistency between compressed and uncompressed HDU types.
    cannot construct a file with multiple types of HDU or work with types of
    HDU other than image and compressed image (perhaps later).

    by default returns the fits file as a BytesIO object.
    pass return_bytes = False to return it as an astropy.io.fits.HDUList.
    """
    import astropy.io.fits

    if compression_type is not None:
        hdu_constructor = partial(
            astropy.io.fits.CompImageHDU,
            compression_type=compression_type,
            **hdu_constructor_kwargs
        )
    else:
        hdu_constructor = partial(
            astropy.io.fits.ImageHDU, **hdu_constructor_kwargs
        )
    if isinstance(element_generator, str):
        element_generator = getattr(RNG, element_generator)
    arrays = []
    for _ in range(base_hdu_count):
        array = strip_irrelevant_kwargs(
            element_generator, *generator_args, size=size, dtype=dtype
        )
        if array.dtype != np.dtype(dtype):
            array = array.astype(dtype)
        arrays.append(array)
    hdu_list = astropy.io.fits.HDUList(
        [astropy.io.fits.PrimaryHDU(), *[hdu_constructor(a) for a in arrays]]
    )
    if return_bytes is False:
        return hdu_list
    stream = io.BytesIO()
    hdu_list.writeto(stream)
    stream.seek(0)
    return stream
