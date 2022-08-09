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


def _procrusteanize(parameters, reference_length, fill_value=0):
    difference = reference_length - len(parameters)
    if difference > 0:
        return list(parameters) + [fill_value for _ in range(difference)]
    if difference < 0:
        return parameters[:reference_length]
    return parameters


def rectangular_slices(
    imsz: Sequence[int],
    cut_count: int,
    lengths: Sequence[int],
    variances: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    generate coordinates for random rectangular slices.

    returns an n x m x 2 array ("offsets"), where n is the number of slices to
    take from the image and m is the number of axes in the image to be sliced.

    then offsets[i] gives specifications for the starting and ending
    indices of cut i, one axis per row, starting index in the 0th column,
    ending index in the 1st column, e.g.:
    slices = np.apply_along_axis(lambda row: slice(*row), 1, offsets[i])
    """
    # if cut_sizes is specified to lower dimensionality than the image fill
    # with 1s; if higher, truncate. similarly with variances, but fill with 0s.
    lengths = _procrusteanize(lengths, len(imsz), 1)
    if variances is not None:
        variances = _procrusteanize(variances, len(imsz), 0)
    if rng is None:
        rng = np.random.default_rng()
    offsets = np.array(
        [
            rng.integers(0, axis_size - axis_cut_size, cut_count)
            for axis_size, axis_cut_size in zip(imsz, lengths)
        ]
    ).T
    offsets = np.dstack([offsets, offsets + np.array(lengths)])
    if variances is None:
        return offsets
    for ax_ix, variance in enumerate(variances):
        if variance < 1:
            continue
        per_cut_variances = rng.integers(-variance, variance, cut_count)
        offsets[:, ax_ix, 1] = np.clip(
            offsets[:, ax_ix, 1] + per_cut_variances, 0, imsz[ax_ix]
        )
    return offsets


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
