from typing import Sequence, Union

from astropy.wcs import WCS
import numpy as np
from gPhoton.io.fits_utils import AgnosticHDUL
from photutils import CircularAperture

from subset.utilz.fits import extract_wcs_keywords


def agnostic_fits_skim(path, loader, get_wcs=True, hdu_indices=(0,), **kwargs):
    metadata = {
        "header": AgnosticHDUL(loader(path))[hdu_indices[0]].header,
        "path": path,
        **kwargs,
    }
    if get_wcs is True:
        metadata["system"] = WCS(extract_wcs_keywords(metadata["header"]))
    return metadata


def pd_combinations(df, columns):
    return df[columns].value_counts().index.to_frame().reset_index(drop=True)


def centered_aperture(cut_record, radius_arcsec):
    system, array = cut_record["system"], cut_record["array"]
    system = system.to_header()
    center = array.shape[1] / 2, array.shape[0] / 2
    degrees_per_pixel = np.abs(
        np.array([system["CDELT1"], system["CDELT2"]])
    ).mean()
    return CircularAperture(
        (center,), r=radius_arcsec / 3600 / degrees_per_pixel
    )


# NOTE: The following visualization-support functions are vendored from
# marslab (https://github.com/MillionConcepts/marslab); its license is
# included by reference.


def find_masked_bounds(image, cheat_low, cheat_high):
    """
    relatively memory-efficient way to perform bound calculations for
    normalize_range on a masked array.
    """
    valid = image[~image.mask].data
    if valid.size == 0:
        return None, None
    if (cheat_low != 0) and (cheat_high != 0):
        minimum, maximum = np.percentile(
            valid, [cheat_low, 100 - cheat_high], overwrite_input=True
        ).astype(image.dtype)
    elif cheat_low != 0:
        maximum = valid.max()
        minimum = np.percentile(valid, cheat_low, overwrite_input=True).astype(
            image.dtype
        )
    elif cheat_high != 0:
        minimum = valid.min()
        maximum = np.percentile(
            valid, 100 - cheat_high, overwrite_input=True
        ).astype(image.dtype)
    else:
        minimum = valid.min()
        maximum = valid.max()
    return minimum, maximum


# noinspection PyArgumentList
def find_unmasked_bounds(image, cheat_low, cheat_high):
    """straightforward way to find unmasked array bounds for normalize_range"""
    if cheat_low != 0:
        minimum = np.percentile(image, cheat_low).astype(image.dtype)
    else:
        minimum = image.min()
    if cheat_high != 0:
        maximum = np.percentile(image, 100 - cheat_high).astype(image.dtype)
    else:
        maximum = image.max()
    return minimum, maximum


def centile_clip(image, centiles=(1, 99)):
    """
    simple clipping function that clips values above and below a given
    percentile range
    """
    finite = np.ma.masked_invalid(image)
    bounds = np.percentile(finite[~finite.mask].data, centiles)
    result = np.ma.clip(finite, *bounds)
    if isinstance(image, np.ma.MaskedArray):
        return result
    return result.data


def normalize_range(
    image: np.ndarray,
    bounds: Sequence[int] = (0, 1),
    stretch: Union[float, tuple[float, float]] = 0,
    inplace: bool = False,
) -> np.ndarray:
    """
    simple linear min-max scaler that optionally percentile-clips the input at
    stretch = (low_percentile, 100 - high_percentile). if inplace is True,
    may transform the original array, with attendant memory savings and
    destructive effects.
    """
    if isinstance(stretch, Sequence):
        cheat_low, cheat_high = stretch
    else:
        cheat_low, cheat_high = (stretch, stretch)
    range_min, range_max = bounds
    if isinstance(image, np.ma.MaskedArray):
        minimum, maximum = find_masked_bounds(image, cheat_low, cheat_high)
        if minimum is None:
            return image
    else:
        minimum, maximum = find_unmasked_bounds(image, cheat_low, cheat_high)
    if not ((cheat_high is None) and (cheat_low is None)):
        if inplace is True:
            image = np.clip(image, minimum, maximum, out=image)
        else:
            image = np.clip(image, minimum, maximum)
    if inplace is True:
        # perform the operation in-place
        image -= minimum
        image *= range_max - range_min
        if image.dtype.char in np.typecodes["AllInteger"]:
            # this loss of precision is probably better than
            # automatically typecasting it.
            # TODO: detect rollover cases, etc.
            image //= maximum - minimum
        else:
            image /= maximum - minimum
        image += range_min
        return image
    return (image - minimum) * (range_max - range_min) / (
        maximum - minimum
    ) + range_min
