from functools import partial
from itertools import product

from s3_fuse.random_generators import RNG


def poisson(lamb, dtype, size):
    return RNG.poisson(lamb, size)


def normal(loc, scale, dtype, size):
    return RNG.normal(loc, scale, size)


def uniform(low, high, dtype, size):
    return RNG.uniform(low, high, size)


def power(exp, scale, dtype, size):
    return RNG.power(exp, size) * scale


def construct_prefix(shape, generator, dtype, hdu_count, compression_type):
    return (
        f"{shape}_{generator}_{dtype}_"
        f"{str(compression_type).lower().replace('_1', '')}_{hdu_count}hdu"
    )


SHAPES = {
    "small square": (1000, 1000),
    "medium square": (4000, 4000),
    #     "medium unsquare": (2000, 6000),
    "large square": (12000, 12000),
    #     "large unsquare": (4000, 20000),
    "small multiband": (1000, 1000, 50),
    #     "medium multiband": (4000, 4000, 50)
}

GENERATORS = {
    "normal 0": partial(normal, 0, 1),
    "normal 1": partial(normal, 0, 100),
    #     "power 0": partial(power, 0.1, 10),
    #     "power 1": partial(power, 2, 255),
    #     "uniform 0": partial(uniform, 0, 1),
    "uniform 1": partial(uniform, -100000, 100000)
}

DTYPES = ("int8", "float32", "float64")

HDU_COUNTS = (1, 3)

# TODO, probably: set a larger default tilesize for 3-axis arrays
COMPRESSION_TYPES = (None, "RICE_1")

CASES = {}
for case in product(SHAPES, GENERATORS, DTYPES, HDU_COUNTS, COMPRESSION_TYPES):
    if case[0].startswith("large") and case[3] > 1:
        continue
    CASES[construct_prefix(*case)] = case
