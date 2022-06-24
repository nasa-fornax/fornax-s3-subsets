"""
galex-specific utilities.
some are vendored from gPhoton 2.
others require a gPhoton 2 installation in the environment.
"""
import random
import warnings
from itertools import chain, product
from multiprocessing import Pool
from typing import Sequence, Any

from gPhoton.aspect import TABLE_PATHS
from gPhoton.coadd import cut_skyboxes
from gPhoton.io.fits_utils import logged_fits_initializer
from gPhoton.pretty import make_monitors
from gPhoton.reference import eclipse_to_paths
from more_itertools import chunked
from pyarrow import parquet

from s3_fuse.utilz import cleanup_greedy_shm


def get_galex_version_path(eclipse, band, depth, obj, version, data_root):
    """
    get file path for the GALEX data object with specified eclipse, band,
    depth, object type, and compression version, relative to data_root.
    this is a semistandardized convention loosely based on gPhoton conventions
    (which don't include multiple compression types!)
    """
    paths = eclipse_to_paths(eclipse, data_root, depth)
    return {
        "rice": paths[band][obj].replace(".fits.gz", "_rice.fits"),
        "none": paths[band][obj].replace(".gz", ""),
        "gz": paths[band][obj],
    }[version]


def pick_galex_eclipses(count=5, eclipse_type="mislike"):
    """randomly select a set of GALEX eclipses matching some criteria"""
    # this is the set of actually-available eclipses. it should be adjusted to
    # contain actually-available eclipses in your bucket of choice, or whatever
    # restricted subset you would like.
    with open("extant_eclipses.txt") as file:
        extant_eclipses = set(map(int, file.readlines()))

    # typically larger files, ~10000 x 10000 pixels per frame
    if eclipse_type == "complex":
        columns, predicates, refs = ["legs"], [">"], [1]
    # typically smaller files, ~3000x3000 pixels per frame
    elif eclipse_type == "mislike":
        columns, predicates, refs = (
            ["legs", "obstype"],
            ["=", "in"],
            [0, ["MIS", "DIS", "GII"]],
        )
    # anything!
    else:
        columns, predicates, refs = [], [], []
    eclipse_slice = parquet_generic_search(
        columns, predicates, refs, table_path=TABLE_PATHS["metadata"]
    )
    sliced_eclipses = set(eclipse_slice["eclipse"].to_pylist())
    eclipses_of_interest = extant_eclipses.intersection(sliced_eclipses)
    actual_count = min(count, len(eclipses_of_interest))
    if actual_count < count:
        warnings.warn("only {len(eclipses_of_interest)} available")
    return random.sample(tuple(eclipses_of_interest), k=actual_count)


def parquet_generic_search(columns, predicates, refs, table_path):
    filters = [
        (column, predicate, ref)
        for column, predicate, ref in zip(columns, predicates, refs)
    ]
    return parquet.read_table(table_path, filters=filters)


# TODO, maybe: merge these with PS1 functions in some way --
#  annoying because requires some kind of chunked parameter passing
#  or giving up on chunked WCS init in PS1
def get_galex_cutouts(
    eclipses,
    loader,
    targets,
    length,
    data_root,
    bands,
    verbose=1,
    logged=True,
    image_chunksize: int = 40,
    image_threads=None,
    cut_threads=None,
):
    stat, note = make_monitors(fake=not logged, silent=True)
    eclipse_chunks = chunked(eclipses, int(image_chunksize / len(bands)))
    eclipse_targets = {
        eclipse: tuple(filter(lambda t: eclipse in t['galex'], targets))
        for eclipse in eclipses
    }
    cuts = []
    for chunk in eclipse_chunks:
        metadata = initialize_galex_chunk(
            loader=loader,
            bands=bands,
            chunk=chunk,
            threads=image_threads,
            data_root=data_root
        )
        plans = []
        for name, file_info in metadata.items():
            eclipse, band = name
            for target in eclipse_targets[eclipse]:
                meta_dict = {
                    'wcs': metadata[(eclipse, band)]['wcs'],
                    'path': metadata[(eclipse, band)]['path'],
                    'band': band,
                    'eclipse': eclipse
                }
                plans.append(target.copy() | meta_dict)
        note(
            f"initialized {len(chunk) * len(bands)} images,{stat()}",
            verbose > 1
        )
        cut_kwargs = {
            "loader": loader, "hdu_indices": (1, 2, 3), "side_length": length
        }
        cuts += cut_skyboxes(plans, cut_threads, cut_kwargs)
        cleanup_greedy_shm(loader)
        note(f"made {len(plans)} cutouts,{stat()}", verbose > 1)
    note(
        f"made {len(cuts)} cuts from {len(eclipses) * len(bands)} images,"
        f"{stat(total=True)}",
        verbose > 0
    )
    return cuts, note(None, eject=True)


def initialize_galex_chunk(
    loader,
    bands,
    chunk: Sequence[int],
    threads,
    data_root
) -> dict[tuple[int, str], Any]:
    pool = Pool(threads) if threads is not None else None
    metadata = {}
    for band, eclipse in product(bands, chunk):
        init_params = {
            # our canonical GALEX path structure, except that these test files
            # do not have 'rice' in the name despite being RICE-compressed
            "path":  eclipse_to_paths(
                eclipse, data_root, None, "none"
            )[band]["image"],
            "get_wcs": True,
            "loader": loader,
            "hdu_indices": (1, 2, 3)
        }
        if pool is None:
            metadata[(eclipse, band)] = logged_fits_initializer(**init_params)
        else:
            metadata[(eclipse, band)] = pool.apply_async(
                logged_fits_initializer, kwds=init_params
            )
    if pool is not None:
        pool.close()
        pool.join()
        metadata |= {k: v.get() for k, v in metadata.items()}
    return metadata
