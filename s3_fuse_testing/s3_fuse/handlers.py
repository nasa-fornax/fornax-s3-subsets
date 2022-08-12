"""
top-level handling functions for s3-slicing testing and benchmarks
"""
from pathlib import Path
from typing import Callable, Optional, Sequence, Any

import numpy as np
from gPhoton.io.fits_utils import logged_fits_initializer
from gPhoton.pretty import print_stats, notary
from killscreen.monitors import Stopwatch, Netstat

from s3_fuse.fits import imsz_from_header
from s3_fuse.random_generators import rectangular_slices
from s3_fuse.utilz import make_loaders, s3_url


def random_cuts_from_file(
    path: str,
    loader: Callable,
    hdu_ix: int,
    count: int,
    shape: Sequence[int],
    rng_seed: Optional[int] = None,
    astropy_handle_attribute: str = "data",
):
    """take random slices from a fits file; examine this process closely"""
    hdu_struct = logged_fits_initializer(
        path,
        loader,
        (hdu_ix,),
        get_handles=True,
        astropy_handle_attribute=astropy_handle_attribute
    )
    array_handle = hdu_struct['handles'][0]
    log, header, stat = [hdu_struct[k] for k in ('log', 'header', 'stat')]
    note = notary(log)
    # pick some boxes to slice from the HDU
    imsz = imsz_from_header(header)
    rng = np.random.default_rng(rng_seed)
    offsets = rectangular_slices(imsz, rng=rng, count=count, shape=shape)
    # and then slice them!
    cuts = {}
    for cut_ix in range(offsets.shape[0]):
        slices = tuple(
            np.apply_along_axis(lambda row: slice(*row), 1, offsets[cut_ix])
        )
        cuts[cut_ix] = array_handle[slices]
        note(f"planned cuts,{path},{stat()}")
    for cut_ix, cut in cuts.items():
        cuts[cut_ix] = cut.copy()
        note(f"got data,{path},{stat()}")
    return cuts, log


def benchmark_cuts(
    paths: Sequence[str],
    loader: Callable,
    shape: Sequence[int],
    count: int,
    hdu_ix: int,
    astropy_handle_attribute: str = "data",
    return_cuts: bool = False,
    n_files: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs
):
    paths = paths[:n_files] if n_files is not None else paths
    # set up monitors: timer, net traffic gauge, dict to put logs in
    watch, netstat, log = Stopwatch(silent=True), Netstat(), {}
    # stat is a formatting/printing function for time and net traffic results;
    # note simultaneously prints messages and puts them timestamped in 'log'
    stat, note = print_stats(watch, netstat), notary(log)
    watch.start()
    cuts = []
    for path in paths:
        path_cuts, path_log = random_cuts_from_file(
            path, loader, hdu_ix, count, shape, seed, astropy_handle_attribute
        )
        note(f"got cuts,{path},{stat()}", loud=True)
        if return_cuts is True:
            cuts.append(path_cuts)
        else:
            del path_cuts
        log |= path_log
    runtime = watch.peek()
    if len(paths) > 1:
        print(f"{runtime}s for total file list,,")
    return cuts, runtime, log


def interpret_benchmark_instructions(
    benchmark_name: str, general_settings: Optional[dict] = None
) -> list:
    """
    produce a set of arguments to random_cuts_from_files() using literals
    defined in a submodule of benchmark_settings
    """
    from copy import deepcopy
    from importlib import import_module
    from itertools import product

    instructions = import_module(f"benchmark_settings.{benchmark_name}")
    settings = {} if general_settings is None else deepcopy(general_settings)
    settings |= {
        "paths": instructions.TEST_FILES,
        "bucket": instructions.BUCKET,
        "hdu_ix": (instructions.HDU_IX,)
    }
    cases = []
    for shape, count, loader in product(
        instructions.CUT_SHAPES, instructions.CUT_COUNTS, instructions.LOADERS
    ):
        case = {
            'identifier': f"{benchmark_name}_{loader}_{shape}_{count}",
            "shape": shape,
            "count": count
        }
        case |= deepcopy(settings)
        case["loader"] = make_loaders(loader)[loader]
        if "section" in loader:
            # the necessary behavior changes in this case are slightly too
            # complex to implement them via straightforwardly wrapping
            # astropy.io.fits.open (or at least it would require
            # monkeypatching members of astropy.io.fits in ways I am not
            # comfortable with)
            case["bucket"] = None
            case["data_handle_attribute"] = "section"
            case["paths"] = tuple(
                map(lambda x: s3_url(settings["bucket"], x), case["paths"])
            )
        elif "mountpoint" in settings.keys():
            case["paths"] = tuple(
                (str(Path(settings['mountpoint'], p)) for p in case['paths'])
            )
        cases.append(case)
    return cases
