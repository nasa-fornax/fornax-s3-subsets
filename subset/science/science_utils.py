import os
import pickle
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from types import MappingProxyType

from astropy.wcs import WCS
from cytoolz import first
from cytoolz.curried import get
from dustgoggles.func import zero
from gPhoton.coadd import cut_skyboxes
from gPhoton.io.fits_utils import AgnosticHDUL
from killscreen.monitors import make_monitors
from killscreen.utilities import filestamp, roundstring

from utilz.fits import extract_wcs_keywords
from utilz.generic import cleanup_greedy_shm, summarize_stat


def agnostic_fits_skim(path, loader, get_wcs=True, hdu_indices=(0,), **kwargs):
    metadata = {
        "header": AgnosticHDUL(loader(path))[hdu_indices[0]].header,
        "path": path,
        **kwargs,
    }
    if get_wcs is True:
        metadata["system"] = WCS(extract_wcs_keywords(metadata["header"]))
    return metadata


def initialize_fits_chunk(
    chunk, bands, data_root, kwarg_assembler, loader, threads
):
    pool = Pool(threads) if threads is not None else None
    metadata = {}
    for band, identifier in product(bands, chunk):
        kwargs = kwarg_assembler(identifier, band, data_root, loader, bands)
        if pool is None:
            metadata[(identifier, band)] = agnostic_fits_skim(**kwargs)
        else:
            metadata[(identifier, band)] = pool.apply_async(
                agnostic_fits_skim, kwds=kwargs
            )
    if pool is not None:
        pool.close()
        pool.join()
        metadata |= {k: v.get() for k, v in metadata.items()}
    return metadata


def merge_chunk_metadata(
    named_targets, chunk_metadata, share_wcs=False, exptime_field="EXPTIME"
):
    plans = []
    for id_band, file_info in chunk_metadata.items():
        for target in named_targets[id_band[0]]:
            meta_dict = {
                "path": file_info["path"],
                "header": file_info["header"],
                "band": id_band[1],
                "exptime": file_info["header"][exptime_field],
            }
            if share_wcs is False:
                meta_dict["system"] = file_info["system"]
            else:
                meta_dict["system"] = chunk_metadata[
                    (id_band[0], first(map(get(1), chunk_metadata.keys())))
                ]["system"]
            plans.append(target.copy() | meta_dict)
    return plans


def cut_and_dump(
    plans,
    cut_kwargs,
    stat=zero,
    note=zero,
    return_cuts=False,
    verbose=1,
    outpath=None,
):
    chunk_cuts = cut_skyboxes(plans, **cut_kwargs)
    chunk_cuts = [plan | cut for plan, cut in zip(plans, chunk_cuts)]
    cleanup_greedy_shm(cut_kwargs["loader"])
    note(f"made {len(plans)} cutouts,{stat()}", verbose > 1)
    if outpath is not None:
        with outpath.with_extension(".pkl").open("wb+") as stream:
            pickle.dump(chunk_cuts, stream)
        note(f"dumped {len(plans)} cutouts to disk,{stat()}", verbose > 1)
    if return_cuts is False:
        for cut in chunk_cuts:
            del cut["arrays"]
    return chunk_cuts


def bulk_skycut(
    ids,
    targets,
    bands,
    chunker,
    kwarg_assembler,
    hdu_indices=(1,),
    data_root=os.getcwd(),
    dump_to=None,
    loader=None,
    return_cuts=False,
    threads=MappingProxyType({"image": None, "cut": None}),
    verbose=1,
    image_chunksize=40,
    name="chunk",
    share_wcs=False,
    exptime_field="EXPTIME"
):
    file_chunks, target_groups = chunker(ids, targets, bands, image_chunksize)
    results, tag = [], filestamp()
    stat, note = make_monitors(silent=True)
    for ix, chunk in enumerate(file_chunks):
        metadata = initialize_fits_chunk(
            chunk=chunk,
            bands=bands,
            data_root=data_root,
            kwarg_assembler=kwarg_assembler,
            loader=loader,
            threads=threads["image"],
        )
        plans = merge_chunk_metadata(
            target_groups, metadata, share_wcs, exptime_field
        )
        note(
            f"initialized {len(chunk) * len(bands)} images,{stat()}",
            verbose > 1,
        )
        if dump_to is None:
            outpath = None
        else:
            outpath = Path(dump_to, f"{name}_{ix}_{tag}")
        cut_kwargs = {
            "loader": loader,
            "hdu_indices": hdu_indices,
            "threads": threads["cut"],
        }
        results += cut_and_dump(
            plans, cut_kwargs, stat, note, return_cuts, verbose, outpath
        )
    note(
        f"made {len(results)} cuts from {len(ids) * len(bands)} "
        f"images,{roundstring(summarize_stat(stat))}",
        verbose > 0,
    )
    return results, note(eject=True)
