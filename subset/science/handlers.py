import os
from itertools import product, chain
from multiprocessing import Pool
from pathlib import Path
import pickle
from types import MappingProxyType
from typing import Optional, Literal

from cytoolz import groupby, first
from cytoolz.curried import get
from dustgoggles.func import zero
from gPhoton.coadd import coadd_image_slices, cut_skyboxes
from killscreen.monitors import make_monitors
from killscreen.utilities import filestamp, roundstring
import numpy as np
import pandas as pd
from photutils import aperture_photometry
import pyarrow as pa
from skimage.transform import resize

from subset.science.galex_utils import counts2mag
from subset.science.ps1_utils import ps1_flux2mag
from subset.science.science_utils import (
    agnostic_fits_skim,
    centered_aperture,
    centile_clip,
    normalize_range,
    pd_combinations,
)
from subset.utilz.generic import cleanup_greedy_shm, summarize_stat


# TODO, maybe: this can be generalized to work with more than just the g and
#  z bands, and with more than just these datasets, although none of that is
#  necessary for the demo.
def extract_cutout_photometry(
    cut_records,
    aperture_radius_arcsec=12.8,
    fields=("ra", "dec", "obj_id", "proj_cell", "sky_cell", "galex", "coords"),
):
    """
    compute multiband aperture photometry on a collection of PS1 and GALEX
    cutouts produced by functions like bulk_skycut()
    """
    cubes = groupby(get("obj_id"), cut_records)
    results = []
    # noinspection PyArgumentList
    for cube in cubes.values():
        result = {k: v for k, v in cube[0].items() if k in fields}
        for record in cube:
            band = record["band"]
            counts = aperture_photometry(
                record["array"],
                centered_aperture(record, aperture_radius_arcsec),
            )["aperture_sum"][0]
            if band not in ("NUV", "FUV"):
                result[f"{band}_mag"] = ps1_flux2mag(counts, record["exptime"])
            else:
                # note that GALEX coadds were converted to cps during coadding
                result[f"{band}_mag"] = counts2mag(counts, band)
        results.append(result)
    return compute_uv_mag_offset(pd.DataFrame(results))


def compute_uv_mag_offset(results):
    results["ex"] = (
        results["NUV_mag"] - (results["g_mag"] + results["z_mag"]) / 2
    )
    ex_na = results.loc[results["ex"].isna()]
    for band in ("g", "z"):
        band_ex = ex_na.loc[ex_na[f"{band}_mag"].notna()]
        results.loc[band_ex.index, "ex"] = (
            band_ex["NUV_mag"] - band_ex[f"{band}_mag"]
        )
    return results


def coadd_galex_cutouts(cutouts, scale=None, band="NUV"):
    coadds = []
    # cutouts chunked by obj_id -- each item in obj_slices
    # is a list of cuts around a single sky position
    # taken from all eclipses which we found that sky position
    obj_slices = groupby("obj_id", cutouts)
    # noinspection PyArgumentList
    for obj_id, images in obj_slices.items():
        if len(images) == 0:
            print("all GALEX images for {obj_id} are bad, skipping")
        coadd, system, exptime = coadd_image_slices(images, scale)
        coadds.append(
            {
                "array": coadd,
                "system": system,
                "exptime": exptime,
                "obj_id": obj_id,
                "band": band,
            }
        )
    return coadds


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
    exptime_field="EXPTIME",
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


def ps_stack_norm(array, low=25, high=99.9):
    clipped = centile_clip(array, (low, high))
    return normalize_range(clipped)


def ps_galex_stack(
    ps0,
    ps1,
    galex,
    channel_order=("red", "green", "blue"),
    ps_norm=ps_stack_norm,
    lift_threshold=None,
    ps_range=(25, 99.9),
    galex_range=(1, 99),
):
    """visualization function for fused ps/galex images."""
    channels = {0: ps_norm(ps0, *ps_range), 1: ps_norm(ps1, *ps_range)}
    channels[2] = np.mean([channels[0], channels[1]], axis=0)
    if lift_threshold is not None:
        channels[2][
            channels[2] > np.percentile(channels[2], [lift_threshold])[0]
        ] = 1
    channels[2] *= centile_clip(galex, galex_range)
    channels[2] = normalize_range(channels[2])
    channel_map = {"red": 0, "green": 1, "blue": 2}
    return np.dstack([channels[channel_map[name]] for name in channel_order])


def load_skycell_index(skycell_path):
    return pa.csv.read_csv(skycell_path).cast(
        pa.schema([("proj_cell", pa.uint16()), ("sky_cell", pa.uint8())])
    )


def filter_ps1_catalog(
    catalog: pd.DataFrame,
    mag_cutoff: Optional[int],
    extension_type: Literal[None, "extended", "point"],
    stack_only: bool = True
):
    if mag_cutoff is not None:
        catalog = catalog.loc[
            (catalog["g_mpsf_mag"] < mag_cutoff)
            & (catalog["z_mpsf_mag"] < mag_cutoff)
        ]
    if extension_type == "extended":
        catalog = catalog.loc[catalog["flag"] % 2 == 1]
    elif extension_type == "point":
        catalog = catalog.loc[catalog["flag"] % 2 == 0]
    if stack_only is True:
        catalog = catalog.loc[catalog['n_stack_detections'] > 0]
    return catalog


def sample_ps1_catalog(catalog, target_count, max_cell_count):
    if max_cell_count is not None:
        cells = pd_combinations(catalog, ["proj_cell", "sky_cell"]).sample(
            max_cell_count
        )
        targets = filter_to_cells(catalog, cells).sample(target_count)
    else:
        targets = catalog.sample(target_count)
    return targets.to_dict(orient="records")


def get_corresponding_images(targets, available_eclipses):
    ps1_stacks = {f"{t['proj_cell']}_{t['sky_cell']}" for t in targets}
    galex_visits = {
        e
        for e in tuple(chain.from_iterable(map(get("galex"), targets)))
        if e in available_eclipses.values
    }
    return ps1_stacks, galex_visits


# TODO, maybe: generalize to work with other GALEX & PAN-STARRS bands
def cutouts_to_channels(cutouts):
    """
    take a collection of PAN-STARRS and GALEX cutout results, as produced
    by functions like bulk_skycut(). group them by PAN-STARRS object ID,
    upsample GALEX cutouts to fit PAN-STARRS cutouts, and return them in a
    data structure suitable for scaling and stacking into RGB images.
    """
    channel_dict = {}
    for obj_id in set(map(get("obj_id"), cutouts)):
        matches = [c for c in cutouts if c["obj_id"] == obj_id]
        nuv_band = [c for c in matches if c["band"] == "NUV"]
        if len(nuv_band) == 0:
            print(f"no valid GALEX coadd for {obj_id}")
            continue
        z_band = [c for c in matches if c["band"] == "z"]
        g_band = [c for c in matches if c["band"] == "g"]
        assert all([len(b) == 1 for b in (nuv_band, z_band, g_band)])
        nuv_cut = nuv_band[0]["array"]
        z_cut = z_band[0]["array"]
        g_cut = g_band[0]["array"]
        assert z_cut.shape == g_cut.shape
        # fortunately, ps1 and galex both use gnomonic projections,
        # so little spatial distortion is added by stretching them to fit
        # one another. we don't care about meticulous pixel positioning,
        # because the pixels of these images don't actually represent
        # physical resolving elements, so it's fine to just use a generic
        # upsampling function (the images are just for visualization anyway).
        # TODO: does the anti_aliasing kwarg do anything here?
        nuv_upsample = resize(
            nuv_cut, g_cut.shape, order=0, anti_aliasing=False
        )
        channel_dict[obj_id] = {"z": z_cut, "g": g_cut, "nuv": nuv_upsample}
    return channel_dict


def filter_to_cells(df, cells):
    proj = df.loc[df["proj_cell"].isin(cells["proj_cell"])]
    return proj.loc[proj["sky_cell"].isin(cells["sky_cell"])]


def load_ps1_cutout_masks(ps1_cutouts):
    import fitsio

    masks = {}
    for cut in ps1_cutouts:
        coords = cut["coords"]
        slices = (
            slice(coords[2], coords[3] + 1),
            slice(coords[0], coords[1] + 1),
        )
        mask = fitsio.FITS(cut["path"].replace(".fits", ".mask.fits"))[1][
            slices
        ]
        masks[(cut["obj_id"], cut["band"])] = mask
    return masks
