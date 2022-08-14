"""
top-level handling functions for s3-subsetting benchmarks
"""
from copy import deepcopy
from functools import partial, reduce
from importlib import import_module
from itertools import product
from operator import mul
from pathlib import Path
from typing import Callable, Optional, Sequence, Mapping

import numpy as np
import pandas as pd
from gPhoton.pretty import notary, print_inline
from killscreen.monitors import Stopwatch, Netstat, CPUMonitor, print_stats

from s3_fuse.fits import imsz_from_header, logged_fits_initializer
from s3_fuse.random_generators import rectangular_slices
from s3_fuse.utilz import (
    make_loaders,
    s3_url,
    load_first_aws_credential,
    Throttle,
)


def random_cuts_from_file(
    path: str,
    loader: Callable,
    hdu_ix: int,
    count: int,
    shape: Sequence[int],
    rng=None,
    astropy_handle_attribute: str = "data",
    preload_hdu: bool = False,
):
    """take random slices from a fits file; examine this process closely"""
    hdu_struct = logged_fits_initializer(
        path,
        loader,
        (hdu_ix,),
        get_handles=True,
        astropy_handle_attribute=astropy_handle_attribute,
        preload_hdus=preload_hdu,
    )
    array_handle = hdu_struct["handles"][0]
    log, header, stat = [hdu_struct[k] for k in ("log", "header", "stat")]
    note = notary(log)
    # pick some boxes to slice from the HDU
    imsz = imsz_from_header(header)
    if rng is None:
        rng = np.random.default_rng()
    indices = rectangular_slices(imsz, rng=rng, count=count, shape=shape)
    # and then slice them!
    cuts = {}
    for cut_ix in range(indices.shape[0]):
        slices = tuple(
            np.apply_along_axis(lambda row: slice(*row), 1, indices[cut_ix])
        )
        # we perform this unusual-looking copy because astropy does not, in
        # general, actually copy memmapped data into memory in response to the
        # __getitem__ call. so if we do not do _something_ with said data,
        # astropy will in most cases never actually retrieve it (unless we've
        # forced it to be "greedy" or similar, of course)
        cuts[cut_ix] = array_handle[slices].copy()
        note(f"got cut {cut_ix},{path},{stat()}")
    note(f"file done,{path},{stat(total=True)}")
    cuts["indices"] = indices
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
    aws_credentials_path: Optional[str] = None,
    authenticate_s3: bool = False,
    preload_hdu: bool = False,
    **_,
):
    paths = paths[:n_files] if n_files is not None else paths
    # set up monitors: timer, net traffic gauge, cpu timer, dict to put logs in
    watch = Stopwatch(silent=True, digits=None)
    netstat = Netstat(round_to=None)
    cpumon = CPUMonitor(round_to=None)
    log = {}
    # stat is a formatting/printing function for monitor results;
    # note simultaneously prints messages and puts them timestamped in 'log'
    stat, note = print_stats(watch, netstat, cpumon), notary(log)
    rng = np.random.default_rng(seed)
    cuts = []
    # although fsspec uses boto3, it doesn't appear to use boto3's default
    # session initialization, so doesn't automatically load default aws
    # credentials. so, if we're looking at s3:// URIs, we manually load
    # creds just in case those URIs reference access-restricted resources.
    # this is a crude credential-loading function that just loads the first
    # credential in the specified file (by default ~/.aws/credentials). we
    # could do it better if we needed to generalize this, probably by messing
    # with the client/resource objects passed around by fsspec.
    if paths[0].startswith("s3://") and (authenticate_s3 is True):
        creds = load_first_aws_credential(aws_credentials_path)
        loader = partial(loader, fsspec_kwargs=creds)
    watch.start()
    print_inline(f"0/{len(paths)} complete")
    for i, path in enumerate(paths):
        path_cuts, path_log = random_cuts_from_file(
            path,
            loader,
            hdu_ix,
            count,
            shape,
            rng,
            astropy_handle_attribute,
            preload_hdu,
        )
        if return_cuts is True:
            cuts.append(path_cuts)
        else:
            del path_cuts
        log |= path_log
        print_inline(f"{i + 1}/{len(paths)} complete")
    return cuts, stat, log


def interpret_benchmark_instructions(
    benchmark_name: str, general_settings: Optional[dict] = None
) -> list:
    """
    produce a set of arguments to random_cuts_from_files() using literals
    defined in a submodule of benchmark_settings
    """
    instructions = import_module(
        f"s3_fuse.benchmark_settings.{benchmark_name}"
    )
    settings = {} if general_settings is None else deepcopy(general_settings)
    settings |= {
        "paths": instructions.TEST_FILES,
        "bucket": instructions.BUCKET,
        "hdu_ix": instructions.HDU_IX,
        "authenticate_s3": instructions.AUTHENTICATE_S3,
    }
    cases = []
    throttle_speeds = general_settings.get("bandwidth")
    throttle_speeds = (None,) if throttle_speeds is None else throttle_speeds
    for element in product(
        instructions.CUT_SHAPES,
        instructions.CUT_COUNTS,
        throttle_speeds,
        instructions.LOADERS,
    ):
        shape, count, throttle, loader = element
        title_parts = (
            benchmark_name,
            loader,
            str(count),
            "_".join(map(str, shape)),
            str(int(throttle / 1000)) if throttle is not None else "None",
        )
        case = {
            "title": "-".join(title_parts),
            "shape": shape,
            "count": count,
            "throttle": throttle,
            "loader": make_loaders(loader)[loader],
        } | deepcopy(settings)
        if "s3" in loader:
            case["bucket"] = None
            case["paths"] = tuple(
                map(lambda x: s3_url(settings["bucket"], x), case["paths"])
            )
        elif "mountpoint" in settings.keys():
            case["paths"] = tuple(
                (str(Path(settings["mountpoint"], p)) for p in case["paths"])
            )
        if "section" in loader:
            # the necessary behavior changes in this case are slightly too
            # complex to implement them via straightforwardly wrapping
            # astropy.io.fits.open (or at least it would require
            # monkeypatching members of astropy.io.fits in ways I am not
            # comfortable with), so we pass a special argument to instruct
            # downstream functions to access the "section" attribute
            # of HDUs instead of the "data" attribute.
            case["astropy_handle_attribute"] = "section"
        if "preload_hdu" in loader:
            # 'semigreedy'. load the full extension before beginning slicing.
            case["preload_hdu"] = True
        cases.append(case)
    return cases


def process_bench_stats(log, test_case, benchmark_name):
    """do lots of nasty formatting to make nice aggregate statistics"""
    log_df = pd.DataFrame.from_dict(log, orient="index")
    log_df["time"] = log_df.index
    fields = log_df[0].str.split(",", expand=True)
    cpu_times = fields[4].str.replace("cpu", "").str.split(";", expand=True)
    cpu_times.columns = "cpu_" + cpu_times.iloc[0].str.extract(r"(\w+)")[0]
    for c in cpu_times.columns:
        cpu_times[c] = cpu_times[c].str.extract(r"([\d\.]+)").astype(float)
    fields = fields.drop(columns=4)
    fields.columns = ["event", "path", "duration_str", "volume_str"]
    log_df = pd.concat([log_df, fields], axis=1)
    log_df["duration"] = (
        log_df["duration_str"].str.extract(r"([\d\.]+)").astype(float)
    )
    log_df["volume"] = (
        log_df["volume_str"].str.extract(r"([\d\.]+)").astype(float)
    )
    log_df["path"] = log_df["path"].map(lambda p: Path(p).name)
    log_df = (
        pd.concat([log_df.drop(columns=0), cpu_times], axis=1)
        .sort_values(by="time")
        .reset_index(drop=True)
    )
    file_totals = (
        log_df.loc[
            log_df["event"].str.contains("file")
            & log_df["duration_str"].str.contains("total")
        ]
        .copy()
        .reset_index(drop=True)
    )
    file_totals = file_totals.drop(columns=["duration_str", "volume_str"])
    file_info = pd.read_csv(
        f"s3_fuse/benchmark_settings/{benchmark_name}_fileinfo.csv",
        index_col=0,
    )
    path_series = file_info["filename"].map(lambda p: Path(p).name)
    file_total_records = []
    for _, total in file_totals.iterrows():
        record = total[["path", "duration", "volume"]].to_dict()
        match_info = file_info.loc[path_series == total["path"]]
        extension = match_info[str(test_case["hdu_ix"])]
        record["file_size"] = int(match_info["filesize"].iloc[0]) / 1000**2
        record["file_size_ratio"] = record["volume"] / record["file_size"]
        record["hdu_size"] = int(extension["datasize"]) / 1000**2
        record["hdu_size_ratio"] = record["volume"] / record["hdu_size"]
        record["n_cuts"] = test_case["count"]
        record["size_per_cut"] = (
            int(extension["itemsize"])
            / 8
            / 1000**2
            * reduce(mul, test_case["shape"])
        )
        record["cut_size"] = record["size_per_cut"] * record["n_cuts"]
        record["cut_size_ratio"] = record["volume"] / record["cut_size"]
        record["mb_rate"] = record["volume"] / record["duration"]
        record["cut_rate"] = record["n_cuts"] / record["duration"]
        record["retained_mb_rate"] = record["cut_size"] / record["duration"]
        record["cpu_busy"] = total.loc[
            total.index.str.startswith("cpu")
            & ~total.index.str.contains("idle")
        ].sum()
        record["cpu_idle"] = total["cpu_idle"]
        record["cpu_busy_ratio"] = record["cpu_busy"] / record["cpu_idle"]
        file_total_records.append(record)
    return pd.DataFrame(file_total_records)


def execute_test_case(
    test_case: Mapping,
    previous_case: Mapping,
    remount: Callable,
    n_throwaways: int
):
    """
    handler/wrapper function for executing a a particular benchmark test case
    """
    case_parts = test_case["title"].split("-")
    name, loader, count, shape = case_parts[0:4]
    if ("n_files" in test_case) and (test_case["n_files"] is not None):
        filecount = min(test_case["n_files"], len(test_case["paths"]))
    else:
        filecount = len(test_case["paths"])
    # check criteria for running throwaway tests, and execute
    # them if met -- if n_throwaways < 1, this will never
    # end up executing a throwaway test
    if (
        test_case["shape"] != previous_case.get("shape")
        or test_case["count"] != previous_case.get("count")
        or (
            (test_case["throttle"] is None)
            and (previous_case.get("throttle") is not None)
        )
    ):
        throwaway_results = run_throwaway_tests(
            loader, n_throwaways, remount, test_case
        )
    else:
        throwaway_results = []
    if "s3" not in loader:
        remount(test_case["bucket"])
    title_line = (
        f"testing {name} with {loader}, "
        f"{count} cut(s) of shape {shape} from {filecount} files"
    )
    if test_case["throttle"] is not None:
        title_line += f" {int(test_case['throttle']) / 1000} Mbps cap"
    print(f"******{title_line}******")
    # note that Throttle harmlessly does nothing at all if throttle is None
    with Throttle(test_case["throttle"]):
        cuts, stat, log = benchmark_cuts(**test_case)
    return cuts, stat, log, throwaway_results


def run_throwaway_tests(loader, n_throwaways, remount, test_case):
    throwaway_results = []
    for i in range(n_throwaways):
        print(
            f"...running throwaway test {i + 1}/{n_throwaways} for new "
            f"cutout setting..."
        )
        if "s3" not in loader:
            remount(test_case["bucket"])
        watch = Stopwatch(silent=True)
        watch.start()
        benchmark_cuts(**test_case)
        result = watch.peek()
        print(f"{result}s")
        throwaway_results.append(result)
    return throwaway_results
