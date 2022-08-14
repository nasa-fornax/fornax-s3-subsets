"""
helper functions for benchmark execution, loading result files into useful
objects, etc.
"""
import csv
import re
from itertools import chain
from pathlib import Path
from typing import Sequence, Callable

import numpy as np
import pandas as pd
import sh


def interpret_bench_fn(fn):
    parts = Path(fn).stem[:-14].split("-")
    return {
        "dataset": parts[0],
        "loader": parts[1],
        "n_cuts": int(parts[2]),
        "dims": tuple(map(int, parts[3].split("_"))),
        "throttle": None if parts[4] == "None" else int(parts[4]),
        "bench_ix": int(Path(fn).stem[-3:]),
        "title": Path(fn).stem[:-14],
    }


def interpret_bench_title(title):
    """truncated version of previous function, for summary tables"""
    parts = title.split("-")
    return {
        "dataset": parts[0],
        "loader": parts[1],
        "n_cuts": int(parts[2]),
        "dims": tuple(map(int, parts[3].split("_"))),
        "throttle": None if parts[4] == "None" else int(parts[4]),
        "title": title,
    }


def read_benchmark_result(benchmark_result_path):
    props = interpret_bench_fn(benchmark_result_path)
    reader = csv.DictReader(benchmark_result_path.open())
    return tuple(map(lambda line: line | props, reader))


def load_benchmark_results(
    result_directory: str,
    summarizers: Sequence[Callable] = (np.sum, np.mean, np.std, np.size),
):
    benchmark_files = [
        p for p in Path(result_directory).iterdir() if "bench" in p.name
    ]
    full_result_df = pd.DataFrame(
        chain.from_iterable(map(read_benchmark_result, benchmark_files))
    )
    dtypes = {}
    for c in full_result_df.columns:
        if c in ("path", "title", "loader", "dataset"):
            dtypes[c] = "string"
        elif c in ("n_cuts", "bench_ix"):
            dtypes[c] = np.int32
        elif c in ("dims",):
            dtypes[c] = object
        else:
            dtypes[c] = np.float64
    full_result_df = full_result_df.astype(dtypes).copy()
    # splitting the independent variables is redundant w/title for
    # differentiation, but may be useful for slicing or whatever
    identifier_columns = [
        "dataset",
        "loader",
        "n_cuts",
        "dims",
        "throttle",
        "bench_ix",
    ]
    pivot_df = full_result_df.drop(columns=identifier_columns + ["path"])
    summary = pivot_df.pivot_table(index="title", aggfunc=summarizers)
    summary = pd.concat(
        [
            pd.DataFrame(summary.index.map(interpret_bench_title).to_list()),
            summary.reset_index(drop=True),
        ],
        axis=1,
    )
    # "flatten" column names that originated in the pivot table multiindex
    # so that numpy does not interpret combinations of regular and multiindex
    # columns as ragged ndarrays
    summary.columns = list(
        summary.columns.map(
            lambda col: "_".join(col) if isinstance(col, tuple) else col
        )
    )
    return full_result_df, summary


def check_existing_benchmarks(
    title: str, create_duplicates: bool, result_directory: str
):
    """
    look for the existence of existing result files corresponding to this
    test case. if any are found, skip this test case if duplicate result
    creation is turned off; otherwise, determine a distinguishing suffix
    for this one.
    """
    matches = filter(
        lambda p: (title in p.name) and (p.suffix == ".csv"),
        Path(result_directory).iterdir(),
    )
    matches = tuple(matches)
    if len(matches) == 0:
        return False, "000"
    if create_duplicates is False:
        return True, None
    suffixes = [int(p.stem[-3:]) for p in matches]
    return False, str(max(suffixes) + 1).zfill(3)


def find_local_test_paths(benchmark):
    for test_case in benchmark:
        if test_case["paths"][0].startswith("s3"):
            continue
        return test_case["paths"]


def summarize_stat(stat):
    """format topline summary of benchmark stat object"""
    duration, volume, cpu = stat(total=True, simple_cpu=True).split(",")
    idle, busy = map(float, re.findall(r"[\d\.]+", cpu))
    return (
        f"{duration}, {volume} transferred, ~"
        f"{round(busy / idle * 100, 1)}% CPU usage"
    )


def dump_bandwidth_allowance_metrics(title, suffix, result_directory):
    """
    dump allowance metrics. intended for execution after a test case to ensure
    that EC2 bandwidth throttling has not meaningfully affected the test.
    """
    allowances = "\n".join(
        [
            line.strip()
            for line in sh.ethtool("-S", "ens5").stdout.decode().splitlines()
            if "allowance" in line
        ]
    )
    with open(
        Path(result_directory, f"{title}_allowance_check_{suffix}.log"), "w"
    ) as stream:
        stream.write(allowances)


def dump_throwaway_results(throwaways, title, suffix, result_directory):
    """save throwaway results in distinct files, for later reference"""
    if len(throwaways) == 0:
        return
    with open(
        Path(result_directory, f"{title}_throwaways_{suffix}.csv"), "w"
    ) as stream:
        stream.write("\n".join([str(a) for a in throwaways]))
