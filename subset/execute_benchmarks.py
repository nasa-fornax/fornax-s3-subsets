from functools import partial
import os
from pathlib import Path

from killscreen.utilities import roundstring

# hacky; can remove if we decide to add an install script or put this in the repo root
os.chdir(globals()['_dh'][0].parent)

from subset.benchmark.bench_utils import (
    check_existing_benchmarks,
    dump_bandwidth_allowance_metrics,
    dump_throwaway_results
)
from subset.benchmark.handlers import (
    execute_test_case, interpret_benchmark_instructions, process_bench_stats
)
from subset.utilz.generic import summarize_stat
from subset.utilz.mount_s3 import mount_bucket

SETTINGS = {
    # directory goofys will use as a mount point for buckets
    "mountpoint": "/home/ec2-user/s3/",
    # max files: tests will use the top n_files elements of the TEST_FILES
    # attribute of each benchmark module (unless less than n_files are given
    # in TEST_FILES)
    "n_files": 25,
    # do we actually want to look at the cuts? probably not in big bulk
    # benchmarks, but can be useful for diagnostics or if you want a
    # screensaver or whatever.
    "return_cuts": False,
    # rng seed for consistent execution w/out having to explicitly define
    # a very long list of rectangles
    "seed": 123456,
    # these values are given in kilobits per second. None means unthrottled.
    # if you don't pass this key, there won't be any throttling in any test.
    # "bandwidth": (None, 100 * 1000)
}
# these correspond to the names of submodules of the benchmark_settings module
BENCHMARK_NAMES = (
    "hst", "panstarrs", "galex_rice", "jwst_crf", "tesscut", "galex_gzip",
    "hst_big", "spitzer_irac", "spitzer_cosmos_irac"
)
# if there is an existing result file corresponding to a particular
# test case, shall we run it again (w/incrementing suffixes attached to
# outputs?)
DUPLICATE_BENCHMARKS = False
# where shall we write benchmark results?
METRIC_DIRECTORY = "subset/benchmark/bench_results"
Path(METRIC_DIRECTORY).mkdir(parents=True, exist_ok=True)
# how many throwaway tests should we run every time we switch the specific
# set of cuts we're using? this is intended to juice S3 serverside caching
# so that earlier-in-order cases aren't 'penalized' without us having to
# either make a separate copy of all S3 objects for every test case, select
# random objects from a larger corpus, or wait a long time (60s - 15m,
# ish) between each test case. this is spooky, because it's a black box --
# we have no way to seriously interrogate how this works. It can affect run
# times by a factor of 2-3 in many cases, though.
N_THROWAWAYS = 3
# "benchmarks" contains instructions for each test case
benchmarks = {
    name: interpret_benchmark_instructions(name, SETTINGS)
    for name in BENCHMARK_NAMES
}
# partially-evaluated S3 mount function, for convenience
remount = partial(mount_bucket, SETTINGS['mountpoint'], remount=True)

"""
execute all benchmarks in a loop and save results. you can mess with this cell
to pretty straightforwardly look at only certain categories of cases,
suppress results, etc.
"""
logs = {}
for benchmark_name, benchmark in benchmarks.items():
    previous = {}
    for case in benchmark:
        skip, suffix = check_existing_benchmarks(
            case['title'], DUPLICATE_BENCHMARKS, METRIC_DIRECTORY
        )
        if skip is True:
            continue
        cuts, stat, log, throwaways = execute_test_case(
            case, previous, remount, N_THROWAWAYS
        )
        print(roundstring(summarize_stat(stat)) + "\n")
        logs[f"{case['title']}"] = log
        process_bench_stats(log, case, benchmark_name).to_csv(
            Path(METRIC_DIRECTORY, f"{case['title']}_benchmark_{suffix}.csv"),
            index=None
        )
        dump_bandwidth_allowance_metrics(case['title'], suffix, METRIC_DIRECTORY)
        dump_throwaway_results(throwaways, case['title'], suffix, METRIC_DIRECTORY)
        previous = case
