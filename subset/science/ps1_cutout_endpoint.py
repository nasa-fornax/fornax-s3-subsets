"""
run a noninteractive version of the PS1 cutout example. used by client nodes
in the scaling example.
"""
import os
from multiprocessing import cpu_count
from pathlib import Path

import fire
import pandas as pd
from cytoolz.curried import get
from killscreen.utilities import filestamp

from subset.utilz.mount_s3 import mount_bucket
from subset.science.ps1_utils import get_ps1_cutouts
from subset.utilz.generic import make_loaders, parse_topline

# default settings'

BUCKET = 'nishapur'
S3_ROOT = '/mnt/s3'

# default cutout side length in degrees
CUTOUT_SIDE_LENGTH = 60 / 3600

# default PS1 bands to consider (currently only g and z are staged.)
PS1_BANDS = ("g", "z")

# where do we dump output?
DUMP_PATH = Path(os.path.expanduser("~"), '.slice_test')

# select loader. options are "astropy", "fitsio", "greedy_astropy",
# "greedy_fitsio"
# NOTE: because all the files this particular notebook is looking
# at are RICE-compressed, there is unlikely to be much difference
# between astropy and greedy_astropy -- astropy does not support
# loading individual tiles from a a tile-compressed FITS file.
LOADER = tuple(make_loaders("fitsio",).items())[0]

# per-loader default performance-tuning parameters
# chunksize: how many images shall we initialize at once?
# image_threads: how many threads shall we init with in parallel?
# cut_threads: how many threads shall we cut with in parallel?
# set thread params to None to disable parallelism.
TUNING = {
    "fitsio": {
        "chunksize": 40,
        "image_threads": cpu_count() * 6,
        "cut_threads": cpu_count() * 6
    },
    "greedy_fitsio": {
        "chunksize": 10,
        "image_threads": cpu_count() * 2,
        "cut_threads": None
    },
    "default": {
        "chunksize": 20,
        "image_threads": cpu_count() * 4,
        "cut_threads": cpu_count() * 4
    },
}


def make_ps1_slices(targets: list[dict]):
    ps1_stacks = set(map(get(['proj_cell', 'sky_cell']), targets))
    mount_bucket(mount_path=S3_ROOT, bucket=BUCKET)
    loader_name, loader = LOADER
    if loader_name in TUNING.keys():
        tuning_params = TUNING[loader_name]
    else:
        tuning_params = TUNING["default"]
    os.makedirs(DUMP_PATH, exist_ok=True)
    cuts, log = get_ps1_cutouts(
        ps1_stacks,
        loader,
        targets,
        CUTOUT_SIDE_LENGTH,
        f"{S3_ROOT}/ps1",
        PS1_BANDS,
        verbose=2,
        return_cuts=False,
        dump=True,
        dump_to=DUMP_PATH,
        **tuning_params
    )
    rate, weight = parse_topline(log)
    print(f"{rate} cutouts/s, {weight} MB / cutout")
    logframe = pd.DataFrame(
        [line.split(",") for line in log.values()], index=log.keys()
    )
    logframe.to_csv(Path(DUMP_PATH, f'{filestamp()}_log.csv'))


# tell fire to handle command line call
if __name__ == "__main__":
    fire.Fire(make_ps1_slices)
