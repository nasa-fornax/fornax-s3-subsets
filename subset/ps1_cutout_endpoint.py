"""
run a noninteractive version of the PS1 cutout example. used by client nodes
in the scaling example.
"""
import os
from multiprocessing import cpu_count
from pathlib import Path
import pickle
import sys

import fire
import pandas as pd
from killscreen.utilities import filestamp

# hacky; can remove if we decide to add an install script or put this in the
# repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from subset.science.ps1_utils import PS1_CUT_CONSTANTS
from subset.utilz.mount_s3 import mount_bucket
from subset.science.handlers import bulk_skycut, get_corresponding_images
from subset.utilz.generic import make_loaders, parse_topline

# default settings'

BUCKET = "nishapur"
S3_ROOT = "/mnt/s3"

# default PS1 bands to consider (currently only g and z are staged.)
PS1_BANDS = ("g", "z")

# where do we dump output?
DUMP_PATH = Path(os.path.expanduser("~"), ".slice_test")

# select loader. options are "astropy", "fitsio", "greedy_astropy",
# "greedy_fitsio"
# NOTE: because all these files are RICE-compressed,
# there is unlikely to be much difference between astropy and greedy_astropy
# -- astropy does not support loading individual tiles from a a
# tile-compressed FITS file.
LOADER = tuple(
    make_loaders(
        "fitsio",
    ).items()
)[0]

# per-loader performance-tuning parameters. you don't need to mess with
# these if you don't care about fiddly performance stuff. chunksize: how
# many images shall we initialize at once? threads['image']: how many
# threads shall we init with in parallel? (None to disable.) threads['cut']:
# how many threads shall we cut with in parallel? (None to disable.) note
# that S3 handles parallel requests very well; on a smaller instance,
# you will run out of CPU or bandwidth before you exhaust its willingness to
# serve parallel requests.
TUNING = {
    "fitsio": {
        "chunksize": 40,
        "threads": {"image": cpu_count() * 4, "cut": cpu_count() * 4},
    },
    "greedy_fitsio": {
        "chunksize": 10,
        "threads": {"image": cpu_count() * 2, "cut": None},
    },
    "default": {
        "chunksize": 20,
        "threads": {"image": cpu_count() * 4, "cut": cpu_count() * 4},
    },
}


def make_ps1_slices(targets: list[dict]):
    ps1_stacks, _ = get_corresponding_images(targets)
    mount_bucket(mount_path=S3_ROOT, bucket=BUCKET)
    loader_name, loader = LOADER
    tuning_params = TUNING.get(loader_name, TUNING["default"])
    os.makedirs(DUMP_PATH, exist_ok=True)
    cuts, log = bulk_skycut(
        ps1_stacks,
        targets,
        loader=loader,
        return_cuts=True,
        data_root=f"{S3_ROOT}/ps1",
        bands=PS1_BANDS,
        verbose=1,
        **PS1_CUT_CONSTANTS,
        **tuning_params,
    )
    rate, weight = parse_topline(log)
    print(f"{rate} cutouts/s, {weight} MB / cutout")
    logframe = pd.DataFrame(
        [line.split(",") for line in log.values()], index=log.keys()
    )
    logframe.to_csv(Path(DUMP_PATH, f"{filestamp()}_log.csv"))
    with open(Path(DUMP_PATH, f"{filestamp()}_cuts.pkl"), "wb") as stream:
        pickle.dump(cuts, stream)


# tell fire to handle command line call
if __name__ == "__main__":
    fire.Fire(make_ps1_slices)
