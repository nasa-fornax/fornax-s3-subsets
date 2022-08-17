"""
simple script for regenerating file size / format information as input
to derived benchmark statistics.
"""
BENCHMARKS_FOR_WHICH_TO_REGENERATE_FILEINFO = (
    "hst",
    "hst_big",
    "jwst_crf",
    "panstarrs",
    "galex_gzip",
    "galex_rice",
    "tesscut",
    "spitzer_irac",
    "spitzer_cosmos_irac"
)
S3_MOUNTPOINT = "/mnt/s3"

if __name__ == "__main__":
    from functools import partial
    from pathlib import Path

    import pandas as pd

    from s3_fuse.fits import fitsstat
    from s3_fuse.handlers import interpret_benchmark_instructions
    from s3_fuse.mount_s3 import mount_bucket

    benchmarks = {
        benchmark_name: interpret_benchmark_instructions(benchmark_name, {})
        for benchmark_name in BENCHMARKS_FOR_WHICH_TO_REGENERATE_FILEINFO
    }
    remount = partial(
        mount_bucket, "goofys", S3_MOUNTPOINT, remount=True
    )
    for benchmark_name in benchmarks.keys():
        paths = benchmarks[benchmark_name][0]['paths']
        remount(benchmarks[benchmark_name][0]['bucket'])
        file_info = {}
        for path in paths:
            file_info[path] = fitsstat(Path(S3_MOUNTPOINT, path))
        infoframes = []
        for filename, info in file_info.items():
            infoframe = pd.DataFrame(info)
            infoframe['filename'] = Path(filename).name
            infoframes.append(infoframe)
        file_info_df = pd.concat(infoframes)
        file_info_df.to_csv(
            f's3_fuse/benchmark_settings/{benchmark_name}_fileinfo.csv'
        )
