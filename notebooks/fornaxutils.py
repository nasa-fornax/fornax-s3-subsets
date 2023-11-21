"""Helper functions and classes to support the `fornax-s3-subsets` project.

Author: Geert Barentsen
"""
import functools
import time

from astropy.table import Table
from fsspec.implementations.http import HTTPFile
from s3fs import S3File


class NetworkProfiler:
    """Profiling tool to monitor byte-range requests made by fsspec.

    This class monkey-patches the fsspec-provided `HTTPFile._fetch_range()`
    and `S3File._fetch_range()` methods to collect and display statistics
    on the byte-range GET requests executed by these objects.  This enables
    the exact network I/O behavior of tools using fsspec to be explored.

    Example
    -------
    >>> with NetworkProfiler() as profiler:
    >>>     with s3fs.open(s3_uri, mode="rb", block_size=50, cache_type="block") as fh:
    >>>         fh.seek(100)
    >>>         fh.read(80)
    >>>     profiler.summary()
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self._requests = []

    def reset(self):
        self._requests = []

    def __enter__(self):
        self.reset()

        if not hasattr(self, "_old_httpfile_method"):
            self._old_httpfile_method = HTTPFile._fetch_range
        HTTPFile._fetch_range = self.get_wrapper(HTTPFile._fetch_range)

        if not hasattr(self, "_old_s3file_method"):
            self._old_s3file_method = S3File._fetch_range
        S3File._fetch_range = self.get_wrapper(S3File._fetch_range)

        return self

    def __exit__(self, exc_type=None, exc_value=None, exc_tb=None):
        HTTPFile._fetch_range = self._old_httpfile_method
        S3File._fetch_range = self._old_s3file_method

    def print(self, msg, end=None, color="\u001b[1m\u001b[31;1m"):
        reset = "\u001b[0m"
        print(f"{color}{msg}{reset}", end=end)

    def summary(self):
        self.print(
            f"Summary: fetched {self.bytes_transferred()} "
            f"bytes ({self.bytes_transferred()/(1024*1024):.2f} MB) "
            f"in {self.time_elapsed():.2f} s ({self.throughput():.2f} MB/s) "
            f"using {self.requests()} requests."
        )

    def log(self, byte_start, byte_end, time_elapsed):
        """Register a byte-range requests."""
        self._requests.append((byte_start, byte_end, time_elapsed))

    def bytes_transferred(self) -> int:
        """Total number of bytes fetched."""
        return sum([req[1] - req[0] for req in self._requests])

    def time_elapsed(self) -> float:
        """Total time elapsed during network I/O in seconds."""
        return sum([req[2] for req in self._requests])

    def throughput(self) -> float:
        """Returns the throughput in MB/s."""
        return _compute_throughput(self.bytes_transferred(), self.time_elapsed())

    def requests(self) -> int:
        """Total number of GET byte-range requests made."""
        return len(self._requests)

    def get_wrapper(self, func):
        """Wrapper intended to monkey-patch the `_fetch_range` methods in fsspec."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            obj, start, end = args
            time_start = time.perf_counter()
            result = func(*args, **kwargs)
            time_elapsed = time.perf_counter() - time_start
            if self.verbose:
                if not self._requests:
                    self.print(f"Retrieving {obj.path}")
                throughput = _compute_throughput(end - start, time_elapsed)
                self.print(f"GET bytes {start}-{end} " f"({throughput:.2f} MB/s)")
            self.log(start, end, time_elapsed)
            return result

        return wrapper


def _compute_throughput(bytes_transferred, time_elapsed) -> float:
    if time_elapsed <= 0:
        return float("inf")
    return bytes_transferred / (time_elapsed * 1024 * 1024)


def byte_layout_table(hdulist) -> Table:
    """Returns the byte layout of a fits file as an AstroPy table."""
    info = [hdu.fileinfo() for hdu in hdulist]
    tbl = Table(info)
    tbl.add_column(range(len(tbl)), name="HDU", index=0)
    tbl["datEnd"] = tbl["datLoc"] + tbl["datSpan"]
    tbl["size"] = (tbl["datEnd"] - tbl["hdrLoc"]) / (1024 * 1024)
    tbl.keep_columns(["HDU", "hdrLoc", "datLoc", "datEnd", "size"])
    for col in ["hdrLoc", "datLoc", "datEnd"]:
        tbl[col].unit = "bytes"
    tbl["size"].unit = "MB"
    return tbl


def print_byte_layout(hdulist) -> str:
    """Prints the byte layout of a FITS file."""
    tbl = byte_layout_table(hdulist)
    result = ""
    result += "HDU       hdrLoc      dataLoc  size\n"
    result += "           bytes        bytes    MB\n"
    for row in tbl:
        result += f"{row['HDU']: 3d} {row['hdrLoc']: 12d} {row['datLoc']: 12d} {row['size']:>5.0f}\n"
    print(result)
