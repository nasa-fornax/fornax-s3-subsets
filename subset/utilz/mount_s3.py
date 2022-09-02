from pathlib import Path
import time

import sh


def conditional_unmount(remount, mount_path):
    try:
        sh.mountpoint(mount_path)
        bucket_is_mounted = True
    except sh.ErrorReturnCode:
        bucket_is_mounted = False
    if (bucket_is_mounted is True) and (remount is False):
        return False
    elif bucket_is_mounted is True:
        sh.umount(mount_path)
    return True


def wait_for_file_output(path, string, delay=0.1):
    """
    this should probably read in reverse order instead.
    use with caution.
    """
    while True:
        try:
            stream = open(path)
            break
        except FileNotFoundError:
            time.sleep(delay)
    while not any(map(lambda line: string in line, stream.readlines())):
        time.sleep(delay)
        stream.seek(0)
    stream.close()


def wait_for_output(stream, string, delay=0.1):
    if isinstance(stream, (str, Path)):
        return wait_for_file_output(stream, string, delay)
    while not any(map(lambda line: string in line, stream)):
        time.sleep(delay)


def _mount_bucket_generic(backend, mount_path, bucket, stream_handlers):
    """
    backend must be executable from path
    and offer an interface similar to s3fs or goofyfs
    """
    mount_method = getattr(sh, backend)
    return mount_method(bucket, mount_path, **stream_handlers)


def _mount_bucket_with_goofys_in_debug_mode(
    mount_path, bucket, stream_handlers
):
    # note: some peculiarity in bufio.Scanner appears to make goofys
    # crash unpredictably and disastrously when attempting to print debug
    # output to python console / managed stdout streams. do not do that.
    # scratch to disk (like this function assumes).
    process = sh.goofys(
        "-f",
        "--debug_fuse",
        "--debug_s3",
        bucket,
        mount_path,
        **stream_handlers,
        _bg=True,
        _no_out=True,
        _no_err=True
    )
    # we expect that the stderr stream handler will be a string or Path
    # we can treat as a reference to a text file, or at least something that
    # returns a filelike object when we call `open` with it
    wait_for_file_output(stream_handlers["_err"], "successfully mounted")
    return process


def mount_bucket(
    mount_path,
    bucket,
    remount=False,
    stream_handlers=None,
    verbose=False,
    backend="goofys",
):
    if conditional_unmount(remount, mount_path) is False:
        return
    stream_handlers = {} if stream_handlers is None else stream_handlers
    if (backend == "goofys") and (verbose is True):
        return _mount_bucket_with_goofys_in_debug_mode(
            mount_path, bucket, stream_handlers
        )
    return _mount_bucket_generic(backend, mount_path, bucket, stream_handlers)
