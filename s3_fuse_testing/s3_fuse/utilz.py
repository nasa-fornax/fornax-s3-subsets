"""
stream handling, generic utilities, etc.
"""
import functools
import os
import random
import re
import shutil
from functools import partial
from inspect import getfullargspec
from io import BytesIO
from pathlib import Path
from typing import Callable

import sh


def preload_target(function, path, *args, **kwargs):
    """
    `function` should be able to accept both streams and paths.
    `preload_target` fully loads path into an in-memory object
    before passing it to `function`, hopefully preempting any clever
    lazy / random-read / memmap / etc. behavior on the part of function,
    OS-level filesystem handlers, etc.
    """
    buffer = BytesIO()
    with open(path, 'rb') as file:
        buffer.write(file.read())
    buffer.seek(0)
    return function(buffer, *args, **kwargs)


def preload_from_shm(
    function, source, *args, shm_path="/dev/shm/slicetemp", **kwargs
):
    """
    preloader for functions that will not accept buffers -- thin wrappers
    to C extensions like fitsio.FITS, for instance. will only work on Linux
    and may fail in some Linux environments depending on security settings.
    """
    if Path(source).parent != Path(shm_path):
        os.makedirs('/dev/shm/slicetemp', exist_ok=True)
        shutil.copy(source, '/dev/shm/slicetemp')
    return function(
        Path(shm_path, Path(source).name), *args, **kwargs
    )


def strip_irrelevant_kwargs(func, *args, **kwargs):
    """
    call a function with subsets of kwargs until it agrees to be called.
    quick hacky way to enable a consistent interface for callables whose real
    signatures cannot be inspected because they live inside extensions or
    whatever.
    """
    while len(kwargs) > 0:
        try:
            return func(*args, **kwargs)
        except TypeError as error:
            if "unexpected keyword" not in str(error):
                raise error
            bad_kwarg = str(error).split("'")[1]
            kwargs.pop(bad_kwarg)
    return func(*args)


def sample_table(table, k=None):
    """
    select a sample of size k from rows of a pyarrow Table.
    """
    if k is None:
        return table
    return table.take(random.sample(range(len(table)), k=k))


def make_loaders(*loader_names: str) -> dict[str, Callable]:
    """
    produce a mapping from FITS-loader names to callable load methods.
    """
    loaders = {}
    for name in loader_names:
        if "astropy" in name:
            import astropy.io.fits
            loaders[name] = astropy.io.fits.open
            if "greedy" in name:
                # "greedily" load files into memory before doing anything with
                # them. a useful bench reference.
                loaders[name] = partial(preload_target, loaders[name])
        elif "fitsio" in name:
            import fitsio
            loaders[name] = fitsio.FITS
            if "greedy" in name:
                # fitsio is a thinnish wrapper for cfitsio and cannot be
                # transparently initialized from python buffer objects.
                # this will probably fail on non-Linux systems and may fail
                # in some Linux environments, depending on settings.
                loaders[name] = partial(preload_from_shm, loaders[name])
    return loaders


def cleanup_greedy_shm(loader):
    func = loader.func if isinstance(loader, functools.partial) else loader
    spec = getfullargspec(func)
    if spec.kwonlydefaults is None:
        return
    if 'shm_path' in spec.kwonlydefaults:
        shutil.rmtree(spec.kwonlydefaults['shm_path'], True)


def parse_topline(log):
    """parse top line of output log from skybox-slicing examples."""
    total = next(reversed(log.values()))
    summary, _, duration, volume = total.split(",")
    cut_count = int(re.search(r"\d+", summary).group())
    seconds = float(re.search(r"\d+\.?\d+", duration).group())
    megabytes = float(re.search(r"\d+\.?\d+", volume).group())
    rate = cut_count / seconds
    weight = megabytes / cut_count
    return round(rate, 2), round(weight, 2)


def s3_url(bucket: str, path: str) -> str:
    """
    make a conventional S3 URL from a bucket name and a 'path'
    (prefixes + object name) to an object in that bucket
    """
    return f"s3://{bucket}/{path}"


def throttle(download=None, upload=None, interface="ens5"):
    """
    throttle network speed using wondershaper. requires a wondershaper
    installation and execution by a user with sudoer privileges and no
    password.
    this permissions hack is sloppy and only suitable on single-user systems
    that have no password for sudoers _anyway_. I do not particularly
    recommend editing this to add your password as plain text.
    it would be better to edit permissions for the parts of the
    system wondershaper touches if you wanted to use this in a
    more general context.
    """
    if (download is None) and (upload is None):
        return
    arguments = ['-a', interface]
    if download is not None:
        arguments += ['-d', download]
    if upload is not None:
        arguments += ['-u', upload]
    with sh.contrib.sudo(password="", _with=True):
        sh.wondershaper(*arguments)


def unthrottle(interface="ens5"):
    """
    unthrottle a network interface previously throttled by wondershaper.
    requires a wondershaper installation and execution by a user with
    sudoer privileges and no password. that permissions hack is sloppy;
    see notes on throttle().
    """
    with sh.contrib.sudo(password="", _with=True):
        try:
            # sloppy! see comments on throttle.
            with sh.contrib.sudo(password="", _with=True):
                sh.wondershaper('-c', '-a', interface)
        # for some reason, wondershaper always throws an error code
        # on clear operations, even when successful.
        except sh.ErrorReturnCode:
            pass


def load_first_aws_credential(cred_file=None):
    """
    load whatever aws credential is placed first in a
    conventionally-formatted aws credentials file,
    by default ~/.aws/credentials. return a dict whose key names
    match corresponding fsspec initialization kwargs.
    """
    if cred_file is None:
        cred_file = Path(os.path.expanduser('~'), ".aws", "credentials")
    with open(cred_file) as stream:
        creds = {}
        for line in stream.readlines():
            if 'key_id' in line.lower():
                creds['key'] = line.split("=")[1].strip()
            elif "secret_access" in line.lower():
                creds['secret'] = line.split("=")[1].strip()
            if len(creds) == 2:
                break
    return creds
