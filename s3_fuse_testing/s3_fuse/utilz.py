"""
helper functions for s3-slice organization and benchmarking.
text i/o and stream handling, conversion, functional utilities, etc.
"""
import datetime as dt
import functools
from inspect import getmodule
from io import BytesIO
from typing import Any, Callable, MutableMapping

from cytoolz import first
from killscreen.utilities import mb


def record_and_yell(message: str, cache: MutableMapping, loud: bool = False):
    """
    place message into a cache object with a timestamp; optionally print it
    """
    if loud is True:
        print(message)
    cache[dt.datetime.now().isoformat()] = message


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


def crudely_find_library(obj: Any) -> str:
    if isinstance(obj, functools.partial):
        if len(obj.args) > 0:
            if isinstance(obj.args[0], Callable):
                return crudely_find_library(obj.args[0])
        return crudely_find_library(obj.func)
    return getmodule(obj).__name__.split(".")[0]


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


def print_stats(watch, stat):
    stat.update()
    print(
        f"{watch.peek()} s; "
        f"{mb(round(first(stat.interval.values())))} MB"
    )
    watch.click()
