"""
helper functions for s3-slice organization and benchmarking.
text i/o and stream handling, conversion, functional utilities, etc.
"""
import datetime as dt
import functools
from functools import partial
from inspect import getmodule
from io import BytesIO
from pathlib import Path
import time
from typing import MutableMapping, Any, Callable

from cytoolz import juxt
from dateutil import parser as dtp


def append_write(path, text):
    with open(path, "a+") as file:
        file.write(text)


def target_to_method(stream_target):
    if isinstance(stream_target, Path):
        return partial(append_write, stream_target)
    for method_name in ("write", "append", "call"):
        if hasattr(stream_target, method_name):
            return getattr(stream_target, method_name)


def make_stream_handler(targets):
    if targets is None:
        return None
    return juxt(tuple(map(target_to_method, targets)))


def console_stream_handlers(out_targets=None, err_targets=None):
    """
    create a pair of stdout and stderr handler functions to pass to a
    `sh` subprocess call.
    """
    out_actions = make_stream_handler(out_targets)
    err_actions = make_stream_handler(err_targets)

    def handle_out(message):
        out_actions(message)

    def handle_err(message):
        err_actions(message)

    return {"_out": handle_out, "_err": handle_err}


def record_and_yell(message: str, cache: MutableMapping, loud: bool = False):
    """
    place message into a cache object with a timestamp; optionally print it
    """
    if loud is True:
        print(message)
    cache[dt.datetime.now().isoformat()] = message


def mb(number):
    return number / 10 ** 6


def gb(number):
    return number / 10 ** 9


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


# the following two classes are vendored from gPhoton
class FakeStopwatch:
    """fake simple timer object"""

    def click(self):
        return


class Stopwatch(FakeStopwatch):
    """
    simple timer object
    """
    def __init__(self, digits=2, silent=False):
        self.digits = digits
        self.last_time = None
        self.start_time = None
        self.silent = silent

    def peek(self):
        return round(time.time() - self.last_time, self.digits)

    def start(self):
        if self.silent is False:
            print("starting timer")
        now = time.time()
        self.start_time = now
        self.last_time = now

    def click(self):
        if self.last_time is None:
            return self.start()
        if self.silent is False:
            print(f"{self.peek()} elapsed seconds, restarting timer")
        self.last_time = time.time()


class TimeSwitcher:
    """
    little object that tracks changing times
    """
    def __init__(self, start_time: str = None):
        if start_time is not None:
            self.times = [start_time]
        else:
            self.times = []

    def check_time(self, string):
        try:
            self.times.append(dtp.parse(string).isoformat())
            return True
        except dtp.ParserError:
            return False

    def __repr__(self):
        if len(self.times) > 0:
            return self.times[-1]
        return None

    def __str__(self):
        return self.__repr__()


