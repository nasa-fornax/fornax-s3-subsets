"""utilities for converting `goofys` debug output into log records"""

import re
from typing import MutableMapping, Sequence

from s3_fuse.utilz import TimeSwitcher


def record_nonempty(
    entries: MutableMapping, 
    entry_lines: Sequence[str], 
    moment: TimeSwitcher
):
    """
    insert timestamped sequence into dictionary iff 
    the sequence has anything in it
    """
    if len(entry_lines) < 1:
        return
    entries[moment.times[-2]] = entry_lines
    

def split_log_times(
    log: Sequence[str], timestamp_slice: slice=slice(0,26)
) -> dict[str, list[str]]:
    """
    split a sequence of lines that are timestamped only occasionally 
    into distinct 'moments'. assumes the lines are chronologically
    ordered.
    """
    moment = TimeSwitcher()
    entries = {}
    entry_lines = []
    for line in log:
        if moment.check_time(line[timestamp_slice]) is True:
            record_nonempty(entries, entry_lines, moment)
            entry_lines = []
        entry_lines.append(line)
    record_nonempty(entries, entry_lines, moment)
    return entries


def parse_s3api_log_block(entry: Sequence[str]) -> dict[str, str]:
    """
    convert a goofys debug stream log block based on a 
    s3api header into a dict 'record'
    """
    parsed = {}
    for line in entry:
        if method_result := re.search(r"s3/(\w+|\d+) detail", line):
            parsed["method"] = method_result.group(1)
        if "content-length" in line:
            parsed["length"] = re.search(r"\d+", line).group()
        if "content-range" in line:
            parsed["range"] = re.search(r"\d+-\d+", line).group()
    return parsed


def make_simple_goofys_log(log: Sequence[str]):
    """
    make a simple list-of-records log from a sequence 
    of lines taken from `goofys` debug output 
    """
    interesting_words = (
        "Details", "Content-Length", "Content-Range", "readFromStream"
    )
    filtered_log = filter(
        lambda line: any((word in line for word in interesting_words)), log
    )
    moments = split_log_times(tuple(filtered_log))
    # note that s3 portions of goofys debug output are printed in 'blocks'
    # based on http request/response headers
    records = []
    for timestamp, entry in moments.items():
        parsed = {'time': timestamp}
        entry = [line.lower() for line in entry]
        call_pattern = r"(request|response)"
        if call_result := re.search(call_pattern, entry[0]):
            parsed["type"] = "s3api"
            parsed["call"] = call_result.group(0)
            parsed |= parse_s3api_log_block(entry)
        elif "readfromstream" in entry[0]:
            assert len(entry) == 1, "there shouldn't be extra lines here"
            parsed["type"] = "fuse"
            parsed["method"] = "readfromstream"
            readfromstream_syntax = r"readfromstream \d+ (.*?) \[(\d+)]"
            properties = re.search(readfromstream_syntax, entry[0])
            parsed["path"] = properties.group(1)
            parsed["length"] = properties.group(2)
        else:
            raise ValueError("haven't implemented parsing for this yet")
        records.append(parsed)
    return records


def make_handler_log_record(timestamp: str, log_line: str) -> dict[str, str]:
    """split a line from our benchmark handler log into a 'record'"""
    record = {'time': timestamp, 'type': "handler"}
    fields = log_line.split(',')
    record['method'] = fields[0]
    if len(fields) > 1:
        record['path'] = fields[1]
    if len(fields) > 2:
        record['cut'] = fields[2]
    return record


def assemble_cut_log(test_result):
    cuts, runtime, volume, handler_log, fuse_log = test_result
    if fuse_log is not None:
        goofys_records = make_simple_goofys_log(fuse_log.split("\n"))
    else:
        goofys_records = []
    handler_records = [
        make_handler_log_record(timestamp, line) 
        for timestamp, line in handler_log.items()
    ]
    return goofys_records + handler_records
