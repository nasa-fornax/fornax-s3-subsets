from functools import partial

from dustgoggles.func import zero
import sh


def throttle(download=None, upload=None, interface="ens5"):
    """
    throttle network speed using wondershaper. requires a wondershaper
    installation and execution by a user with sudoer privileges and no
    password (e.g., the default user on most stock EC2 images).

    this permissions hack is sloppy and only suitable on single-user systems
    that have no password for sudoers _anyway_. I do not particularly
    recommend editing this to add your password as plain text. it would be
    better to edit permissions for the parts of the system wondershaper
    touches if you wanted to extend this to a more general context.
    """
    if (download is None) and (upload is None):
        return
    arguments = ["-a", interface]
    if download is not None:
        arguments += ["-d", download]
    if upload is not None:
        arguments += ["-u", upload]
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
                sh.wondershaper("-c", "-a", interface)
        # for some reason, wondershaper always throws an error code
        # on clear operations, even when successful.
        except sh.ErrorReturnCode:
            pass


class Throttle:
    """ "
    apply network transfer caps using wondershaper. basically exists
    just to provide a context manager around throttle() and unthrottle() in
    order to permit clean application of caps to particular code sections.
    all caveats on throttle() and unthrottle() about crudeness and security
    and so on apply here as well.

    does nothing whatsoever if you pass None for both upload and download.
    """

    def __init__(
        self, download=None, upload=None, interface="ens5", verbose=False
    ):
        if (download is not None) or (upload is not None):
            self.throttle = partial(throttle, download, upload, interface)
            self.unthrottle = partial(unthrottle, interface)
            self.verbose = verbose
        else:
            self.throttle, self.unthrottle = partial(zero), partial(zero)
            self.verbose = False

    def __enter__(self):
        self.throttle()
        if self.verbose is True:
            print(f"throttled interface: {self.throttle.args}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unthrottle()
        if self.verbose is True:
            print(f"unthrottled interface: {self.unthrottle.args}")
