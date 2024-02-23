from __future__ import annotations

from contextlib import contextmanager

from modules import shared

@contextmanager
def cn_allow_script_control():
    orig = False
    if "control_net_allow_script_control" in shared.opts.data:
        try:
            orig = shared.opts.data["control_net_allow_script_control"]
            shared.opts.data["control_net_allow_script_control"] = True
            yield
        finally:
            shared.opts.data["control_net_allow_script_control"] = orig
    else:
        yield
