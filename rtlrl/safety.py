"""Reject designs that try to talk to the testbench instead of being hardware.

An allowlist, not a banlist: you cannot list every trick a model might find,
but you can list the few system functions real designs need.
"""
import re

ALLOWED = {
    "$clog2", "$signed", "$unsigned", "$bits", "$countones",
    "$onehot", "$onehot0", "$size", "$high", "$low", "$left", "$right",
}
_SYSTEM_NAME = re.compile(r"\$(\w*)")


def is_safe(code: str) -> bool:
    """True if the design uses only allowed $-functions and no `include."""
    if "`include" in code:
        return False
    return all("$" + m.group(1) in ALLOWED for m in _SYSTEM_NAME.finditer(code))
