"""Pull the Verilog design out of a model's reply."""
import re

_BEGIN_DONE = re.compile(r"\[BEGIN\](.*?)\[DONE\]", re.S)
_CODE_BEGIN_END = re.compile(r"CODE BEGIN(.*?)CODE END", re.S)
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_MODULE = re.compile(r"^[ \t]*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b.*?\bendmodule\b", re.S | re.M)
_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def _strip_comments(code: str) -> str:
    return _COMMENTS.sub("", code)


def extract_code(reply: str) -> str:
    """Return the Verilog in a reply, or "" if there is none.

    Looks inside [BEGIN]...[DONE], then CODE BEGIN...CODE END, then ``` fences,
    and falls back to the text from the first `module` line to the last
    `endmodule`. If the same module is written twice (a draft, then a fix),
    only the last copy is kept, so the result still compiles.
    """
    text = None
    for pattern in (_BEGIN_DONE, _CODE_BEGIN_END, _FENCE):
        blocks = [b for b in pattern.findall(reply) if "endmodule" in b]
        if blocks:
            text = "\n".join(blocks)
            break
    if text is None:
        start = re.search(r"^[ \t]*module\s", reply, re.M)
        end = reply.rfind("endmodule")
        if not start or end < start.start():
            return ""
        text = reply[start.start():end + len("endmodule")]

    # a model sometimes puts ``` lines inside [BEGIN]...[DONE]; drop them
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("```"))

    clean = _strip_comments(text)
    names = [m.group(1) for m in _MODULE.finditer(clean)]
    if not names:
        return ""
    if len(names) == len(set(names)):
        return text.strip()

    # some module appears twice: keep compiler directives and the last copy of each module
    directives = [l for l in clean.splitlines() if l.lstrip().startswith("`")]
    last = {}
    for m in _MODULE.finditer(clean):
        last[m.group(1)] = m.group(0)
    return "\n".join(directives + list(last.values())).strip()
