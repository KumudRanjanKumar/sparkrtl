"""Find training problems that look too much like test problems.

Two checks, both based on n-grams (runs of n words in a row):
  spec overlap: the share of a test problem's 8-word runs that also appear in the training problem
  code overlap: Jaccard similarity of 10-token runs in the two reference solutions
"""
import re
from collections import defaultdict

BOILERPLATE = [
    "I would like you to implement a module named TopModule with the following interface.",
    "All input and output ports are one bit unless otherwise specified.",
    "Make sure your input and output interface has the same names as described in the question.",
    "Please start your Verilog code with CODE BEGIN and end with CODE END.",
]
_WORD = re.compile(r"[a-z0-9_]+")
_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_CODE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")


def _spec_grams(text, n=8):
    for b in BOILERPLATE:
        text = text.replace(b, " ")
    words = _WORD.findall(text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _code_grams(code, n=10):
    toks = _CODE_TOKEN.findall(_COMMENTS.sub(" ", code))
    toks = ["TopModule" if t == "RefModule" else t for t in toks]
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


class Index:
    """Holds the test problems; ask it how close a training problem is."""

    def __init__(self, tests):  # tests: list of (name, spec, ref_code)
        self.names = [t[0] for t in tests]
        self.spec = [_spec_grams(t[1]) for t in tests]
        self.code = [_code_grams(t[2]) for t in tests]
        self.inv_spec, self.inv_code = defaultdict(set), defaultdict(set)
        for i, g in enumerate(self.spec):
            for x in g:
                self.inv_spec[x].add(i)
        for i, g in enumerate(self.code):
            for x in g:
                self.inv_code[x].add(i)

    def closest(self, spec, code):
        """Return (spec_overlap, spec_match, code_jaccard, code_match)."""
        sg, cg = _spec_grams(spec), _code_grams(code)
        spec_hits, code_hits = defaultdict(int), defaultdict(int)
        for x in sg:
            for i in self.inv_spec.get(x, ()):
                spec_hits[i] += 1
        for x in cg:
            for i in self.inv_code.get(x, ()):
                code_hits[i] += 1
        s, si = max(((h / max(1, len(self.spec[i])), i) for i, h in spec_hits.items()),
                    default=(0.0, None))
        c, ci = max(((h / max(1, len(cg | self.code[i])), i) for i, h in code_hits.items()),
                    default=(0.0, None))
        name = lambda i: self.names[i] if i is not None else ""
        return s, name(si), c, name(ci)
