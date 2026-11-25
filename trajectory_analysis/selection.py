"""A small atom-selection language.

Grammar (case-insensitive keywords, left to right with the usual precedence
``not`` > ``and`` > ``or``, parentheses allowed)::

    all | none
    element O H        (alias: elem, symbol)
    resname SOL WAT    (alias: res)
    name OW HW1
    resid 1 5-10
    index 0-99 120     (0-based atom indices)
    water | protein | hydrogen | heavy
    O                  (a bare element symbol)

Values of one keyword are OR-ed: ``element O N`` selects oxygens and nitrogens.
"""

import re

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a host dependency
    np = None

from . import elements

_KEYWORDS = {
    "element": "element",
    "elem": "element",
    "symbol": "element",
    "resname": "resname",
    "res": "resname",
    "name": "name",
    "resid": "resid",
    "index": "index",
}
_OPERATORS = {"and", "or", "not", "(", ")"}
_SINGLE = {"all", "none", "water", "protein", "hydrogen", "heavy"}
_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
_RANGE_RE = re.compile(r"^(-?\d+)(?:(?:-|:|\.\.)(-?\d+))?$")


class SelectionError(ValueError):
    """The selection text cannot be parsed or selects nothing it should."""


def _tokens(text):
    return _TOKEN_RE.findall(text or "")


def _ranges(values, keyword):
    out = []
    for value in values:
        match = _RANGE_RE.match(value)
        if not match:
            raise SelectionError(
                f"{keyword} expects numbers or ranges like 1-10, not {value!r}"
            )
        lo = int(match.group(1))
        hi = int(match.group(2)) if match.group(2) is not None else lo
        if hi < lo:
            lo, hi = hi, lo
        out.append((lo, hi))
    return out


class _Parser:
    def __init__(self, tokens, topology):
        self.tokens = tokens
        self.pos = 0
        self.top = topology
        self.n = topology.n_atoms

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def parse(self):
        if not self.tokens:
            raise SelectionError("the selection is empty")
        mask = self.parse_or()
        if self.peek() is not None:
            raise SelectionError(f"unexpected {self.peek()!r}")
        return mask

    def parse_or(self):
        mask = self.parse_and()
        while self.peek() is not None and self.peek().lower() == "or":
            self.take()
            mask = mask | self.parse_and()
        return mask

    def parse_and(self):
        mask = self.parse_not()
        while self.peek() is not None and self.peek().lower() == "and":
            self.take()
            mask = mask & self.parse_not()
        return mask

    def parse_not(self):
        if self.peek() is not None and self.peek().lower() == "not":
            self.take()
            return ~self.parse_not()
        return self.parse_atom()

    def _values(self):
        values = []
        while self.peek() is not None:
            tok = self.peek()
            low = tok.lower()
            if low in _OPERATORS or low in _KEYWORDS or low in _SINGLE:
                break
            values.append(self.take())
        return values

    def parse_atom(self):
        tok = self.take()
        if tok is None:
            raise SelectionError("the selection ends too early")
        low = tok.lower()
        if tok == "(":
            mask = self.parse_or()
            if self.take() != ")":
                raise SelectionError("a parenthesis is not closed")
            return mask
        if low in _SINGLE:
            return self._single(low)
        if low in _KEYWORDS:
            key = _KEYWORDS[low]
            values = self._values()
            if not values:
                raise SelectionError(f"{tok} needs at least one value")
            return self._keyword(key, values)
        if elements.atomic_number(tok) and not tok.isdigit():
            return self._keyword("element", [tok])
        raise SelectionError(f"unknown selection word {tok!r}")

    def _single(self, word):
        top = self.top
        if word == "all":
            return np.ones(self.n, dtype=bool)
        if word == "none":
            return np.zeros(self.n, dtype=bool)
        if word == "water":
            return np.array(
                [r.upper() in elements.WATER_RESIDUES for r in top.resnames], dtype=bool
            )
        if word == "protein":
            return np.array(
                [r.upper() in elements.PROTEIN_RESIDUES for r in top.resnames],
                dtype=bool,
            )
        if word == "hydrogen":
            return np.array([s == "H" for s in top.symbols], dtype=bool)
        return np.array([s != "H" for s in top.symbols], dtype=bool)  # heavy

    def _keyword(self, key, values):
        top = self.top
        if key == "element":
            wanted = {elements.normalize_symbol(v) for v in values}
            return np.array([s in wanted for s in top.symbols], dtype=bool)
        if key == "resname":
            wanted = {v.upper() for v in values}
            return np.array([r.upper() in wanted for r in top.resnames], dtype=bool)
        if key == "name":
            wanted = {v.upper() for v in values}
            return np.array([n.upper() in wanted for n in top.names], dtype=bool)
        spans = _ranges(values, key)
        if key == "resid":
            ids = np.asarray(top.resids, dtype=np.int64)
        else:
            ids = np.arange(self.n)
        mask = np.zeros(self.n, dtype=bool)
        for lo, hi in spans:
            mask |= (ids >= lo) & (ids <= hi)
        return mask


def select(text, topology):
    """Boolean mask [N] of the atoms the selection text picks."""
    return _Parser(_tokens(text), topology).parse()


def select_indices(text, topology, require=True):
    """Sorted atom indices; raises SelectionError when `require` and none match."""
    idx = np.flatnonzero(select(text, topology))
    if require and len(idx) == 0:
        raise SelectionError(f"the selection {text!r} matches no atoms")
    return idx
