"""Trajectory readers: multi-frame PDB, GRO and XYZ, SDF conformers, RDKit molecules.

Text trajectories are memory-mapped and indexed by the byte range of every frame; a
frame is parsed only when the analysis asks for it, so files much larger than memory
can be analysed.  Fixed-width coordinates (PDB, GRO) are converted with numpy
character arithmetic.  The readers are adapted from the Large System Viewer plugin
(copied, not imported) and extended with per-frame energies, frame times and the
residue information the selections need.
"""

import json
import mmap
import os
import re
from dataclasses import dataclass, field

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a host dependency
    np = None

from . import elements

EXTENSIONS = (".pdb", ".ent", ".gro", ".xyz", ".sdf")
FILE_FILTER = "Trajectories (*.pdb *.ent *.gro *.xyz *.sdf);;All files (*)"
CELL_PROPERTY = "moleditpy_cell"
_CHUNK = 64 * 1024 * 1024


class FormatError(ValueError):
    """The file is not a trajectory this reader understands."""


def format_of(path):
    ext = os.path.splitext(path)[1].lower()
    return {".ent": "pdb"}.get(ext, ext.lstrip("."))


# ------------------------------------------------------------------------ topology


@dataclass
class Topology:
    """Per-atom information that stays fixed over the trajectory."""

    symbols: list
    names: list = field(default_factory=list)
    resnames: list = field(default_factory=list)
    resids: list = field(default_factory=list)

    def __post_init__(self):
        n = len(self.symbols)
        self.symbols = [elements.normalize_symbol(s) for s in self.symbols]
        if not self.names:
            self.names = list(self.symbols)
        if not self.resnames:
            self.resnames = [""] * n
        if not self.resids:
            self.resids = [0] * n
        if not (len(self.names) == len(self.resnames) == len(self.resids) == n):
            raise ValueError("topology fields have different lengths")

    @property
    def n_atoms(self):
        return len(self.symbols)

    @property
    def has_residues(self):
        return any(r for r in self.resnames)

    def masses(self):
        return np.array(
            [elements.mass(elements.atomic_number(s)) for s in self.symbols],
            dtype=np.float64,
        )

    def residue_keys(self):
        """One key per atom identifying its residue instance (resid changes split)."""
        keys = []
        current = -1
        previous = None
        for resname, resid in zip(self.resnames, self.resids):
            if (resname, resid) != previous:
                current += 1
                previous = (resname, resid)
            keys.append(current)
        return keys


# -------------------------------------------------------------------------- frames


class FrameSource:
    """Frames of one topology.

    read_frame(i) returns (float64 [N, 3] in Angstrom, box) where box is a 3x3
    matrix whose rows are the cell vectors in Angstrom, or None when there is no cell.
    energy(i) and time(i) return a float or None.
    """

    n_frames = 0
    n_atoms = 0
    default_box = None
    energy_label = "Energy (file units)"

    def read_frame(self, index):  # pragma: no cover - interface
        raise NotImplementedError

    def energy(self, index):
        return None

    def time(self, index):
        return None

    def close(self):
        pass


class ArrayFrames(FrameSource):
    def __init__(self, frames, boxes=None, default_box=None, energies=None, times=None):
        self.frames = np.asarray(frames, dtype=np.float64)
        if self.frames.ndim != 3 or self.frames.shape[2] != 3:
            raise FormatError("frames must have the shape [F, N, 3]")
        self.n_frames = int(self.frames.shape[0])
        self.n_atoms = int(self.frames.shape[1])
        self.boxes = boxes
        self.default_box = default_box
        if default_box is None and boxes:
            self.default_box = next((b for b in boxes if b is not None), None)
        self.energies = energies
        self.times = times

    def read_frame(self, index):
        box = self.boxes[index] if self.boxes else None
        return self.frames[index], box if box is not None else self.default_box

    def energy(self, index):
        return self.energies[index] if self.energies else None

    def time(self, index):
        return self.times[index] if self.times else None


class MappedFrames(FrameSource):
    def __init__(self, path, ranges, n_atoms, parse, default_box=None, meta=None):
        self._file = open(path, "rb")  # noqa: SIM115 - kept open for the mmap
        try:
            self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._file.close()
            raise
        self.ranges = ranges
        self.n_frames = len(ranges)
        self.n_atoms = n_atoms
        self._parse = parse
        self._meta = meta
        self._meta_cache = {}
        self.default_box = default_box

    def read_frame(self, index):
        start, end = self.ranges[index]
        coords, box = self._parse(self._mm[start:end])
        if len(coords) != self.n_atoms:
            raise FormatError(
                f"frame {index + 1} has {len(coords)} atoms, expected {self.n_atoms}"
            )
        return (
            np.asarray(coords, dtype=np.float64),
            box if box is not None else self.default_box,
        )

    def _frame_meta(self, index):
        if self._meta is None:
            return (None, None)
        if index not in self._meta_cache:
            start, end = self.ranges[index]
            # Metadata lives in the first lines of a frame; 4 kB is plenty.
            head = self._mm[start : min(end, start + 4096)]
            self._meta_cache[index] = self._meta(head)
        return self._meta_cache[index]

    def energy(self, index):
        return self._frame_meta(index)[0]

    def time(self, index):
        return self._frame_meta(index)[1]

    def close(self):
        try:
            self._mm.close()
        finally:
            self._file.close()


def _open_mmap(path):
    f = open(path, "rb")  # noqa: SIM115 - returned with the mmap
    try:
        if os.fstat(f.fileno()).st_size == 0:
            raise FormatError("the file is empty")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    except Exception:
        f.close()
        raise
    return f, mm


# ---------------------------------------------------------------- energy and time

_NUMBER = r"([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?)"
_ENERGY_RES = [
    # extended XYZ / ASE: energy=-1.23, Energy: -1.23, E_tot = ...
    re.compile(
        r"(?:^|[\s,;])(?:total[_ ]?energy|energy|etot|e_tot|epot|e)\s*[:=]\s*" + _NUMBER,
        re.IGNORECASE,
    ),
    # ORCA trajectories: "Coordinates from ORCA-job x E -76.32"; xtb "energy: -5.07"
    re.compile(r"(?:^|\s)(?:E|energy)\s+" + _NUMBER + r"(?:\s|$)"),
]
_BARE_NUMBER_RE = re.compile(r"^\s*" + _NUMBER + r"\s*$")
_TIME_RE = re.compile(r"(?:^|\s)(?:t|time)\s*=\s*" + _NUMBER, re.IGNORECASE)


def _to_float(text):
    try:
        return float(text.replace("d", "e").replace("D", "E"))
    except ValueError:
        return None


def parse_energy(comment):
    """Energy written on a comment/title line, or None."""
    if not comment:
        return None
    for regex in _ENERGY_RES:
        match = regex.search(comment)
        if match:
            return _to_float(match.group(1))
    match = _BARE_NUMBER_RE.match(comment)
    if match and ("." in match.group(1) or "e" in match.group(1).lower()):
        # CREST/conformer ensembles: the comment is just the energy.
        return _to_float(match.group(1))
    return None


def parse_time(comment):
    """Time from a 't= 10.0' style title (GROMACS trjconv), or None."""
    match = _TIME_RE.search(comment or "")
    return _to_float(match.group(1)) if match else None


# ---------------------------------------------------------------- fixed-width floats


def _line_starts(arr):
    nl = np.flatnonzero(arr == 10)
    starts = np.concatenate(([0], nl + 1))
    return starts[starts < len(arr)]


if np is not None:
    _DIGIT_VALUE = np.zeros(256, dtype=np.float64)
    _DIGIT_VALUE[48:58] = np.arange(10)
    _ALLOWED = np.zeros(256, dtype=bool)
    _ALLOWED[48:58] = True
    _ALLOWED[[32, 43, 45, 46]] = True  # space + - .


def parse_fixed(arr, starts, width, count):
    """Parse `count` fixed-width decimal fields beginning at each offset in `starts`."""
    m = len(starts)
    if m == 0:
        return np.empty((0, count), dtype=np.float64)
    span = width * count
    padded = np.concatenate([arr, np.full(span, 32, dtype=np.uint8)])
    chars = padded[np.asarray(starts)[:, None] + np.arange(span)]
    fields = chars.reshape(m * count, width)
    j = np.arange(width)

    dots = fields == 46
    first = int(dots[0].argmax())
    if dots[0, first] and dots[:, first].all() and int(dots.sum()) == len(fields):
        weights = np.where(j < first, 10.0 ** (first - j - 1), 10.0 ** (first - j))
        weights[first] = 0.0
        values = _DIGIT_VALUE[fields] @ weights
    else:
        has_dot = dots.any(axis=1)
        dot_pos = np.where(has_dot, dots.argmax(axis=1), width)[:, None]
        exponent = np.where(j < dot_pos, dot_pos - j - 1, dot_pos - j)
        values = (_DIGIT_VALUE[fields] * np.power(10.0, exponent)).sum(axis=1)
    negative = (fields == 45).any(axis=1)
    values[negative] *= -1.0
    values = values.reshape(m, count)

    bad_rows = np.flatnonzero(~_ALLOWED[chars].all(axis=1))
    for r in bad_rows.tolist():
        for k in range(count):
            raw = bytes(chars[r, k * width : (k + 1) * width])
            if b"\n" in raw or b"\r" in raw:
                raise FormatError(
                    "a coordinate line is shorter than the fixed-width fields"
                )
            text = raw.decode("latin-1").strip()
            try:
                values[r, k] = float(text)
            except ValueError as exc:
                raise FormatError(f"cannot read coordinate {text!r}") from exc
    return values


# -------------------------------------------------------------------------- cells


def cell_from_parameters(a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
    """Cell vectors (rows) from lengths and angles in degrees; a along x, b in xy."""
    al, be, ga = np.radians([alpha, beta, gamma])
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([va, vb, [cx, cy, cz]])


def valid_cell(box):
    if box is None:
        return None
    try:
        box = np.asarray(box, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if box.shape != (3, 3) or not np.isfinite(box).all():
        return None
    if abs(np.linalg.det(box)) < 1e-6:
        return None
    return box


def cell_from_property(text):
    """Cell matrix from the suite's `moleditpy_cell` JSON string, or None."""
    if not text:
        return None
    try:
        data = json.loads(text)
        lattice = data["lattice"]
    except (TypeError, ValueError, KeyError):
        return None
    return valid_cell(lattice)


# ---------------------------------------------------------------------------- PDB

_MODEL_RE = re.compile(rb"^MODEL", re.MULTILINE)
_END_RE = re.compile(rb"^END[ \t\r]*$", re.MULTILINE)
_ATOMLINE_RE = re.compile(rb"^(?:ATOM  |HETATM)", re.MULTILINE)
_CRYST1_RE = re.compile(rb"^CRYST1.*$", re.MULTILINE)
_REMARK_RE = re.compile(rb"^(?:REMARK|TITLE|HEADER).*$", re.MULTILINE)
if np is not None:
    _ATOM = np.frombuffer(b"ATOM", dtype=np.uint8)
    _HETATM = np.frombuffer(b"HETATM", dtype=np.uint8)


def _pdb_cell(data):
    match = _CRYST1_RE.search(data)
    if not match:
        return None
    text = match.group(0).decode("latin-1")
    try:
        params = [
            float(text[6:15]),
            float(text[15:24]),
            float(text[24:33]),
            float(text[33:40]),
            float(text[40:47]),
            float(text[47:54]),
        ]
    except ValueError:
        return None
    # 1 x 1 x 1 cells are placeholders written for structures without a lattice.
    if params[0] <= 1.0 and params[1] <= 1.0 and params[2] <= 1.0:
        return None
    return valid_cell(cell_from_parameters(*params))


def _pdb_atom_starts(arr):
    starts = _line_starts(arr)
    padded = np.concatenate([arr, np.zeros(8, dtype=np.uint8)])
    head = padded[starts[:, None] + np.arange(6)]
    is_atom = (head[:, :4] == _ATOM).all(axis=1)
    is_het = (head == _HETATM).all(axis=1)
    return starts[is_atom | is_het]


def parse_pdb_frame(block):
    arr = np.frombuffer(block, dtype=np.uint8)
    return parse_fixed(arr, _pdb_atom_starts(arr) + 30, 8, 3), _pdb_cell(block)


def _pdb_meta(head):
    energy = time = None
    for line in _REMARK_RE.findall(head):
        text = line.decode("latin-1")[6:]
        if energy is None:
            energy = parse_energy(text)
        if time is None:
            time = parse_time(text)
    return energy, time


def _pdb_frame_ranges(mm):
    size = len(mm)
    models = [m.start() for m in _MODEL_RE.finditer(mm)]
    if models:
        ranges = list(zip(models, models[1:] + [size]))
    else:
        ends = [m.end() for m in _END_RE.finditer(mm)]
        if len(ends) > 1:
            ranges = list(zip([0] + ends[:-1], ends))
        else:
            ranges = [(0, size)]
    return [r for r in ranges if _ATOMLINE_RE.search(mm, r[0], r[1])]


def _pdb_topology(block):
    symbols, names, resnames, resids = [], [], [], []
    for line in block.split(b"\n"):
        rec = line[:6]
        if rec != b"HETATM" and line[:4] != b"ATOM":
            continue
        text = line.decode("latin-1").rstrip("\r")
        name = text[12:16].ljust(4)
        resname = text[17:21].strip()
        element = text[76:78].strip()
        if not (element.isalpha() and elements.atomic_number(element)):
            element = elements.guess_pdb_element(name, resname, rec == b"HETATM")
        try:
            resids.append(int(text[22:26]))
        except ValueError:
            resids.append(0)
        symbols.append(element)
        names.append(name.strip())
        resnames.append(resname)
    return Topology(symbols, names, resnames, resids)


def open_pdb(path):
    f, mm = _open_mmap(path)
    try:
        ranges = _pdb_frame_ranges(mm)
        if not ranges:
            raise FormatError("no ATOM or HETATM records")
        first_start, first_end = ranges[0]
        block = mm[first_start:first_end]
        # The header before the first MODEL belongs to the first frame.
        header = mm[0:first_end]
        topology = _pdb_topology(block)
        coords, _ = parse_pdb_frame(block)
        default_box = _pdb_cell(header)
    finally:
        mm.close()
        f.close()
    if len(coords) != topology.n_atoms or topology.n_atoms == 0:
        raise FormatError("could not read the atom records")
    ranges[0] = (0, ranges[0][1])
    frames = MappedFrames(
        path, ranges, topology.n_atoms, parse_pdb_frame, default_box, _pdb_meta
    )
    return topology, frames


# -------------------------------------------------------------- line-periodic files


def _periodic_line_offsets(mm, period):
    """Byte ranges of every `period`-line block, for complete frames only."""
    size = len(mm)
    offsets = [0]
    line_no = 0
    for chunk_start in range(0, size, _CHUNK):
        block = mm[chunk_start : chunk_start + _CHUNK]
        nl = np.flatnonzero(np.frombuffer(block, dtype=np.uint8) == 10)
        if len(nl):
            line_index = line_no + 1 + np.arange(len(nl))
            hits = nl[(line_index % period) == 0] + chunk_start + 1
            offsets.extend(hits.tolist())
            line_no += len(nl)
    total_lines = line_no + (0 if mm[size - 1] == 10 else 1)
    n_frames = total_lines // period
    offsets = [o for o in offsets if o < size][:n_frames]
    ends = offsets[1:] + [size]
    return list(zip(offsets, ends))


def _first_lines(mm, count, start=0):
    lines = []
    pos = start
    size = len(mm)
    for _ in range(count):
        if pos >= size:
            break
        nl = mm.find(b"\n", pos)
        if nl < 0:
            nl = size
        lines.append(mm[pos:nl].decode("latin-1").rstrip("\r"))
        pos = nl + 1
    return lines, pos


def _nth_line(block, n):
    lines = block.split(b"\n", n + 1)
    return lines[n].decode("latin-1").rstrip("\r") if len(lines) > n else ""


# ---------------------------------------------------------------------------- GRO


def _gro_width(line):
    first = line.find(".", 20)
    second = line.find(".", first + 1)
    if first < 0 or second < 0:
        raise FormatError("cannot find the GRO coordinate columns")
    return second - first


def gro_cell(line):
    """Cell from a GRO box line (nm): v1x v2y v3z [v1y v1z v2x v2z v3x v3y]."""
    try:
        v = [float(x) for x in line.split()]
    except ValueError:
        return None
    if len(v) == 3:
        box = np.diag(v)
    elif len(v) >= 9:
        box = np.array([[v[0], v[3], v[4]], [v[5], v[1], v[6]], [v[7], v[8], v[2]]])
    else:
        return None
    return valid_cell(box * 10.0)


def _gro_parser(width, n_atoms):
    def parse(block):
        arr = np.frombuffer(block, dtype=np.uint8)
        all_starts = _line_starts(arr)
        starts = all_starts[2 : 2 + n_atoms]
        coords = parse_fixed(arr, starts + 20, width, 3) * 10.0
        box = None
        if len(all_starts) > 2 + n_atoms:
            box_start = int(all_starts[2 + n_atoms])
            box_end = block.find(b"\n", box_start)
            if box_end < 0:
                box_end = len(block)
            box = gro_cell(block[box_start:box_end].decode("latin-1"))
        return coords, box

    return parse


def _gro_meta(head):
    title = _nth_line(head, 0)
    return parse_energy(title), parse_time(title)


def open_gro(path):
    f, mm = _open_mmap(path)
    try:
        head, _ = _first_lines(mm, 3)
        try:
            n = int(head[1].split()[0])
        except (IndexError, ValueError) as exc:
            raise FormatError("line 2 of a GRO file must be the atom count") from exc
        if n <= 0:
            raise FormatError("the GRO file has no atoms")
        if len(head) < 3:
            raise FormatError("the GRO frame is incomplete")
        width = _gro_width(head[2])
        ranges = _periodic_line_offsets(mm, n + 3)
        if not ranges:
            raise FormatError("the GRO frame is incomplete")
        lines, _ = _first_lines(mm, n + 2)
        block = mm[ranges[0][0] : ranges[0][1]]
    finally:
        mm.close()
        f.close()
    parse = _gro_parser(width, n)
    _, box = parse(block)
    symbols, names, resnames, resids = [], [], [], []
    for line in lines[2 : 2 + n]:
        try:
            resids.append(int(line[0:5]))
        except ValueError:
            resids.append(0)
        resname = line[5:10].strip()
        name = line[10:15].strip()
        resnames.append(resname)
        names.append(name)
        symbols.append(elements.guess_name_element(name, resname))
    topology = Topology(symbols, names, resnames, resids)
    return topology, MappedFrames(path, ranges, n, parse, box, _gro_meta)


# ---------------------------------------------------------------------------- XYZ

_LATTICE_RE = re.compile(r'Lattice\s*=\s*"([^"]*)"', re.IGNORECASE)


def xyz_cell(comment):
    """Cell from an extended-XYZ comment (Lattice="ax ay az bx by bz cx cy cz")."""
    match = _LATTICE_RE.search(comment or "")
    if not match:
        return None
    try:
        v = [float(x) for x in match.group(1).split()]
    except ValueError:
        return None
    if len(v) != 9:
        return None
    return valid_cell(np.array(v).reshape(3, 3))


def _xyz_parser(n_atoms):
    def parse(block):
        lines = block.split(b"\n", n_atoms + 2)
        body = lines[2 : 2 + n_atoms]
        box = xyz_cell(lines[1].decode("latin-1")) if len(lines) > 1 else None
        tokens = b" ".join(body).split()
        if len(tokens) == 4 * n_atoms:
            table = np.array(tokens, dtype="S").reshape(n_atoms, 4)[:, 1:]
            return table.astype(np.float64), box
        rows = [line.split()[1:4] for line in body]
        try:
            return np.array(rows, dtype=np.float64), box
        except ValueError as exc:
            raise FormatError(f"cannot read the XYZ coordinates: {exc}") from exc

    return parse


def _xyz_meta(head):
    comment = _nth_line(head, 1)
    # Strip the Lattice="..." and Properties=... fields before looking for energy.
    stripped = _LATTICE_RE.sub(" ", comment)
    stripped = re.sub(r"Properties\s*=\s*\S+", " ", stripped, flags=re.IGNORECASE)
    return parse_energy(stripped), parse_time(stripped)


def _xyz_count(line):
    try:
        return int(line.split()[0])
    except (IndexError, ValueError) as exc:
        raise FormatError("line 1 of an XYZ file must be the atom count") from exc


def open_xyz(path):
    f, mm = _open_mmap(path)
    try:
        head, _ = _first_lines(mm, 1)
        n = _xyz_count(head[0])
        if n <= 0:
            raise FormatError("the XYZ file has no atoms")
        ranges = _periodic_line_offsets(mm, n + 2)
        valid = []
        for start, end in ranges:
            line, _ = _first_lines(mm, 1, start)
            try:
                if _xyz_count(line[0]) != n:
                    break
            except FormatError:
                break
            valid.append((start, end))
        if not valid:
            raise FormatError("the XYZ frame is incomplete")
        lines, _ = _first_lines(mm, n + 2)
        block = mm[valid[0][0] : valid[0][1]]
    finally:
        mm.close()
        f.close()
    parse = _xyz_parser(n)
    _, box = parse(block)
    symbols = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        symbols.append(elements.symbol(elements.atomic_number(parts[0])) if parts else "X")
    topology = Topology(symbols)
    return topology, MappedFrames(path, valid, n, parse, box, _xyz_meta)


# ---------------------------------------------------------------------------- SDF

_ENERGY_PROP_RE = re.compile(r"energy|^e$|^etot$|^e_tot$", re.IGNORECASE)


def _energy_from_props(mol):
    for name in mol.GetPropNames():
        if _ENERGY_PROP_RE.search(name):
            value = _to_float(mol.GetProp(name).strip().split()[0] or "x")
            if value is not None:
                return value
    return parse_energy(mol.GetProp("_Name")) if mol.HasProp("_Name") else None


def open_sdf(path):
    from rdkit import Chem

    supplier = Chem.ForwardSDMolSupplier(path, removeHs=False, sanitize=False)
    first = None
    frames, energies = [], []
    for mol in supplier:
        if mol is None:
            continue
        nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
        if first is None:
            first = mol
            first_nums = nums
        elif nums != first_nums:
            raise FormatError("the SDF records are different molecules, not frames")
        frames.append(mol.GetConformer().GetPositions())
        energies.append(_energy_from_props(mol))
    if first is None:
        raise FormatError("no readable SDF record")
    topology = topology_from_rdkit(first)
    if all(e is None for e in energies):
        energies = None
    return topology, ArrayFrames(np.stack(frames), energies=energies)


# --------------------------------------------------------------- RDKit molecules


def topology_from_rdkit(mol):
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    infos = [a.GetPDBResidueInfo() for a in mol.GetAtoms()]
    if symbols and all(i is not None for i in infos):
        return Topology(
            symbols,
            [i.GetName().strip() for i in infos],
            [i.GetResidueName().strip() for i in infos],
            [i.GetResidueNumber() for i in infos],
        )
    return Topology(symbols)


def from_rdkit(mol):
    """(Topology, frames) for a molecule; every conformer is one frame.

    The cell comes from the suite's `moleditpy_cell` property when it is set.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        raise FormatError("the molecule has no atoms")
    confs = list(mol.GetConformers())
    if not confs:
        raise FormatError("the molecule has no 3D coordinates")
    topology = topology_from_rdkit(mol)
    box = None
    if mol.HasProp(CELL_PROPERTY):
        box = cell_from_property(mol.GetProp(CELL_PROPERTY))
    energies = []
    for conf in confs:
        value = None
        for name in ("energy", "Energy", "E"):
            if conf.HasProp(name):
                value = _to_float(str(conf.GetProp(name)))
                break
        energies.append(value)
    if all(e is None for e in energies):
        energies = None
    frames = np.stack([c.GetPositions() for c in confs])
    return topology, ArrayFrames(frames, default_box=box, energies=energies)


# ----------------------------------------------------------------------- dispatch

_OPEN = {"pdb": open_pdb, "gro": open_gro, "xyz": open_xyz, "sdf": open_sdf}


def supported(path):
    return os.path.splitext(path)[1].lower() in EXTENSIONS


def open_trajectory(path):
    """(Topology, FrameSource) for a supported file."""
    fmt = format_of(path)
    if fmt not in _OPEN:
        raise FormatError(f"unsupported file type: .{fmt}")
    return _OPEN[fmt](path)
