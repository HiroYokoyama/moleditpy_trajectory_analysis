"""Trajectory analyses: RDF, MSD/diffusion, RMSD, RMSF, radius of gyration, density
profiles and per-frame energies.

Everything here is numpy only and independent of Qt, so it is tested headlessly and
run from a worker thread.  Every analysis takes the same arguments::

    run_<name>(topology, frames, params, progress=None, cancelled=None) -> Result

``params`` is a plain dict (see DEFAULTS); ``progress(done, total)`` is called after
every frame and ``cancelled()`` is polled, raising Cancelled when it returns True.
Lengths are Angstrom, times picoseconds, masses amu.
"""

from dataclasses import dataclass, field

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a host dependency
    np = None

from .formats import valid_cell
from .selection import select_indices

# 1 amu / Angstrom^3 in g / cm^3
AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660
# 1 Angstrom^2 / ps = 1e-4 cm^2 / s = 1e-8 m^2 / s
A2_PER_PS_TO_CM2_PER_S = 1e-4

ANALYSES = (
    ("rdf", "Radial distribution function g(r)"),
    ("msd", "Mean square displacement / diffusion"),
    ("rmsd", "RMSD against a reference frame"),
    ("rmsf", "RMSF per atom"),
    ("rg", "Radius of gyration"),
    ("density", "Density profile along an axis"),
    ("energy", "Per-frame energies from the file"),
)

DEFAULTS = {
    # frame range: 1-based, inclusive; last = 0 means the last frame
    "first": 1,
    "last": 0,
    "stride": 1,
    # time between consecutive frames of the file in ps; 0 = file times or frame number
    "dt": 0.0,
    # cell: "file" (file / molecule cell), "custom" (custom_cell lengths), "none"
    "cell_mode": "file",
    "custom_cell": [0.0, 0.0, 0.0, 90.0, 90.0, 90.0],
    # RDF
    "rdf_sel_a": "O",
    "rdf_sel_b": "O",
    "rdf_rmax": 10.0,
    "rdf_bins": 200,
    # MSD
    "msd_sel": "all",
    "msd_dims": "xyz",
    "msd_remove_com": True,
    "msd_fit_start": 0.1,
    "msd_fit_end": 0.5,
    # RMSD / RMSF
    "rmsd_sel": "heavy",
    "rmsd_fit_sel": "",
    "rmsd_align": True,
    "rmsd_ref": 1,
    "unwrap": True,
    # Rg
    "rg_sel": "all",
    "rg_mass_weighted": True,
    # density
    "dens_sel": "all",
    "dens_axis": "c",
    "dens_bins": 100,
    "dens_kind": "mass",
    # energy
    "energy_relative": False,
}


class Cancelled(Exception):
    """The user cancelled the analysis."""


class AnalysisError(ValueError):
    """The analysis cannot run with these inputs."""


@dataclass
class Series:
    label: str
    x: object
    y: object
    style: str = "line"  # "line", "dash" or "points"


@dataclass
class Result:
    title: str
    xlabel: str
    ylabel: str
    series: list
    columns: list
    rows: object  # 2D array-like, one row per data point
    summary: str = ""
    notes: list = field(default_factory=list)

    def csv_text(self):
        lines = []
        for note in (
            [self.title] + ([self.summary] if self.summary else []) + self.notes
        ):
            for part in str(note).splitlines():
                lines.append("# " + part)
        lines.append(",".join(_csv_cell(c) for c in self.columns))
        for row in self.rows:
            lines.append(",".join(_csv_cell(v) for v in row))
        return "\n".join(lines) + "\n"


def _csv_cell(value):
    if isinstance(value, float) or (np is not None and isinstance(value, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.8g}"
    text = str(value)
    if any(ch in text for ch in ',"\n'):
        text = '"' + text.replace('"', '""') + '"'
    return text


# ---------------------------------------------------------------- shared helpers


def merged_params(params):
    out = dict(DEFAULTS)
    out.update(params or {})
    return out


def frame_indices(n_frames, first=1, last=0, stride=1):
    """0-based frame indices for a 1-based inclusive range; last <= 0 means the end."""
    if n_frames <= 0:
        raise AnalysisError("the trajectory has no frames")
    first = max(1, int(first))
    last = n_frames if int(last) <= 0 else min(int(last), n_frames)
    stride = max(1, int(stride))
    if first > last:
        raise AnalysisError(
            f"the first frame ({first}) is after the last frame ({last})"
        )
    return list(range(first - 1, last, stride))


def time_axis(frames, indices, dt):
    """(times, label): dt*index when dt > 0, else file times, else frame numbers."""
    if dt and dt > 0:
        return np.asarray(indices, dtype=np.float64) * float(dt), "Time (ps)"
    times = [frames.time(i) for i in indices]
    if indices and all(t is not None for t in times):
        return np.asarray(times, dtype=np.float64), "Time (ps, from file)"
    return np.asarray(indices, dtype=np.float64) + 1.0, "Frame"


def frame_spacing(frames, indices, dt):
    """Time between consecutive analysed frames, or 1 frame when unknown."""
    if dt and dt > 0:
        step = indices[1] - indices[0] if len(indices) > 1 else 1
        return float(dt) * step, "ps"
    if len(indices) > 1:
        t0, t1 = frames.time(indices[0]), frames.time(indices[1])
        if t0 is not None and t1 is not None and t1 > t0:
            return float(t1 - t0), "ps"
    step = indices[1] - indices[0] if len(indices) > 1 else 1
    return float(step), "frame"


def resolve_cell(box, params):
    """The cell used for this frame according to params['cell_mode']."""
    mode = params.get("cell_mode", "file")
    if mode == "none":
        return None
    if mode == "custom":
        from .formats import cell_from_parameters

        a, b, c, al, be, ga = (float(v) for v in params.get("custom_cell", [0] * 6))
        if min(a, b, c) <= 0:
            raise AnalysisError("the custom cell needs three positive lengths")
        return valid_cell(cell_from_parameters(a, b, c, al, be, ga))
    return valid_cell(box)


def minimum_image(vectors, box):
    """Apply the minimum-image convention to difference vectors [..., 3].

    Exact for orthorhombic cells and for triclinic cells as long as distances stay
    below half of the smallest cell width (see max_minimum_image_distance).
    """
    if box is None:
        return vectors
    inv = np.linalg.inv(box)
    frac = vectors @ inv
    frac -= np.round(frac)
    return frac @ box


def cell_widths(box):
    """Perpendicular widths of the cell (distance between opposite faces)."""
    a, b, c = box
    vol = abs(np.linalg.det(box))
    return np.array(
        [
            vol / np.linalg.norm(np.cross(b, c)),
            vol / np.linalg.norm(np.cross(c, a)),
            vol / np.linalg.norm(np.cross(a, b)),
        ]
    )


def max_minimum_image_distance(box):
    return 0.5 * float(cell_widths(box).min())


def kabsch(mobile, target):
    """Rotation R minimising |(mobile - c_m) R - (target - c_t)|; returns (R, c_m, c_t)."""
    cm = mobile.mean(axis=0)
    ct = target.mean(axis=0)
    h = (mobile - cm).T @ (target - ct)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    corr = np.diag([1.0, 1.0, d])
    return u @ corr @ vt, cm, ct


def superpose(coords, fit_idx, reference_fit):
    """All of `coords` moved so that coords[fit_idx] best fit `reference_fit`."""
    rot, cm, ct = kabsch(coords[fit_idx], reference_fit)
    return (coords - cm) @ rot + ct


class Unwrapper:
    """Follow atoms across periodic boundaries between consecutive frames.

    Each displacement is taken as its minimum image, so atoms must move less than
    half a cell width between the frames analysed (keep the stride small).
    """

    def __init__(self):
        self.previous = None
        self.current = None

    def __call__(self, coords, box):
        if box is None:
            self.previous = self.current = coords
            return coords
        if self.current is None:
            self.previous = coords
            self.current = coords.copy()
            return self.current
        step = minimum_image(coords - self.previous, box)
        self.previous = coords
        self.current = self.current + step
        return self.current


def _iterate(frames, indices, params, progress, cancelled):
    """Yield (k, index, coords, box) for the analysed frames, honouring cancel."""
    total = len(indices)
    for k, index in enumerate(indices):
        if cancelled is not None and cancelled():
            raise Cancelled()
        coords, box = frames.read_frame(index)
        yield k, index, coords, resolve_cell(box, params)
        if progress is not None:
            progress(k + 1, total)


def _selection(text, topology, what):
    try:
        return select_indices(text, topology)
    except ValueError as exc:
        raise AnalysisError(f"{what}: {exc}") from exc


def _range_note(indices):
    first = indices[0] + 1
    last = indices[-1] + 1
    stride = indices[1] - indices[0] if len(indices) > 1 else 1
    return f"Frames {first}-{last}, stride {stride} ({len(indices)} frames analysed)"


# ----------------------------------------------------------------------------- RDF


def run_rdf(topology, frames, params, progress=None, cancelled=None):
    p = merged_params(params)
    indices = frame_indices(frames.n_frames, p["first"], p["last"], p["stride"])
    sel_a = _selection(p["rdf_sel_a"], topology, "Selection A")
    sel_b = _selection(p["rdf_sel_b"], topology, "Selection B")
    rmax = float(p["rdf_rmax"])
    nbins = int(p["rdf_bins"])
    if rmax <= 0 or nbins < 1:
        raise AnalysisError("r max and the number of bins must be positive")
    edges = np.linspace(0.0, rmax, nbins + 1)
    hist = np.zeros(nbins, dtype=np.float64)
    density_sum = 0.0  # sum over frames of distinct pairs per A atom per volume
    overlap = np.intersect1d(sel_a, sel_b)
    n_a, n_b = len(sel_a), len(sel_b)
    n_pairs = n_a * n_b - len(overlap)
    if n_pairs <= 0:
        raise AnalysisError("the selections contain no distinct atom pairs")
    # Pairwise blocks of at most ~4 million distances keep memory bounded.
    chunk = max(1, 4_000_000 // max(1, n_b))
    notes = []
    periodic = None
    for _k, _index, coords, box in _iterate(frames, indices, p, progress, cancelled):
        pa = coords[sel_a]
        pb = coords[sel_b]
        if box is not None:
            limit = max_minimum_image_distance(box)
            if rmax > limit + 1e-9:
                raise AnalysisError(
                    f"r max ({rmax:g} A) exceeds half the smallest cell width "
                    f"({limit:.3f} A); lower r max for the minimum-image convention"
                )
            volume = abs(float(np.linalg.det(box)))
            periodic = True if periodic is None else periodic
        else:
            span = coords.max(axis=0) - coords.min(axis=0)
            volume = float(np.prod(np.maximum(span, 1e-6)))
            periodic = False
        for start in range(0, n_a, chunk):
            block = pa[start : start + chunk]
            d = pb[None, :, :] - block[:, None, :]
            if box is not None:
                d = minimum_image(d, box)
            r = np.sqrt((d * d).sum(axis=-1))
            if len(overlap):
                same = sel_a[start : start + chunk, None] == sel_b[None, :]
                r = r[~same]
            else:
                r = r.ravel()
            counts, _ = np.histogram(r[r < rmax], bins=edges)
            hist += counts
        density_sum += n_pairs / (n_a * volume)
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    centres = 0.5 * (edges[1:] + edges[:-1])
    n_frames = len(indices)
    g = hist / (n_a * shell * density_sum) if density_sum > 0 else hist * 0.0
    coordination = np.cumsum(hist) / (n_a * n_frames)
    if not periodic:
        notes.append(
            "No cell: distances are not periodic and g(r) is normalised by the "
            "bounding-box volume of each frame, so its long-range limit is not 1."
        )
    notes.append(_range_note(indices))
    first_peak = ""
    if g.max() > 0:
        k = int(np.argmax(g))
        first_peak = f"Highest peak g = {g[k]:.3f} at r = {centres[k]:.3f} A. "
    summary = (
        f"g(r) between '{p['rdf_sel_a']}' ({n_a} atoms) and '{p['rdf_sel_b']}' "
        f"({n_b} atoms), {nbins} bins to {rmax:g} A. {first_peak}"
        f"Mean pair density {density_sum / n_frames:.5g} A^-3."
    )
    return Result(
        title="Radial distribution function",
        xlabel="r (A)",
        ylabel="g(r)",
        series=[Series("g(r)", centres, g)],
        columns=["r_A", "g_r", "coordination_number"],
        rows=np.column_stack([centres, g, coordination]),
        summary=summary,
        notes=notes,
    )


# ----------------------------------------------------------------------------- MSD


def msd_fft(positions):
    """MSD(lag) averaged over atoms and time origins for positions [T, N, D].

    Uses the FFT algorithm (Calandrini et al.), O(T log T) per coordinate.
    """
    positions = np.asarray(positions, dtype=np.float64)
    t = positions.shape[0]
    if t < 2:
        return np.zeros(t)
    sq = (positions * positions).sum(axis=2)  # [T, N]
    sq = np.concatenate([sq, np.zeros((1, sq.shape[1]))])
    q = 2.0 * sq[:t].sum(axis=0)
    s1 = np.empty((t, positions.shape[1]))
    for m in range(t):
        q = q - sq[m - 1] - sq[t - m]
        s1[m] = q / (t - m)
    # Autocorrelation through a zero-padded FFT, summed over dimensions.
    f = np.fft.rfft(positions, n=2 * t, axis=0)
    acf = np.fft.irfft((f * f.conj()).real, n=2 * t, axis=0)[:t].sum(axis=2)
    s2 = acf / (t - np.arange(t))[:, None]
    return (s1 - 2.0 * s2).mean(axis=1)


def run_msd(topology, frames, params, progress=None, cancelled=None):
    p = merged_params(params)
    indices = frame_indices(frames.n_frames, p["first"], p["last"], p["stride"])
    if len(indices) < 3:
        raise AnalysisError("MSD needs at least three frames")
    sel = _selection(p["msd_sel"], topology, "Selection")
    dims = {"xyz": [0, 1, 2], "xy": [0, 1], "xz": [0, 2], "yz": [1, 2]}
    dims.update({"x": [0], "y": [1], "z": [2]})
    axes = dims.get(str(p["msd_dims"]).lower())
    if axes is None:
        raise AnalysisError(f"unknown MSD dimensions {p['msd_dims']!r}")
    needed = len(indices) * len(sel) * 3 * 8
    if needed > 2 * 1024**3:
        raise AnalysisError(
            f"MSD of {len(sel)} atoms over {len(indices)} frames needs "
            f"{needed / 1024**3:.1f} GB; select fewer atoms or use a larger stride"
        )
    masses = topology.masses()[sel]
    unwrap = Unwrapper()
    traj = np.empty((len(indices), len(sel), 3))
    periodic = False
    for k, _index, coords, box in _iterate(frames, indices, p, progress, cancelled):
        periodic = periodic or box is not None
        traj[k] = unwrap(coords[sel], box)
    if p["msd_remove_com"] and len(sel) > 1:
        w = masses if masses.sum() > 0 else np.ones(len(sel))
        com = (traj * w[None, :, None]).sum(axis=1) / w.sum()
        traj -= (com - com[0])[:, None, :]
    msd = msd_fft(traj[:, :, axes])
    spacing, unit = frame_spacing(frames, indices, p["dt"])
    lag = np.arange(len(indices)) * spacing
    lo = round(float(p["msd_fit_start"]) * (len(indices) - 1))
    hi = round(float(p["msd_fit_end"]) * (len(indices) - 1))
    lo = max(1, min(lo, len(indices) - 2))
    hi = max(lo + 1, min(hi, len(indices) - 1))
    slope, intercept = np.polyfit(lag[lo : hi + 1], msd[lo : hi + 1], 1)
    ndim = len(axes)
    d_coef = slope / (2.0 * ndim)
    fit = slope * lag + intercept
    xlabel = "Lag time (ps)" if unit == "ps" else "Lag (frames)"
    notes = [_range_note(indices)]
    if periodic:
        notes.append(
            "Coordinates were unwrapped through the cell frame by frame; atoms "
            "must move less than half a cell width between analysed frames."
        )
    if unit == "ps":
        summary = (
            f"D = {d_coef:.6g} A^2/ps = {d_coef * A2_PER_PS_TO_CM2_PER_S * 1e5:.6g} "
            f"x 1e-5 cm^2/s ({ndim}D, '{p['msd_sel']}', {len(sel)} atoms), fitted "
            f"over lag {lag[lo]:.4g}-{lag[hi]:.4g} ps."
        )
    else:
        summary = (
            f"Slope {slope:.6g} A^2/frame, D = {d_coef:.6g} A^2/frame ({ndim}D). "
            "Set the time per frame to get D in physical units."
        )
        notes.append("No time per frame was given and the file carries no times.")
    return Result(
        title="Mean square displacement",
        xlabel=xlabel,
        ylabel="MSD (A^2)",
        series=[
            Series("MSD", lag, msd),
            Series(
                f"fit, D = {d_coef:.4g}", lag[lo : hi + 1], fit[lo : hi + 1], "dash"
            ),
        ],
        columns=["lag", "msd_A2", "fit_A2"],
        rows=np.column_stack([lag, msd, fit]),
        summary=summary,
        notes=notes,
    )


# ------------------------------------------------------------------- RMSD / RMSF


def _rmsd_setup(topology, frames, p):
    indices = frame_indices(frames.n_frames, p["first"], p["last"], p["stride"])
    sel = _selection(p["rmsd_sel"], topology, "Selection")
    fit_text = str(p.get("rmsd_fit_sel") or "").strip() or p["rmsd_sel"]
    fit = _selection(fit_text, topology, "Fit selection")
    ref_index = int(p["rmsd_ref"]) - 1
    if not 0 <= ref_index < frames.n_frames:
        raise AnalysisError(
            f"the reference frame {ref_index + 1} is outside 1-{frames.n_frames}"
        )
    if p["rmsd_align"] and len(fit) < 3:
        raise AnalysisError("alignment needs at least three atoms in the fit selection")
    ref_coords, _ = frames.read_frame(ref_index)
    return (
        indices,
        sel,
        fit,
        fit_text,
        ref_index,
        np.array(ref_coords, dtype=np.float64),
    )


def run_rmsd(topology, frames, params, progress=None, cancelled=None):
    p = merged_params(params)
    indices, sel, fit, fit_text, ref_index, ref = _rmsd_setup(topology, frames, p)
    unwrap = Unwrapper()
    values = np.empty(len(indices))
    for k, _index, coords, box in _iterate(frames, indices, p, progress, cancelled):
        if p["unwrap"]:
            coords = unwrap(coords, box)
        if p["rmsd_align"]:
            coords = superpose(coords, fit, ref[fit])
        diff = coords[sel] - ref[sel]
        values[k] = np.sqrt((diff * diff).sum(axis=1).mean())
    times, xlabel = time_axis(frames, indices, p["dt"])
    summary = (
        f"RMSD of '{p['rmsd_sel']}' ({len(sel)} atoms) against frame {ref_index + 1}"
        + (f", aligned on '{fit_text}' ({len(fit)} atoms)" if p["rmsd_align"] else "")
        + f". Mean {values.mean():.4f} A, max {values.max():.4f} A, "
        f"final {values[-1]:.4f} A."
    )
    return Result(
        title="RMSD",
        xlabel=xlabel,
        ylabel="RMSD (A)",
        series=[Series("RMSD", times, values)],
        columns=["frame", xlabel.split(" (")[0].lower().replace(" ", "_"), "rmsd_A"],
        rows=np.column_stack([np.asarray(indices) + 1, times, values]),
        summary=summary,
        notes=[_range_note(indices)],
    )


def run_rmsf(topology, frames, params, progress=None, cancelled=None):
    p = merged_params(params)
    indices, sel, fit, fit_text, ref_index, ref = _rmsd_setup(topology, frames, p)
    unwrap = Unwrapper()
    total = np.zeros((len(sel), 3))
    total_sq = np.zeros(len(sel))
    for _k, _index, coords, box in _iterate(frames, indices, p, progress, cancelled):
        if p["unwrap"]:
            coords = unwrap(coords, box)
        if p["rmsd_align"]:
            coords = superpose(coords, fit, ref[fit])
        x = coords[sel]
        total += x
        total_sq += (x * x).sum(axis=1)
    n = len(indices)
    mean = total / n
    rmsf = np.sqrt(np.maximum(total_sq / n - (mean * mean).sum(axis=1), 0.0))
    atom_numbers = sel + 1
    rows = [
        [
            int(i + 1),
            topology.symbols[i],
            topology.names[i],
            topology.resnames[i],
            int(topology.resids[i]),
            float(v),
        ]
        for i, v in zip(sel, rmsf)
    ]
    k = int(np.argmax(rmsf))
    summary = (
        f"RMSF of '{p['rmsd_sel']}' ({len(sel)} atoms) over {n} frames"
        + (
            f", aligned on '{fit_text}' to frame {ref_index + 1}"
            if p["rmsd_align"]
            else ""
        )
        + f". Mean {rmsf.mean():.4f} A; largest {rmsf[k]:.4f} A at atom "
        f"{sel[k] + 1} ({topology.names[sel[k]]})."
    )
    return Result(
        title="RMSF",
        xlabel="Atom number",
        ylabel="RMSF (A)",
        series=[
            Series("RMSF", atom_numbers, rmsf, "points" if len(sel) < 3 else "line")
        ],
        columns=["atom", "element", "name", "resname", "resid", "rmsf_A"],
        rows=rows,
        summary=summary,
        notes=[_range_note(indices)],
    )


# ------------------------------------------------------------------------------ Rg


def run_rg(topology, frames, params, progress=None, cancelled=None):
    p = merged_params(params)
    indices = frame_indices(frames.n_frames, p["first"], p["last"], p["stride"])
    sel = _selection(p["rg_sel"], topology, "Selection")
    w = topology.masses()[sel] if p["rg_mass_weighted"] else np.ones(len(sel))
    if w.sum() <= 0:
        w = np.ones(len(sel))
    unwrap = Unwrapper()
    values = np.empty(len(indices))
    for k, _index, coords, box in _iterate(frames, indices, p, progress, cancelled):
        x = coords[sel]
        if p["unwrap"]:
            x = unwrap(x, box)
        centre = (x * w[:, None]).sum(axis=0) / w.sum()
        d = x - centre
        values[k] = np.sqrt((w * (d * d).sum(axis=1)).sum() / w.sum())
    times, xlabel = time_axis(frames, indices, p["dt"])
    kind = "mass-weighted" if p["rg_mass_weighted"] else "geometric"
    summary = (
        f"{kind.capitalize()} radius of gyration of '{p['rg_sel']}' ({len(sel)} atoms): "
        f"mean {values.mean():.4f} A, std {values.std():.4f} A, "
        f"range {values.min():.4f}-{values.max():.4f} A."
    )
    return Result(
        title="Radius of gyration",
        xlabel=xlabel,
        ylabel="Rg (A)",
        series=[Series("Rg", times, values)],
        columns=["frame", xlabel.split(" (")[0].lower().replace(" ", "_"), "rg_A"],
        rows=np.column_stack([np.asarray(indices) + 1, times, values]),
        summary=summary,
        notes=[_range_note(indices)],
    )


# ------------------------------------------------------------------------- density


def run_density(topology, frames, params, progress=None, cancelled=None):
    p = merged_params(params)
    indices = frame_indices(frames.n_frames, p["first"], p["last"], p["stride"])
    sel = _selection(p["dens_sel"], topology, "Selection")
    nbins = int(p["dens_bins"])
    if nbins < 1:
        raise AnalysisError("the number of bins must be positive")
    axis_name = str(p["dens_axis"]).lower()
    axis = {"a": 0, "b": 1, "c": 2, "x": 0, "y": 1, "z": 2}.get(axis_name)
    if axis is None:
        raise AnalysisError(f"unknown axis {p['dens_axis']!r}")
    mass_kind = p["dens_kind"] == "mass"
    weights = topology.masses()[sel] if mass_kind else np.ones(len(sel))

    hist = np.zeros(nbins)
    height_sum = 0.0
    volume_sum = 0.0
    periodic = None
    lo = hi = None
    cached = []
    for _k, _index, coords, box in _iterate(frames, indices, p, progress, cancelled):
        x = coords[sel]
        if box is not None and periodic is not False:
            periodic = True
            frac = (x @ np.linalg.inv(box))[:, axis]
            frac -= np.floor(frac)
            counts, _ = np.histogram(
                frac, bins=nbins, range=(0.0, 1.0), weights=weights
            )
            hist += counts
            height_sum += cell_widths(box)[axis]
            volume_sum += abs(float(np.linalg.det(box)))
        else:
            if periodic:
                raise AnalysisError("some frames have a cell and others do not")
            periodic = False
            values = x[:, axis]
            cached.append(values)
            lo = values.min() if lo is None else min(lo, values.min())
            hi = values.max() if hi is None else max(hi, values.max())
    n = len(indices)
    notes = [_range_note(indices)]
    if periodic:
        height = height_sum / n
        volume = volume_sum / n
        edges = np.linspace(0.0, height, nbins + 1)
        bin_volume = volume / nbins
        density = hist / n / bin_volume
        axis_label = f"Position along the {'abc'[axis]} axis normal (A)"
        if mass_kind:
            density = density * AMU_PER_A3_TO_G_PER_CM3
            ylabel, column = "Mass density (g/cm^3)", "density_g_cm3"
        else:
            ylabel, column = "Number density (atoms/A^3)", "density_per_A3"
        mean_value = density.mean()
    else:
        if hi is None or hi - lo < 1e-9:
            hi = (lo or 0.0) + 1.0
        edges = np.linspace(lo, hi, nbins + 1)
        for values in cached:
            counts, _ = np.histogram(values, bins=edges, weights=weights[: len(values)])
            hist += counts
        width = edges[1] - edges[0]
        density = hist / n / width
        axis_label = f"{'xyz'[axis]} (A)"
        if mass_kind:
            ylabel, column = "Linear mass density (amu/A)", "density_amu_per_A"
        else:
            ylabel, column = "Linear number density (atoms/A)", "density_per_A"
        mean_value = density.mean()
        notes.append(
            "No cell: the profile is a linear density (per Angstrom along the axis), "
            "not a volume density."
        )
    centres = 0.5 * (edges[1:] + edges[:-1])
    summary = (
        f"{'Mass' if mass_kind else 'Number'} density of '{p['dens_sel']}' "
        f"({len(sel)} atoms) in {nbins} bins; mean {mean_value:.5g}, "
        f"max {density.max():.5g}."
    )
    return Result(
        title="Density profile",
        xlabel=axis_label,
        ylabel=ylabel,
        series=[Series("density", centres, density)],
        columns=["position_A", column],
        rows=np.column_stack([centres, density]),
        summary=summary,
        notes=notes,
    )


# -------------------------------------------------------------------------- energy


def run_energy(topology, frames, params, progress=None, cancelled=None):
    p = merged_params(params)
    indices = frame_indices(frames.n_frames, p["first"], p["last"], p["stride"])
    values = []
    for k, index in enumerate(indices):
        if cancelled is not None and cancelled():
            raise Cancelled()
        values.append(frames.energy(index))
        if progress is not None:
            progress(k + 1, len(indices))
    found = [i for i, v in enumerate(values) if v is not None]
    if not found:
        raise AnalysisError(
            "no per-frame energies were found in this file (looked for "
            "energy=..., E ..., a bare number on XYZ comment lines, PDB REMARKs "
            "and SDF energy properties)"
        )
    energies = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    times, xlabel = time_axis(frames, indices, p["dt"])
    label = frames.energy_label
    if p["energy_relative"]:
        energies = energies - np.nanmin(energies)
        label = "Relative energy (file units)"
    finite = energies[np.isfinite(energies)]
    lowest = indices[int(np.nanargmin(energies))] + 1
    summary = (
        f"{len(found)} of {len(indices)} frames carry an energy. Mean {finite.mean():.8g}, "
        f"std {finite.std():.4g}, min {finite.min():.8g} (frame {lowest}), "
        f"max {finite.max():.8g}. Units are whatever the file uses."
    )
    mask = np.isfinite(energies)
    return Result(
        title="Per-frame energies",
        xlabel=xlabel,
        ylabel=label,
        series=[Series("energy", times[mask], energies[mask])],
        columns=["frame", xlabel.split(" (")[0].lower().replace(" ", "_"), "energy"],
        rows=np.column_stack([np.asarray(indices) + 1, times, energies]),
        summary=summary,
        notes=[_range_note(indices)],
    )


RUNNERS = {
    "rdf": run_rdf,
    "msd": run_msd,
    "rmsd": run_rmsd,
    "rmsf": run_rmsf,
    "rg": run_rg,
    "density": run_density,
    "energy": run_energy,
}


def run(kind, topology, frames, params, progress=None, cancelled=None):
    if kind not in RUNNERS:
        raise AnalysisError(f"unknown analysis {kind!r}")
    return RUNNERS[kind](topology, frames, params, progress, cancelled)
