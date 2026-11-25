import pytest

np = pytest.importorskip("numpy")

from trajectory_analysis import analysis as A
from trajectory_analysis.formats import ArrayFrames, Topology


def _rotation(angle, axis):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


# ------------------------------------------------------------------ helpers


def test_frame_indices():
    assert A.frame_indices(10) == list(range(10))
    assert A.frame_indices(10, 3, 7, 2) == [2, 4, 6]
    assert A.frame_indices(10, 1, 99, 4) == [0, 4, 8]
    with pytest.raises(A.AnalysisError):
        A.frame_indices(10, 8, 3)
    with pytest.raises(A.AnalysisError):
        A.frame_indices(0)


def test_minimum_image_triclinic():
    box = np.array([[10.0, 0, 0], [5.0, 8.660254, 0], [0, 0, 10.0]])
    d = np.array([9.0, 0.0, 0.0])
    assert np.allclose(A.minimum_image(d, box), [-1.0, 0.0, 0.0])
    d = box[1] + np.array([0.5, 0.0, 0.0])
    assert np.allclose(A.minimum_image(d, box), [0.5, 0.0, 0.0])
    assert A.max_minimum_image_distance(box) == pytest.approx(4.330127, rel=1e-5)


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 3))
    y = x @ _rotation(0.7, [1, 2, 3]).T + [1.0, -2.0, 0.5]
    moved = A.superpose(x, np.arange(12), y)
    assert np.allclose(moved, y, atol=1e-10)


def test_unwrapper_follows_atoms():
    box = np.eye(3) * 10.0
    unwrap = A.Unwrapper()
    out = [
        unwrap(np.array([[x % 10.0, 0, 0]]), box)[0, 0] for x in (9.0, 9.8, 10.6, 11.4)
    ]
    assert np.allclose(out, [9.0, 9.8, 10.6, 11.4])


# ---------------------------------------------------------------------- RDF


def test_rdf_ideal_gas_is_one():
    rng = np.random.default_rng(0)
    frames = rng.uniform(0, 20.0, (20, 400, 3))
    src = ArrayFrames(frames, default_box=np.eye(3) * 20.0)
    r = A.run_rdf(
        Topology(["Ar"] * 400),
        src,
        {"rdf_sel_a": "all", "rdf_sel_b": "all", "rdf_rmax": 9.0, "rdf_bins": 30},
    )
    assert abs(r.series[0].y[5:].mean() - 1.0) < 0.03
    # n(r) at r max equals the expected number of neighbours in that sphere
    expected = 399 / 20.0**3 * 4 / 3 * np.pi * 9.0**3
    assert r.rows[-1][2] == pytest.approx(expected, rel=0.03)


def test_rdf_pair_across_boundary():
    # Two atoms 1 A apart through the periodic boundary.
    frames = np.array([[[0.5, 5, 5], [9.5, 5, 5]]])
    src = ArrayFrames(frames, default_box=np.eye(3) * 10.0)
    top = Topology(["Na", "Cl"])
    r = A.run_rdf(
        top,
        src,
        {"rdf_sel_a": "Na", "rdf_sel_b": "Cl", "rdf_rmax": 4.0, "rdf_bins": 40},
    )
    peak = r.series[0].x[int(np.argmax(r.series[0].y))]
    assert peak == pytest.approx(1.0, abs=0.06)
    assert r.rows[-1][2] == pytest.approx(1.0)


def test_rdf_rejects_rmax_beyond_half_box():
    src = ArrayFrames(np.zeros((1, 2, 3)), default_box=np.eye(3) * 10.0)
    with pytest.raises(A.AnalysisError):
        A.run_rdf(
            Topology(["O", "O"]),
            src,
            {"rdf_sel_a": "O", "rdf_sel_b": "O", "rdf_rmax": 6},
        )


def test_rdf_without_cell_notes_normalisation():
    src = ArrayFrames(np.random.default_rng(2).uniform(0, 5, (2, 10, 3)))
    r = A.run_rdf(Topology(["O"] * 10), src, {"rdf_rmax": 3.0, "cell_mode": "file"})
    assert any("No cell" in n for n in r.notes)


def test_custom_and_disabled_cell():
    frames = np.array([[[0.5, 5, 5], [9.5, 5, 5]]])
    src = ArrayFrames(frames)
    p = {"rdf_sel_a": "all", "rdf_sel_b": "all", "rdf_rmax": 4.0, "rdf_bins": 40}
    custom = A.run_rdf(
        Topology(["O", "O"]),
        src,
        dict(p, cell_mode="custom", custom_cell=[10, 10, 10, 90, 90, 90]),
    )
    assert custom.rows[-1][2] == pytest.approx(1.0)
    src = ArrayFrames(frames, default_box=np.eye(3) * 10.0)
    none = A.run_rdf(Topology(["O", "O"]), src, dict(p, cell_mode="none"))
    assert none.rows[-1][2] == 0.0


# ---------------------------------------------------------------------- MSD


def test_msd_fft_matches_direct():
    rng = np.random.default_rng(3)
    pos = np.cumsum(rng.normal(size=(30, 4, 3)), axis=0)
    fast = A.msd_fft(pos)
    direct = [
        np.mean(((pos[lag:] - pos[: len(pos) - lag]) ** 2).sum(axis=2))
        for lag in range(30)
    ]
    assert np.allclose(fast, direct)


def test_msd_ballistic_and_units():
    t = np.arange(20, dtype=float)
    frames = np.zeros((20, 2, 3))
    frames[:, 0, 0] = 0.5 * t
    frames[:, 1, 1] = 0.5 * t
    r = A.run_msd(
        Topology(["O", "O"]), ArrayFrames(frames), {"dt": 2.0, "msd_remove_com": False}
    )
    lag = r.series[0].x
    assert np.allclose(r.series[0].y, (0.25 * lag) ** 2)
    assert "A^2/ps" in r.summary


def test_msd_random_walk_diffusion_through_cell():
    rng = np.random.default_rng(4)
    pos = np.cumsum(rng.normal(0, 0.2, (300, 400, 3)), axis=0)
    src = ArrayFrames(pos % 8.0, default_box=np.eye(3) * 8.0)
    r = A.run_msd(Topology(["O"] * 400), src, {"dt": 1.0, "msd_remove_com": False})
    d = float(r.summary.split("D = ")[1].split()[0])
    assert d == pytest.approx(0.04 / 2, rel=0.1)  # sigma^2 / 2 per ps


def test_msd_needs_three_frames():
    with pytest.raises(A.AnalysisError):
        A.run_msd(Topology(["O"]), ArrayFrames(np.zeros((2, 1, 3))), {})


# -------------------------------------------------------------- RMSD / RMSF


def test_rmsd_zero_for_rigid_motion_and_known_without_alignment():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(8, 3)) * 2
    frames = np.stack([x @ _rotation(a, [0, 0, 1]).T + a for a in (0.0, 0.5, 1.0)])
    top = Topology(["C"] * 8)
    aligned = A.run_rmsd(top, ArrayFrames(frames), {"rmsd_sel": "all"})
    assert np.allclose(aligned.series[0].y, 0.0, atol=1e-9)

    shifted = np.stack([x, x + [3.0, 4.0, 0.0]])
    raw = A.run_rmsd(
        top, ArrayFrames(shifted), {"rmsd_sel": "all", "rmsd_align": False}
    )
    assert np.allclose(raw.series[0].y, [0.0, 5.0])


def test_rmsd_reference_out_of_range():
    with pytest.raises(A.AnalysisError):
        A.run_rmsd(
            Topology(["C"] * 3), ArrayFrames(np.zeros((2, 3, 3))), {"rmsd_ref": 5}
        )


def test_rmsf_known_oscillation():
    frames = np.zeros((4, 4, 3))
    frames[:, :3] = [[0, 0, 0], [5, 0, 0], [0, 5, 0]]
    frames[:, 3] = [[1, 1, 1]] * 4
    frames[:, 3, 2] += [1.0, -1.0, 1.0, -1.0]
    r = A.run_rmsf(
        Topology(["C"] * 4),
        ArrayFrames(frames),
        {"rmsd_sel": "all", "rmsd_fit_sel": "index 0-2"},
    )
    assert np.allclose([row[-1] for row in r.rows], [0, 0, 0, 1.0], atol=1e-9)
    assert r.columns[-1] == "rmsf_A"


# ------------------------------------------------------------------- Rg


def test_radius_of_gyration():
    frames = np.array([[[-1.0, 0, 0], [1.0, 0, 0]], [[-2.0, 0, 0], [2.0, 0, 0]]])
    r = A.run_rg(Topology(["C", "C"]), ArrayFrames(frames), {})
    assert np.allclose(r.series[0].y, [1.0, 2.0])
    heavy = A.run_rg(Topology(["H", "O"]), ArrayFrames(frames[:1]), {})
    light = A.run_rg(
        Topology(["H", "O"]), ArrayFrames(frames[:1]), {"rg_mass_weighted": False}
    )
    assert heavy.series[0].y[0] < light.series[0].y[0]


# --------------------------------------------------------------- density


def test_density_uniform_mass_density():
    rng = np.random.default_rng(6)
    n, length = 3000, 20.0
    frames = rng.uniform(0, length, (5, n, 3))
    src = ArrayFrames(frames, default_box=np.eye(3) * length)
    r = A.run_density(Topology(["O"] * n), src, {"dens_bins": 10})
    expected = n * 15.999 / length**3 * A.AMU_PER_A3_TO_G_PER_CM3
    assert np.mean(r.series[0].y) == pytest.approx(expected, rel=1e-9)
    assert r.series[0].x[-1] == pytest.approx(19.0)


def test_density_without_cell_is_linear():
    frames = np.zeros((1, 4, 3))
    frames[0, :, 2] = [0.0, 1.0, 2.0, 3.0]
    r = A.run_density(
        Topology(["O"] * 4),
        ArrayFrames(frames),
        {"dens_kind": "number", "dens_bins": 3},
    )
    assert r.ylabel.startswith("Linear")
    assert r.series[0].y.sum() * 1.0 == pytest.approx(4.0)


# ---------------------------------------------------------------- energy


def test_energy_series_and_missing():
    src = ArrayFrames(np.zeros((3, 1, 3)), energies=[-1.0, None, -3.0])
    r = A.run_energy(Topology(["H"]), src, {"energy_relative": True})
    assert np.allclose(r.series[0].y, [2.0, 0.0])
    assert np.isnan(r.rows[1][2])
    with pytest.raises(A.AnalysisError):
        A.run_energy(Topology(["H"]), ArrayFrames(np.zeros((2, 1, 3))), {})


# ------------------------------------------------------- progress / cancel / csv


def test_progress_and_cancel():
    src = ArrayFrames(np.zeros((5, 3, 3)))
    seen = []
    A.run_rg(Topology(["C"] * 3), src, {}, progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (5, 5)
    with pytest.raises(A.Cancelled):
        A.run_rg(Topology(["C"] * 3), src, {}, cancelled=lambda: True)


def test_time_axis_prefers_dt_then_file_times():
    src = ArrayFrames(np.zeros((3, 1, 3)), times=[10.0, 12.0, 14.0])
    t, label = A.time_axis(src, [0, 2], 0.5)
    assert np.allclose(t, [0.0, 1.0]) and label == "Time (ps)"
    t, label = A.time_axis(src, [0, 2], 0.0)
    assert np.allclose(t, [10.0, 14.0]) and "file" in label
    t, label = A.time_axis(ArrayFrames(np.zeros((3, 1, 3))), [0, 2], 0.0)
    assert np.allclose(t, [1.0, 3.0]) and label == "Frame"


def test_csv_text():
    frames = np.array([[[-1.0, 0, 0], [1.0, 0, 0]]] * 2)
    r = A.run_rmsf(Topology(["C", "C"]), ArrayFrames(frames), {"rmsd_align": False})
    text = r.csv_text()
    lines = text.splitlines()
    assert lines[0].startswith("# RMSF")
    assert "atom,element,name,resname,resid,rmsf_A" in lines
    assert lines[-1].startswith("2,C,C,,0,")


def test_run_dispatch_unknown():
    with pytest.raises(A.AnalysisError):
        A.run("nope", None, None, {})
