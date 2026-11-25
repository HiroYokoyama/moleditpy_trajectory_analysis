import pytest

np = pytest.importorskip("numpy")

from trajectory_analysis.formats import Topology
from trajectory_analysis.selection import (
    SelectionError,
    select,
    select_indices,
)


@pytest.fixture
def top():
    return Topology(
        ["O", "H", "H", "Na", "Cl", "C", "N"],
        ["OW", "HW1", "HW2", "NA", "CL", "CA", "N"],
        ["SOL", "SOL", "SOL", "NA", "CL", "ALA", "ALA"],
        [1, 1, 1, 2, 3, 4, 4],
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("all", [0, 1, 2, 3, 4, 5, 6]),
        ("O", [0]),
        ("element h", [1, 2]),
        ("elem Na Cl", [3, 4]),
        ("resname SOL", [0, 1, 2]),
        ("name OW", [0]),
        ("resid 2-3", [3, 4]),
        ("index 0 5-6", [0, 5, 6]),
        ("water", [0, 1, 2]),
        ("protein", [5, 6]),
        ("heavy and water", [0]),
        ("not hydrogen and not water", [3, 4, 5, 6]),
        ("O or N", [0, 6]),
        ("(resname SOL or resname ALA) and not hydrogen", [0, 5, 6]),
        ("none", []),
    ],
)
def test_selections(top, text, expected):
    assert np.flatnonzero(select(text, top)).tolist() == expected


@pytest.mark.parametrize("text", ["", "element", "resid a", "(O", "bogus", "O )"])
def test_errors(top, text):
    with pytest.raises(SelectionError):
        select(text, top)


def test_empty_selection_raises(top):
    with pytest.raises(SelectionError):
        select_indices("element Fe", top)
