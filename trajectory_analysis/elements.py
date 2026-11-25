"""Element symbols, standard atomic masses and element guessing from atom names.

Pure Python so the readers and the analysis core can be tested without RDKit.
The guessing rules follow the Large System Viewer plugin's readers (copied, not
imported, so this plugin works on its own).
"""

# Symbol and standard atomic weight (IUPAC conventional values, amu), Z = 0..86.
_TABLE = (
    ("X", 0.0),
    ("H", 1.008),
    ("He", 4.0026),
    ("Li", 6.94),
    ("Be", 9.0122),
    ("B", 10.81),
    ("C", 12.011),
    ("N", 14.007),
    ("O", 15.999),
    ("F", 18.998),
    ("Ne", 20.180),
    ("Na", 22.990),
    ("Mg", 24.305),
    ("Al", 26.982),
    ("Si", 28.085),
    ("P", 30.974),
    ("S", 32.06),
    ("Cl", 35.45),
    ("Ar", 39.948),
    ("K", 39.098),
    ("Ca", 40.078),
    ("Sc", 44.956),
    ("Ti", 47.867),
    ("V", 50.942),
    ("Cr", 51.996),
    ("Mn", 54.938),
    ("Fe", 55.845),
    ("Co", 58.933),
    ("Ni", 58.693),
    ("Cu", 63.546),
    ("Zn", 65.38),
    ("Ga", 69.723),
    ("Ge", 72.630),
    ("As", 74.922),
    ("Se", 78.971),
    ("Br", 79.904),
    ("Kr", 83.798),
    ("Rb", 85.468),
    ("Sr", 87.62),
    ("Y", 88.906),
    ("Zr", 91.224),
    ("Nb", 92.906),
    ("Mo", 95.95),
    ("Tc", 98.0),
    ("Ru", 101.07),
    ("Rh", 102.91),
    ("Pd", 106.42),
    ("Ag", 107.87),
    ("Cd", 112.41),
    ("In", 114.82),
    ("Sn", 118.71),
    ("Sb", 121.76),
    ("Te", 127.60),
    ("I", 126.90),
    ("Xe", 131.29),
    ("Cs", 132.91),
    ("Ba", 137.33),
    ("La", 138.91),
    ("Ce", 140.12),
    ("Pr", 140.91),
    ("Nd", 144.24),
    ("Pm", 145.0),
    ("Sm", 150.36),
    ("Eu", 151.96),
    ("Gd", 157.25),
    ("Tb", 158.93),
    ("Dy", 162.50),
    ("Ho", 164.93),
    ("Er", 167.26),
    ("Tm", 168.93),
    ("Yb", 173.05),
    ("Lu", 174.97),
    ("Hf", 178.49),
    ("Ta", 180.95),
    ("W", 183.84),
    ("Re", 186.21),
    ("Os", 190.23),
    ("Ir", 192.22),
    ("Pt", 195.08),
    ("Au", 196.97),
    ("Hg", 200.59),
    ("Tl", 204.38),
    ("Pb", 207.2),
    ("Bi", 208.98),
    ("Po", 209.0),
    ("At", 210.0),
    ("Rn", 222.0),
)

SYMBOLS = [s for s, _ in _TABLE]
MASSES = [m for _, m in _TABLE]
ATOMIC_NUMBER = {s.upper(): z for z, s in enumerate(SYMBOLS) if z > 0}

# Residue names that are single-ion residues in MD force fields.
ION_RESIDUES = {
    "NA": "Na",
    "NA+": "Na",
    "SOD": "Na",
    "CL": "Cl",
    "CL-": "Cl",
    "CLA": "Cl",
    "K": "K",
    "K+": "K",
    "POT": "K",
    "MG": "Mg",
    "MG2": "Mg",
    "CA": "Ca",
    "CAL": "Ca",
    "ZN": "Zn",
    "ZN2": "Zn",
    "LI": "Li",
    "RB": "Rb",
    "CS": "Cs",
    "BR": "Br",
    "IOD": "I",
    "F": "F",
}

PROTEIN_RESIDUES = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR "
    "VAL HID HIE HIP HSD HSE HSP CYX ASH GLH LYN NALA NGLY CALA".split()
)

WATER_RESIDUES = frozenset(
    "HOH WAT H2O SOL TIP3 TIP4 TIP5 TIP SPC DOD T3P T4P".split()
)


def atomic_number(symbol):
    """Atomic number for an element symbol or number string; 0 when unknown."""
    text = str(symbol).strip()
    if not text:
        return 0
    if text.isdigit():
        z = int(text)
        return z if 0 <= z < len(SYMBOLS) else 0
    return ATOMIC_NUMBER.get(text.upper(), 0)


def symbol(z):
    """Element symbol for an atomic number; 'X' when out of range."""
    return SYMBOLS[z] if 0 <= z < len(SYMBOLS) else "X"


def mass(z):
    """Standard atomic weight in amu; 0 for unknown elements."""
    return MASSES[z] if 0 <= z < len(MASSES) else 0.0


def normalize_symbol(text):
    """'CL' / 'cl' / 'Cl' -> 'Cl'; unknown symbols are returned title-cased."""
    z = atomic_number(text)
    if z:
        return SYMBOLS[z]
    text = str(text).strip()
    return text[:1].upper() + text[1:].lower()


def _first_alpha(text):
    for ch in text:
        if ch.isalpha():
            return ch.upper()
    return ""


def guess_pdb_element(name4, resname="", hetatm=False):
    """Element of a PDB atom from its 4-column name (columns 13-16)."""
    name4 = (name4 or "").ljust(4)[:4]
    stripped = name4.strip()
    if not stripped:
        return "X"
    ion = ION_RESIDUES.get(resname.strip().upper())
    if ion and stripped.upper().rstrip("+-0123456789") == ion.upper():
        return ion
    if name4[0].isdigit():
        return _first_alpha(name4[1:]) or "X"
    if name4[0] != " " and hetatm:
        candidate = name4[:2].strip()
        if (
            len(candidate) == 2
            and candidate.isalpha()
            and candidate.upper() in ATOMIC_NUMBER
            and not (candidate[0].upper() == "H" and len(stripped) == 4)
        ):
            return candidate[0].upper() + candidate[1].lower()
    return _first_alpha(stripped) or "X"


def guess_name_element(name, resname=""):
    """Element of an atom named without column conventions (GRO names)."""
    ion = ION_RESIDUES.get(resname.strip().upper())
    stripped = name.strip()
    if ion and stripped.upper().rstrip("+-0123456789") in (
        ion.upper(),
        resname.strip().upper(),
    ):
        return ion
    letters = "".join(ch for ch in stripped if ch.isalpha())
    if not letters:
        return "X"
    if (
        letters[:2].upper() in ("CL", "BR")
        and resname.strip().upper() not in PROTEIN_RESIDUES
    ):
        return letters[0].upper() + letters[1].lower()
    return letters[0].upper()
