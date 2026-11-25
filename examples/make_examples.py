"""Regenerate the example trajectories in this folder (needs numpy and RDKit).

water_box.gro      64 rigid waters diffusing in a 12.42 A cubic box, 40 frames,
                   2 ps apart, written the way GROMACS trjconv writes them.
ethanol_conf.xyz   10 ethanol conformers with MMFF94 energies on the comment line.
"""

import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))


def water_box(path, n_side=4, length=12.42, frames=40, seed=7):
    rng = np.random.default_rng(seed)
    geom = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2400, 0.9266, 0.0]])
    spacing = length / n_side
    centres = (
        np.stack(np.meshgrid(*[np.arange(n_side)] * 3, indexing="ij"), -1).reshape(
            -1, 3
        )
        + 0.5
    ) * spacing
    n = len(centres)
    rots = [np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(n)]
    out = []
    for f in range(frames):
        lines = [f"Water box t= {2.0 * f:.5f} step= {1000 * f}", f"{3 * n:5d}"]
        k = 1
        for m in range(n):
            atoms = centres[m] + geom @ rots[m].T
            # GROMACS keeps molecules whole: wrap by the oxygen only.
            shift = np.floor(atoms[0] / length) * length
            atoms = atoms - shift
            for name, xyz in zip(("OW", "HW1", "HW2"), atoms):
                x, y, z = xyz / 10.0
                lines.append(
                    f"{m + 1:5d}{'SOL':<5}{name:>5}{k:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
                )
                k += 1
        lines.append(f"{length / 10:10.5f}{length / 10:10.5f}{length / 10:10.5f}")
        out.append("\n".join(lines))
        centres = centres + rng.normal(0.0, 0.35, centres.shape)
        for m in range(n):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.normal(0.0, 0.3)
            kx = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
            )
            r = np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx
            rots[m] = r @ rots[m]
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")


def ethanol_conformers(path, n=10, seed=11):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=n, randomSeed=seed)
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=0)
    blocks = []
    for cid, (_, energy) in zip(ids, results):
        conf = mol.GetConformer(cid)
        lines = [str(mol.GetNumAtoms()), f"conformer {cid + 1} energy={energy:.6f}"]
        for atom in mol.GetAtoms():
            p = conf.GetAtomPosition(atom.GetIdx())
            lines.append(f"{atom.GetSymbol():2s} {p.x:12.6f} {p.y:12.6f} {p.z:12.6f}")
        blocks.append("\n".join(lines))
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(blocks) + "\n")


if __name__ == "__main__":
    water_box(os.path.join(HERE, "water_box.gro"))
    ethanol_conformers(os.path.join(HERE, "ethanol_conf.xyz"))
