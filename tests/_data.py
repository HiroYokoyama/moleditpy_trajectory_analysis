"""Small trajectory files written on the fly for the tests."""


def write_gro(path, frames, box_nm=None, names=None, resnames=None, times=None):
    n = frames.shape[1]
    names = names or ["OW"] * n
    resnames = resnames or ["SOL"] * n
    blocks = []
    for f, coords in enumerate(frames):
        title = "test" if times is None else f"test t= {times[f]:.3f}"
        lines = [title, f"{n:5d}"]
        for i, (x, y, z) in enumerate(coords / 10.0):
            lines.append(
                f"{i + 1:5d}{resnames[i]:<5}{names[i]:>5}{i + 1:5d}"
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
            )
        b = box_nm if box_nm is not None else [0.0, 0.0, 0.0]
        lines.append("".join(f"{v:10.5f}" for v in b))
        blocks.append("\n".join(lines))
    path.write_text("\n".join(blocks) + "\n", encoding="ascii")


def write_xyz(path, symbols, frames, comments=None):
    blocks = []
    for f, coords in enumerate(frames):
        comment = comments[f] if comments else f"frame {f}"
        lines = [str(len(symbols)), comment]
        for s, (x, y, z) in zip(symbols, coords):
            lines.append(f"{s} {x:.6f} {y:.6f} {z:.6f}")
        blocks.append("\n".join(lines))
    path.write_text("\n".join(blocks) + "\n", encoding="ascii")


def write_pdb(path, frames, cryst=None, remarks=None, element="C", resname="LIG"):
    lines = []
    if cryst:
        a, b, c, al, be, ga = cryst
        lines.append(
            f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} P 1           1"
        )
    for f, coords in enumerate(frames):
        lines.append(f"MODEL     {f + 1:4d}")
        if remarks:
            lines.append(f"REMARK   1 {remarks[f]}")
        for i, (x, y, z) in enumerate(coords):
            lines.append(
                f"HETATM{i + 1:5d}  {element:<2}  {resname:<3} A{1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"
            )
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
