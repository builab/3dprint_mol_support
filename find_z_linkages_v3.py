"""
find_z_linkages.py
==================
Finds candidate support linkages in a PDB file for 3D printing.

Linkage criteria (Z-direction overhang support):
  - XY distance  <=  xy_threshold   (atoms are nearly vertically aligned)
  - Z separation >= z_min_gap       (there is a real vertical gap worth bridging)
  - Z separation <= z_max_gap       (not so far apart that a link is impractical)
  - 3D distance  <= dist_3d_max     (overall distance cap)
  - Different residues              (no intra-residue links)

Output:
  - Console table of candidate pairs sorted by XY distance (tightest alignment first)
  - CSV file  <input_stem>_z_linkages.csv
  - PDB file  <input_stem>_z_linkages.pdb  with CONECT records for the pairs
  - CMM file  <input_stem>_z_linkages.cmm  ChimeraX marker/link file for visualisation

Usage:
  python find_z_linkages.py protein.pdb [options]

Options:
  --xy_threshold  float  Max XY distance to consider "vertically aligned"  (default 2.5 Å)
  --z_min_gap     float  Min Z separation to count as an overhang gap       (default 3.0 Å)
  --z_max_gap     float  Max Z separation for a practical link              (default 15.0 Å)
  --dist_3d_max   float  Max 3-D distance cap                               (default 20.0 Å)
  --atoms         str    Comma-separated atom names to consider, e.g. CA,CB (default CA)
  --chain         str    Restrict to one chain (default: all chains)
  --output_dir    str    Directory for output files                          (default: same as input)
"""

import argparse
import csv
import math
import os
import sys
from collections import namedtuple
from itertools import combinations
from typing import List, Optional, Set

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Atom = namedtuple("Atom", [
    "serial", "name", "res_name", "chain", "res_seq", "x", "y", "z", "element"
])

Pair = namedtuple("Pair", [
    "atom1", "atom2",
    "xy_dist", "z_gap", "dist_3d",
    "res_sep"          # |res_seq difference| within same chain, else None
])


# ---------------------------------------------------------------------------
# PDB parser (no external dependencies)
# ---------------------------------------------------------------------------

def parse_pdb(path: str, atom_names: Set[str], chain_filter: Optional[str]) -> List[Atom]:
    """Return a list of Atom records matching the requested atom names / chain."""
    atoms = []
    with open(path) as fh:
        for line in fh:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue
            try:
                serial   = int(line[6:11])
                name     = line[12:16].strip()
                res_name = line[17:20].strip()
                chain    = line[21].strip()
                res_seq  = int(line[22:26])
                x        = float(line[30:38])
                y        = float(line[38:46])
                z        = float(line[46:54])
                element  = line[76:78].strip() if len(line) > 76 else ""
            except (ValueError, IndexError):
                continue

            if name not in atom_names:
                continue
            if chain_filter and chain != chain_filter:
                continue

            atoms.append(Atom(serial, name, res_name, chain, res_seq, x, y, z, element))
    return atoms


# ---------------------------------------------------------------------------
# Linkage search
# ---------------------------------------------------------------------------

def find_z_linkages(
    atoms: List[Atom],
    xy_threshold: float,
    z_min_gap: float,
    z_max_gap: float,
    dist_3d_max: float,
    min_res_sep: int = 0,
) -> List[Pair]:
    """
    Return candidate pairs where atoms are nearly vertically aligned (small XY
    offset) but separated by a meaningful Z gap — ideal overhang supports.

    Args:
        min_res_sep: minimum residue-sequence separation required for same-chain
                     pairs (cross-chain pairs always pass this filter).
    """
    pairs = []
    for a1, a2 in combinations(atoms, 2):
        # Skip same residue
        if a1.chain == a2.chain and a1.res_seq == a2.res_seq:
            continue

        dx = a2.x - a1.x
        dy = a2.y - a1.y
        dz = abs(a2.z - a1.z)

        xy_dist = math.hypot(dx, dy)
        dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)

        if xy_dist > xy_threshold:
            continue
        if dz < z_min_gap:
            continue
        if dz > z_max_gap:
            continue
        if dist_3d > dist_3d_max:
            continue

        # Residue separation (only meaningful within the same chain)
        if a1.chain == a2.chain:
            res_sep = abs(a2.res_seq - a1.res_seq)
            if res_sep < min_res_sep:
                continue
        else:
            res_sep = None   # cross-chain — separation not applicable

        pairs.append(Pair(a1, a2, xy_dist, dz, dist_3d, res_sep))

    # Sort: shortest 3-D distance first (best support link), then XY tightness
    pairs.sort(key=lambda p: (p.dist_3d, p.xy_dist))
    return pairs


def select_top_links(pairs: List[Pair], n: int, exclusion_dist: int = 0) -> List[Pair]:
    """
    Pick up to n links greedily from the shortest-first sorted list with
    a spatial exclusion rule:

    Once a residue R on chain C is committed to a selected link, any
    candidate whose atom1 OR atom2 falls within `exclusion_dist` residues
    of R on the same chain is rejected.

    Cross-chain comparisons only exclude if the chains match.

    Algorithm (greedy, O(n * k)):
      1. Iterate candidates in 3-D-distance order (shortest first).
      2. For each candidate, check both endpoints against all already-
         committed (chain, res_seq) values.
      3. Accept if neither endpoint is within exclusion_dist of any
         committed residue on the same chain.
      4. Stop once n links are accepted or candidates are exhausted.

    Args:
        pairs          : candidates sorted shortest-first (output of find_z_linkages)
        n              : max number of links to return
        exclusion_dist : minimum residue-sequence gap from any already-selected
                         residue on the same chain (0 = no exclusion, same as old behaviour)
    """
    if exclusion_dist <= 0:
        return pairs[:n]

    selected: List[Pair] = []
    # committed: list of (chain, res_seq) for every endpoint already chosen
    committed: List[tuple] = []

    def too_close(chain: str, res_seq: int) -> bool:
        """True if this residue is within exclusion_dist of any committed residue on the same chain."""
        for c_chain, c_res in committed:
            if c_chain == chain and abs(res_seq - c_res) < exclusion_dist:
                return True
        return False

    for p in pairs:
        a1, a2 = p.atom1, p.atom2
        if too_close(a1.chain, a1.res_seq):
            continue
        if too_close(a2.chain, a2.res_seq):
            continue
        # Accept this link
        selected.append(p)
        committed.append((a1.chain, a1.res_seq))
        committed.append((a2.chain, a2.res_seq))
        if len(selected) >= n:
            break

    return selected


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_table(pairs: List[Pair], title: str = "Linkage candidates") -> None:
    if not pairs:
        print("No Z-direction linkage candidates found with these thresholds.")
        return

    header = (
        f"{'#':>4}  "
        f"{'Chain1':>6} {'ResSeq1':>7} {'ResName1':>8} {'Atom1':>5}  "
        f"{'Chain2':>6} {'ResSeq2':>7} {'ResName2':>8} {'Atom2':>5}  "
        f"{'ResSep':>7} {'XY(Å)':>8} {'Z-gap(Å)':>9} {'3D(Å)':>8}"
    )
    print(f"\n{'='*len(header)}")
    print(f"  {title}  ({len(pairs)} entries)")
    print(f"{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")
    for i, p in enumerate(pairs, 1):
        a1, a2 = p.atom1, p.atom2
        sep_str = str(p.res_sep) if p.res_sep is not None else "X-chain"
        print(
            f"{i:>4}  "
            f"{a1.chain:>6} {a1.res_seq:>7} {a1.res_name:>8} {a1.name:>5}  "
            f"{a2.chain:>6} {a2.res_seq:>7} {a2.res_name:>8} {a2.name:>5}  "
            f"{sep_str:>7} {p.xy_dist:>8.3f} {p.z_gap:>9.3f} {p.dist_3d:>8.3f}"
        )
    print(f"{'='*len(header)}\n")


def write_csv(pairs: List[Pair], out_path: str) -> None:
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rank",
            "chain1", "res_seq1", "res_name1", "atom_name1", "serial1", "x1", "y1", "z1",
            "chain2", "res_seq2", "res_name2", "atom_name2", "serial2", "x2", "y2", "z2",
            "res_sep", "xy_dist_A", "z_gap_A", "dist_3d_A"
        ])
        for i, p in enumerate(pairs, 1):
            a1, a2 = p.atom1, p.atom2
            writer.writerow([
                i,
                a1.chain, a1.res_seq, a1.res_name, a1.name, a1.serial,
                round(a1.x, 3), round(a1.y, 3), round(a1.z, 3),
                a2.chain, a2.res_seq, a2.res_name, a2.name, a2.serial,
                round(a2.x, 3), round(a2.y, 3), round(a2.z, 3),
                p.res_sep if p.res_sep is not None else "X-chain",
                round(p.xy_dist, 3), round(p.z_gap, 3), round(p.dist_3d, 3)
            ])
    print(f"CSV written  → {out_path}")


def write_pdb_conect(source_pdb: str, pairs: List[Pair], out_path: str) -> None:
    """
    Copy original PDB ATOM/HETATM lines, then append CONECT records for each pair.
    The REMARK block at the top documents the linkage parameters.
    """
    with open(source_pdb) as fh:
        original_lines = fh.readlines()

    with open(out_path, "w") as fh:
        fh.write("REMARK  Z-direction overhang support linkages (find_z_linkages.py)\n")
        fh.write(f"REMARK  Total linkage candidates: {len(pairs)}\n")
        fh.write("REMARK  CONECT records below encode the suggested support bridges\n")

        # Write original structure lines (skip existing CONECT/END)
        for line in original_lines:
            if line[:6].strip() in ("CONECT", "END", "MASTER"):
                continue
            fh.write(line)

        # Append CONECT records
        fh.write("\n")
        for p in pairs:
            s1, s2 = p.atom1.serial, p.atom2.serial
            fh.write(f"CONECT{s1:5d}{s2:5d}\n")
            fh.write(f"CONECT{s2:5d}{s1:5d}\n")

        fh.write("END\n")
    print(f"PDB written  → {out_path}")


# ---------------------------------------------------------------------------
# ChimeraX CMM writer
# ---------------------------------------------------------------------------

def write_cmm(pairs: List[Pair], out_path: str,
              marker_radius: float = 0.5,
              link_radius: float = 0.2) -> None:
    """
    Write a ChimeraX Marker (.cmm) file encoding each Z-linkage as:
      - marker1  — lower atom  (cyan,   at the atom's XYZ)
      - marker2  — upper atom  (orange, at the atom's XYZ)
      - link     — connecting the two markers (yellow-green)

    Colour scheme conveys Z-direction:
      cyan   = lower Z endpoint  (closer to the print bed)
      orange = upper Z endpoint  (overhanging atom being supported)
      link   = the bridging rod

    The CMM format is plain XML; load in ChimeraX with:
      open z_linkages.cmm

    Args:
        pairs        : list of Pair named-tuples from find_z_linkages()
        out_path     : file path to write
        marker_radius: sphere radius in Å for each endpoint marker
        link_radius  : cylinder radius in Å for the link rod
    """
    # Colour constants  (r, g, b, a)  0–1 float range, serialised as 0–255 ints
    COLOUR_LOWER = (0.18, 0.80, 0.90, 1.0)   # cyan
    COLOUR_UPPER = (1.00, 0.55, 0.15, 1.0)   # orange
    COLOUR_LINK  = (0.60, 0.90, 0.20, 1.0)   # yellow-green

    def to255(c):
        """Convert 0-1 float colour tuple to (r, g, b) 0-255 ints."""
        return tuple(round(v * 255) for v in c[:3])

    lines = []
    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append('<marker_sets>')

    marker_id = 1   # CMM requires unique integer IDs across the whole file

    for idx, p in enumerate(pairs, 1):
        # Ensure lower is always the smaller-Z atom
        if p.atom1.z <= p.atom2.z:
            lower, upper = p.atom1, p.atom2
        else:
            lower, upper = p.atom2, p.atom1

        id_lower = marker_id
        id_upper = marker_id + 1
        marker_id += 2

        label_lower = (
            f"link{idx}_lower "
            f"/{lower.chain}:{lower.res_seq}({lower.res_name})@{lower.name}"
        )
        label_upper = (
            f"link{idx}_upper "
            f"/{upper.chain}:{upper.res_seq}({upper.res_name})@{upper.name}"
        )

        lr, lg, lb = to255(COLOUR_LOWER)
        ur, ug, ub = to255(COLOUR_UPPER)
        kr, kg, kb = to255(COLOUR_LINK)

        # Each link gets its own marker_set so draw_dual_tubes can read
        # markers[0] and markers[1] independently per link
        lines.append(f'  <marker_set name="Z-link{idx}">')

        # --- marker1 : lower endpoint (cyan) ---
        lines.append(
            f'    <marker id="{id_lower}"'
            f' x="{lower.x:.3f}" y="{lower.y:.3f}" z="{lower.z:.3f}"'
            f' red="{lr}" green="{lg}" blue="{lb}"'
            f' radius="{marker_radius:.3f}"'
            f' label="{label_lower}"/>'
        )

        # --- marker2 : upper endpoint (orange) ---
        lines.append(
            f'    <marker id="{id_upper}"'
            f' x="{upper.x:.3f}" y="{upper.y:.3f}" z="{upper.z:.3f}"'
            f' red="{ur}" green="{ug}" blue="{ub}"'
            f' radius="{marker_radius:.3f}"'
            f' label="{label_upper}"/>'
        )

        # --- link : rod connecting the two markers (yellow-green) ---
        lines.append(
            f'    <link id1="{id_lower}" id2="{id_upper}"'
            f' red="{kr}" green="{kg}" blue="{kb}"'
            f' radius="{link_radius:.3f}"'
            f' label="Z-link{idx}'
            f' XY={p.xy_dist:.2f}A Z={p.z_gap:.2f}A 3D={p.dist_3d:.2f}A"/>'
        )

        lines.append('  </marker_set>')
    lines.append('</marker_sets>')

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"CMM written  → {out_path}")


# ---------------------------------------------------------------------------
# Z-layer analysis helper
# ---------------------------------------------------------------------------

def z_layer_summary(atoms: List[Atom], pairs: List[Pair]) -> None:
    """Print a brief Z-range and layer coverage summary."""
    if not atoms:
        return
    zs = [a.z for a in atoms]
    z_min, z_max = min(zs), max(zs)
    print(f"Z range of structure : {z_min:.2f} Å  →  {z_max:.2f} Å  "
          f"(span = {z_max - z_min:.2f} Å)")
    if pairs:
        zg = [p.z_gap for p in pairs]
        print(f"Z-gap of candidates  : min={min(zg):.2f} Å, "
              f"max={max(zg):.2f} Å, "
              f"mean={sum(zg)/len(zg):.2f} Å")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Find Z-directional overhang support linkages in a PDB file."
    )
    p.add_argument("pdb", help="Input PDB file path")
    p.add_argument("--xy_threshold", type=float, default=2.5,
                   help="Max XY distance (Å) for 'vertically aligned' (default 2.5)")
    p.add_argument("--z_min_gap", type=float, default=3.0,
                   help="Min Z separation (Å) to qualify as an overhang gap (default 3.0)")
    p.add_argument("--z_max_gap", type=float, default=15.0,
                   help="Max Z separation (Å) for a practical link (default 15.0)")
    p.add_argument("--dist_3d_max", type=float, default=20.0,
                   help="Max 3-D distance cap (Å) (default 20.0)")
    p.add_argument("--atoms", type=str, default="CA",
                   help="Comma-separated atom names to consider, e.g. CA,CB (default CA)")
    p.add_argument("--chain", type=str, default=None,
                   help="Restrict to one chain ID (default: all chains)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: same as input PDB)")
    p.add_argument("--marker_radius", type=float, default=0.5,
                   help="CMM sphere radius in Å for endpoint markers (default 0.5)")
    p.add_argument("--link_radius", type=float, default=0.2,
                   help="CMM cylinder radius in Å for link rods (default 0.2)")
    p.add_argument("--dist", type=int, default=0,
                   help="Min residue-sequence separation for same-chain pairs (default 0, "
                        "i.e. no filter). Cross-chain pairs always pass. Example: --dist 20")
    p.add_argument("--no_links", type=int, default=None,
                   help="Select only the top N shortest linkages after all filters "
                        "(default: keep all). Example: --no_links 10")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    pdb_path = args.pdb
    if not os.path.isfile(pdb_path):
        sys.exit(f"ERROR: File not found: {pdb_path}")

    atom_names = {n.strip() for n in args.atoms.split(",")}
    out_dir    = args.output_dir or os.path.dirname(os.path.abspath(pdb_path))
    stem       = os.path.splitext(os.path.basename(pdb_path))[0]

    print(f"\nPDB file         : {pdb_path}")
    print(f"Atom types       : {', '.join(sorted(atom_names))}")
    print(f"Chain filter     : {args.chain or 'all'}")
    print(f"XY threshold     : {args.xy_threshold} Å")
    print(f"Z gap range      : {args.z_min_gap} – {args.z_max_gap} Å")
    print(f"3-D dist cap     : {args.dist_3d_max} Å")
    print(f"Min residue sep  : {args.dist} residues (same-chain only)")
    print(f"Top-N selection  : {args.no_links if args.no_links else 'all'}\n")

    # Parse
    atoms = parse_pdb(pdb_path, atom_names, args.chain)
    if not atoms:
        sys.exit("ERROR: No matching atoms found. Check --atoms and --chain options.")
    print(f"Atoms loaded     : {len(atoms)}")

    # Z summary
    z_layer_summary(atoms, [])

    # Find linkages (applies geometry + residue-sep filter)
    pairs = find_z_linkages(
        atoms,
        xy_threshold=args.xy_threshold,
        z_min_gap=args.z_min_gap,
        z_max_gap=args.z_max_gap,
        dist_3d_max=args.dist_3d_max,
        min_res_sep=args.dist,
    )

    print(f"After geometry + residue-sep filter : {len(pairs)} candidate(s)")

    # Apply top-N selection with spread exclusion
    if args.no_links is not None:
        pairs_all = pairs
        pairs = select_top_links(pairs, args.no_links, exclusion_dist=args.dist)
        if len(pairs_all) > len(pairs):
            print(f"After --no_links {args.no_links} + exclusion zone (±{args.dist} res) : "
                  f"{len(pairs)} kept (discarded {len(pairs_all) - len(pairs)})")

    # Z summary with final pairs
    z_layer_summary(atoms, pairs)

    # Print table
    title = f"Top {len(pairs)} Z-linkages (ranked shortest 3-D first)"
    print_table(pairs, title=title)

    if not pairs:
        sys.exit(0)

    # Write outputs
    csv_path = os.path.join(out_dir, f"{stem}_z_linkages.csv")
    pdb_out  = os.path.join(out_dir, f"{stem}_z_linkages.pdb")
    cmm_out  = os.path.join(out_dir, f"{stem}_z_linkages.cmm")

    write_csv(pairs, csv_path)
    write_pdb_conect(pdb_path, pairs, pdb_out)
    write_cmm(pairs, cmm_out,
              marker_radius=args.marker_radius,
              link_radius=args.link_radius)

    print(f"\nDone. {len(pairs)} Z-support linkage(s) written.\n")


if __name__ == "__main__":
    main()
