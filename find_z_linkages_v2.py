"""
find_z_linkages.py
==================
Finds candidate support linkages in a PDB file for 3D printing.
Updated to output individual marker_sets per link and removed PDB/CSV output.
"""

import argparse
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
    "res_sep"          
])


# ---------------------------------------------------------------------------
# PDB parser
# ---------------------------------------------------------------------------

def parse_pdb(path: str, atom_names: Set[str], chain_filter: Optional[str]) -> List[Atom]:
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
    pairs = []
    for a1, a2 in combinations(atoms, 2):
        if a1.chain == a2.chain and a1.res_seq == a2.res_seq:
            continue

        dx = a2.x - a1.x
        dy = a2.y - a1.y
        dz = abs(a2.z - a1.z)

        xy_dist = math.hypot(dx, dy)
        dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)

        if xy_dist > xy_threshold or dz < z_min_gap or dz > z_max_gap or dist_3d > dist_3d_max:
            continue

        if a1.chain == a2.chain:
            res_sep = abs(a2.res_seq - a1.res_seq)
            if res_sep < min_res_sep:
                continue
        else:
            res_sep = None

        pairs.append(Pair(a1, a2, xy_dist, dz, dist_3d, res_sep))

    pairs.sort(key=lambda p: (p.dist_3d, p.xy_dist))
    return pairs


def select_top_links(pairs: List[Pair], n: int, exclusion_dist: int = 0) -> List[Pair]:
    if exclusion_dist <= 0:
        return pairs[:n]

    selected: List[Pair] = []
    committed: List[tuple] = []

    def too_close(chain: str, res_seq: int) -> bool:
        for c_chain, c_res in committed:
            if c_chain == chain and abs(res_seq - c_res) < exclusion_dist:
                return True
        return False

    for p in pairs:
        a1, a2 = p.atom1, p.atom2
        if too_close(a1.chain, a1.res_seq) or too_close(a2.chain, a2.res_seq):
            continue
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
        print("No Z-direction linkage candidates found.")
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


# ---------------------------------------------------------------------------
# ChimeraX CMM writer
# ---------------------------------------------------------------------------

def write_cmm(pairs: List[Pair], out_path: str,
              marker_radius: float = 1.0,
              link_radius: float = 0.4) -> None:
    """
    Writes CMM in the specific format: <marker_sets> containing individual <marker_set> 
    per link, with IDs resetting to 1 and 2.
    """
    # Cyan for lower, Orange for upper
    COL_LOWER = (0.180, 0.800, 0.900)
    COL_UPPER = (1.000, 0.550, 0.150)
    COL_LINK  = (1.000, 1.000, 1.000) 

    lines = ['<?xml version="1.0" encoding="utf-8"?>', '<marker_sets>']

    for i, p in enumerate(pairs, 1):
        if p.atom1.z <= p.atom2.z:
            lower, upper = p.atom1, p.atom2
        else:
            lower, upper = p.atom2, p.atom1

        lines.append(f'  <marker_set name="z_link_{i}">')
        
        # Marker 1 (Lower)
        note_l = f"{lower.chain}:{lower.res_name}{lower.res_seq}({lower.name})"
        lines.append(f'    <marker id="1" x="{lower.x:.3f}" y="{lower.y:.3f}" z="{lower.z:.3f}" '
                     f'r="{COL_LOWER[0]:.3f}" g="{COL_LOWER[1]:.3f}" b="{COL_LOWER[2]:.3f}" '
                     f'radius="{marker_radius:.3f}" note="{note_l}"/>')
        
        # Marker 2 (Upper)
        note_u = f"{upper.chain}:{upper.res_name}{upper.res_seq}({upper.name})"
        lines.append(f'    <marker id="2" x="{upper.x:.3f}" y="{upper.y:.3f}" z="{upper.z:.3f}" '
                     f'r="{COL_UPPER[0]:.3f}" g="{COL_UPPER[1]:.3f}" b="{COL_UPPER[2]:.3f}" '
                     f'radius="{marker_radius:.3f}" note="{note_u}"/>')
        
        # Link
        lines.append(f'    <link id1="1" id2="2" r="{COL_LINK[0]:.3f}" g="{COL_LINK[1]:.3f}" b="{COL_LINK[2]:.3f}" '
                     f'radius="{link_radius:.3f}"/>')
        
        lines.append('  </marker_set>')

    lines.append('</marker_sets>')

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"CMM written  → {out_path}")


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------

def z_layer_summary(atoms: List[Atom], pairs: List[Pair]) -> None:
    if not atoms: return
    zs = [a.z for a in atoms]
    print(f"Z range of structure : {min(zs):.2f} Å  →  {max(zs):.2f} Å")

def main():
    parser = argparse.ArgumentParser(description="Find Z-directional overhang support linkages.")
    parser.add_argument("pdb", help="Input PDB file path")
    parser.add_argument("--xy_threshold", type=float, default=2.5)
    parser.add_argument("--z_min_gap", type=float, default=3.0)
    parser.add_argument("--z_max_gap", type=float, default=15.0)
    parser.add_argument("--dist_3d_max", type=float, default=20.0)
    parser.add_argument("--atoms", type=str, default="CA")
    parser.add_argument("--chain", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dist", type=int, default=0)
    parser.add_argument("--no_links", type=int, default=None)
    args = parser.parse_args()

    atom_names = {n.strip() for n in args.atoms.split(",")}
    out_dir    = args.output_dir or os.path.dirname(os.path.abspath(args.pdb))
    stem       = os.path.splitext(os.path.basename(args.pdb))[0]

    atoms = parse_pdb(args.pdb, atom_names, args.chain)
    if not atoms: sys.exit("ERROR: No matching atoms found.")

    pairs = find_z_linkages(atoms, args.xy_threshold, args.z_min_gap, args.z_max_gap, args.dist_3d_max, args.dist)

    if args.no_links is not None:
        pairs = select_top_links(pairs, args.no_links, exclusion_dist=args.dist)

    print_table(pairs, title=f"Top {len(pairs)} Z-linkages")

    if pairs:
        cmm_out  = os.path.join(out_dir, f"{stem}_z_linkages.cmm")
        write_cmm(pairs, cmm_out)

if __name__ == "__main__":
    main()