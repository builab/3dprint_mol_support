#!/usr/bin/env python3

"""
find_z_linkages_2chains.py
==========================
Finds Z-direction inter-chain linkages between TWO specified protein chains
in a PDB file, for 3D printing overhang support.

Every candidate link crosses from chain A to chain B (never intra-chain).

Linkage criteria:
  - INTER-CHAIN only          (atom1.chain != atom2.chain)
  - XY distance  <= --xy_threshold   (atoms nearly vertically aligned)
  - Z separation >= --z_min_gap      (real vertical gap to bridge)
  - Z separation <= --z_max_gap      (practical link length)
  - 3D distance  <= --dist_3d_max    (overall cap)
  - Residue proximity filter:        each residue in a selected link must be
                                     at least --dist residues away (on its own
                                     chain) from every residue already committed
                                     to a previously selected link.

Selection / ranking:
  - All passing pairs sorted by 3-D distance (shortest first)
  - --no_links N  keeps the top N with the spread-exclusion rule applied

Output:
  - Console table
  - ChimeraX CMM marker/link file (nested format for hierarchical control)

Usage:
  python find_z_linkages_2chains.py protein.pdb --chain1 A --chain2 B [options]
"""

import argparse
import math
import os
import sys
from collections import namedtuple
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Atom = namedtuple("Atom", [
    "serial", "name", "res_name", "chain", "res_seq", "x", "y", "z", "element"
])

Pair = namedtuple("Pair", [
    "atom1", "atom2",   # atom1 is always on chain1, atom2 always on chain2
    "xy_dist",          # Å distance in XY plane
    "z_gap",            # Å |Δz|
    "dist_3d",          # Å full 3-D distance
])


# ---------------------------------------------------------------------------
# PDB parser
# ---------------------------------------------------------------------------

def parse_pdb_two_chains(
    path: str,
    atom_names: Set[str],
    chain1: str,
    chain2: str,
) -> Tuple[List[Atom], List[Atom]]:
    """
    Parse the PDB and return two separate atom lists, one per chain.
    """
    atoms1: List[Atom] = []
    atoms2: List[Atom] = []

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
            if chain not in (chain1, chain2):
                continue

            atom = Atom(serial, name, res_name, chain, res_seq, x, y, z, element)
            if chain == chain1:
                atoms1.append(atom)
            else:
                atoms2.append(atom)

    return atoms1, atoms2


# ---------------------------------------------------------------------------
# Z-linkage search
# ---------------------------------------------------------------------------

def find_z_linkages_2chains(
    atoms1: List[Atom],
    atoms2: List[Atom],
    xy_threshold: float,
    z_min_gap: float,
    z_max_gap: float,
    dist_3d_max: float,
) -> List[Pair]:
    pairs: List[Pair] = []

    for a1 in atoms1:
        for a2 in atoms2:
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

            pairs.append(Pair(a1, a2, xy_dist, dz, dist_3d))

    pairs.sort(key=lambda p: (p.dist_3d, p.xy_dist))
    return pairs


def select_top_links(
    pairs: List[Pair],
    n: int,
    exclusion_dist: int,
) -> List[Pair]:
    if exclusion_dist <= 0:
        return pairs[:n]

    selected: List[Pair] = []
    committed: List[Tuple[str, int]] = []

    def too_close(chain: str, res_seq: int) -> bool:
        for c_chain, c_res in committed:
            if c_chain == chain and abs(res_seq - c_res) < exclusion_dist:
                return True
        return False

    for p in pairs:
        if too_close(p.atom1.chain, p.atom1.res_seq):
            continue
        if too_close(p.atom2.chain, p.atom2.res_seq):
            continue
        selected.append(p)
        committed.append((p.atom1.chain, p.atom1.res_seq))
        committed.append((p.atom2.chain, p.atom2.res_seq))
        if len(selected) >= n:
            break

    return selected


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_table(pairs: List[Pair], chain1: str, chain2: str,
                title: str = "Inter-chain Z-linkages") -> None:
    if not pairs:
        print("No inter-chain Z-linkage candidates found.")
        return

    header = (
        f"{'#':>4}  "
        f"{'Chain1':>6} {'ResSeq1':>7} {'ResName1':>8} {'Atom1':>5}  "
        f"{'Chain2':>6} {'ResSeq2':>7} {'ResName2':>8} {'Atom2':>5}  "
        f"{'XY(Å)':>8} {'Z-gap(Å)':>9} {'3D(Å)':>8}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(f"  {title}  ({len(pairs)} links, {chain1} ↔ {chain2})")
    print(sep)
    print(header)
    print("-" * len(header))
    for i, p in enumerate(pairs, 1):
        a1, a2 = p.atom1, p.atom2
        print(
            f"{i:>4}  "
            f"{a1.chain:>6} {a1.res_seq:>7} {a1.res_name:>8} {a1.name:>5}  "
            f"{a2.chain:>6} {a2.res_seq:>7} {a2.res_name:>8} {a2.name:>5}  "
            f"{p.xy_dist:>8.3f} {p.z_gap:>9.3f} {p.dist_3d:>8.3f}"
        )
    print(f"{sep}\n")


def z_range_summary(atoms1: List[Atom], atoms2: List[Atom],
                    chain1: str, chain2: str) -> None:
    def zrange(atoms: List[Atom]) -> Tuple[float, float]:
        zs = [a.z for a in atoms]
        return min(zs), max(zs)
    z1lo, z1hi = zrange(atoms1)
    z2lo, z2hi = zrange(atoms2)
    print(f"  Chain {chain1} Z range : {z1lo:.2f} → {z1hi:.2f} Å")
    print(f"  Chain {chain2} Z range : {z2lo:.2f} → {z2hi:.2f} Å")


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def write_cmm(pairs: List[Pair], out_path: str,
              chain1: str, chain2: str,
              marker_radius: float = 0.5,
              link_radius: float = 0.2) -> None:
    """
    Write a nested ChimeraX CMM file. Each link is its own marker_set
    under a shared parent name, facilitating hierarchical control.
    """
    COL = {
        "chain1": (0.18, 0.80, 0.90),   # cyan
        "chain2": (0.90, 0.20, 0.75),   # magenta
        "link":   (0.60, 0.90, 0.20),   # yellow-green
    }

    def to255(c: tuple) -> Tuple[int, int, int]:
        return tuple(round(v * 255) for v in c)

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<marker_sets>'
    ]

    parent_name = f"Z-Linkages ({chain1}-{chain2})"

    for idx, p in enumerate(pairs, 1):
        a1, a2 = p.atom1, p.atom2
        
        # Nested naming structure for ChimeraX hierarchy
        set_name = f"{parent_name}/Link {idx} ({a1.res_seq}-{a2.res_seq})"
        lines.append(f'  <marker_set name="{set_name}">')

        r1, g1, b1 = to255(COL["chain1"])
        r2, g2, b2 = to255(COL["chain2"])
        rl, gl, bl = to255(COL["link"])

        label1 = f"link{idx}_ch1 /{a1.chain}:{a1.res_seq}@{a1.name}"
        label2 = f"link{idx}_ch2 /{a2.chain}:{a2.res_seq}@{a2.name}"

        lines.append(
            f'    <marker id="1" x="{a1.x:.3f}" y="{a1.y:.3f}" z="{a1.z:.3f}"'
            f' r="{r1}" g="{g1}" b="{b1}" radius="{marker_radius:.3f}" label="{label1}"/>'
        )
        lines.append(
            f'    <marker id="2" x="{a2.x:.3f}" y="{a2.y:.3f}" z="{a2.z:.3f}"'
            f' r="{r2}" g="{g2}" b="{b2}" radius="{marker_radius:.3f}" label="{label2}"/>'
        )
        lines.append(
            f'    <link id1="1" id2="2" r="{rl}" g="{gl}" b="{bl}" radius="{link_radius:.3f}"'
            f' label="Dist={p.dist_3d:.2f}A"/>'
        )
        lines.append('  </marker_set>')

    lines += ['</marker_sets>', '']
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Nested CMM written → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Find Z-direction inter-chain linkages.")
    p.add_argument("pdb", help="Input PDB file")
    p.add_argument("--chain1", required=True, help="First chain ID")
    p.add_argument("--chain2", required=True, help="Second chain ID")
    p.add_argument("--atoms", type=str, default="CA", help="Atom names (default: CA)")
    p.add_argument("--xy_threshold", type=float, default=2.5)
    p.add_argument("--z_min_gap", type=float, default=3.0)
    p.add_argument("--z_max_gap", type=float, default=15.0)
    p.add_argument("--dist_3d_max", type=float, default=20.0)
    p.add_argument("--dist", type=int, default=10, help="Residue exclusion zone")
    p.add_argument("--no_links", type=int, default=None, help="Top N links")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--marker_radius", type=float, default=0.5)
    p.add_argument("--link_radius", type=float, default=0.2)
    args = p.parse_args()

    if not os.path.isfile(args.pdb):
        sys.exit(f"ERROR: File not found: {args.pdb}")

    atom_names = {n.strip() for n in args.atoms.split(",")}
    out_dir    = args.output_dir or os.path.dirname(os.path.abspath(args.pdb))
    stem       = os.path.splitext(os.path.basename(args.pdb))[0]

    atoms1, atoms2 = parse_pdb_two_chains(args.pdb, atom_names, args.chain1, args.chain2)
    if not atoms1 or not atoms2:
        sys.exit("ERROR: No matching atoms found in specified chains.")

    z_range_summary(atoms1, atoms2, args.chain1, args.chain2)

    pairs = find_z_linkages_2chains(atoms1, atoms2, args.xy_threshold, 
                                   args.z_min_gap, args.z_max_gap, args.dist_3d_max)

    if args.no_links is not None:
        pairs = select_top_links(pairs, args.no_links, exclusion_dist=args.dist)

    print_table(pairs, args.chain1, args.chain2)

    cmm_out = os.path.join(out_dir, f"{stem}_2chain_z_linkages.cmm")
    write_cmm(pairs, cmm_out, args.chain1, args.chain2, args.marker_radius, args.link_radius)

if __name__ == "__main__":
    main()