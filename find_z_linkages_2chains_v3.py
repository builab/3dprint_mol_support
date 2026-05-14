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

Output (all named  <stem>_2chain_z_linkages.*):
  - Console table
  - CSV
  - PDB with CONECT records
  - ChimeraX CMM marker/link file

Usage:
  python find_z_linkages_2chains.py protein.pdb --chain1 A --chain2 B [options]

Required:
  --chain1  str    First  chain ID  (e.g. A)
  --chain2  str    Second chain ID  (e.g. B)

Options:
  --atoms        str    Atom names to search, comma-separated  (default: CA)
  --xy_threshold float  Max XY distance in Å                   (default: 2.5)
  --z_min_gap    float  Min Z gap in Å                         (default: 3.0)
  --z_max_gap    float  Max Z gap in Å                         (default: 15.0)
  --dist_3d_max  float  Max 3-D distance cap in Å              (default: 20.0)
  --dist         int    Min residue separation for exclusion zone on each
                        chain when selecting top links          (default: 10)
  --no_links     int    Keep only the top N shortest links      (default: all)
  --output_dir   str    Output directory  (default: same dir as PDB)
  --marker_radius float CMM sphere radius in Å                 (default: 0.5)
  --link_radius   float CMM rod    radius in Å                 (default: 0.2)
"""

import argparse
import csv
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
    "atom1", "atom2",   # atom1 is always the LOWER-Z atom, atom2 the UPPER-Z atom
    "xy_dist",          # Å distance in XY plane
    "z_gap",            # Å |Δz|  (always atom2.z - atom1.z  > 0)
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
    Only ATOM / HETATM records matching the requested atom names are kept.
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
# Z-linkage search  (inter-chain only)
# ---------------------------------------------------------------------------

def find_z_linkages_2chains(
    atoms1: List[Atom],
    atoms2: List[Atom],
    xy_threshold: float,
    z_min_gap: float,
    z_max_gap: float,
    dist_3d_max: float,
) -> List[Pair]:
    """
    Find all inter-chain pairs (one atom from each chain) that satisfy the
    Z-direction geometry criteria.

    Results are sorted by 3-D distance (shortest first).
    """
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

            # Normalise: atom1 = lower Z (bed side), atom2 = upper Z (overhang)
            lower, upper = (a1, a2) if a1.z <= a2.z else (a2, a1)
            pairs.append(Pair(lower, upper, xy_dist, dz, dist_3d))

    # Shortest 3-D distance first → best structural support links
    pairs.sort(key=lambda p: (p.dist_3d, p.xy_dist))
    return pairs


# ---------------------------------------------------------------------------
# Greedy top-N selection with per-chain exclusion zones
# ---------------------------------------------------------------------------

def select_top_links(
    pairs: List[Pair],
    n: int,
    exclusion_dist: int,
) -> List[Pair]:
    """
    Pick up to n inter-chain links greedily (shortest first) such that no
    two selected links share a residue neighbourhood on the same chain.

    Exclusion rule:
      Once residue R on chain C is committed, any candidate whose endpoint
      on chain C falls within `exclusion_dist` residues of R is rejected.
      The two chains are treated independently — exclusion on chain A does
      not affect chain B.

    Args:
        pairs          : candidates sorted shortest-first
        n              : maximum links to return
        exclusion_dist : minimum residue-sequence gap from any committed
                         residue on the same chain (0 = keep all up to n)
    """
    if exclusion_dist <= 0:
        return pairs[:n]

    selected: List[Pair] = []
    # committed: list of (chain_id, res_seq)
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
        print("No inter-chain Z-linkage candidates found with these thresholds.")
        return

    header = (
        f"{'#':>4}  "
        f"{'Lower(bed)':>10} {'ResSeq':>7} {'ResName':>8} {'Atom':>5}  "
        f"{'Upper(top)':>10} {'ResSeq':>7} {'ResName':>8} {'Atom':>5}  "
        f"{'XY(Å)':>8} {'Z-gap(Å)':>9} {'3D(Å)':>8}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(f"  {title}  ({len(pairs)} links, chain {chain1} ↔ chain {chain2})")
    print(f"  atom1 = lower Z (print-bed side)   atom2 = upper Z (overhang side)")
    print(sep)
    print(header)
    print("-" * len(header))
    for i, p in enumerate(pairs, 1):
        lo, up = p.atom1, p.atom2   # atom1=lower, atom2=upper by construction
        print(
            f"{i:>4}  "
            f"{lo.chain:>10} {lo.res_seq:>7} {lo.res_name:>8} {lo.name:>5}  "
            f"{up.chain:>10} {up.res_seq:>7} {up.res_name:>8} {up.name:>5}  "
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
    print(f"  Chain {chain1} Z range : {z1lo:.2f} → {z1hi:.2f} Å  (span {z1hi-z1lo:.2f} Å)")
    print(f"  Chain {chain2} Z range : {z2lo:.2f} → {z2hi:.2f} Å  (span {z2hi-z2lo:.2f} Å)")


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def write_csv(pairs: List[Pair], out_path: str) -> None:
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rank",
            "lower_chain", "lower_res_seq", "lower_res_name", "lower_atom", "lower_serial",
            "lower_x", "lower_y", "lower_z",
            "upper_chain", "upper_res_seq", "upper_res_name", "upper_atom", "upper_serial",
            "upper_x", "upper_y", "upper_z",
            "xy_dist_A", "z_gap_A", "dist_3d_A",
        ])
        for i, p in enumerate(pairs, 1):
            lo, up = p.atom1, p.atom2   # atom1=lower, atom2=upper
            writer.writerow([
                i,
                lo.chain, lo.res_seq, lo.res_name, lo.name, lo.serial,
                round(lo.x, 3), round(lo.y, 3), round(lo.z, 3),
                up.chain, up.res_seq, up.res_name, up.name, up.serial,
                round(up.x, 3), round(up.y, 3), round(up.z, 3),
                round(p.xy_dist, 3), round(p.z_gap, 3), round(p.dist_3d, 3),
            ])
    print(f"CSV written  → {out_path}")


def write_pdb_conect(source_pdb: str, pairs: List[Pair], out_path: str,
                     chain1: str, chain2: str) -> None:
    with open(source_pdb) as fh:
        original_lines = fh.readlines()

    with open(out_path, "w") as fh:
        fh.write("REMARK  Inter-chain Z-direction linkages (find_z_linkages_2chains.py)\n")
        fh.write(f"REMARK  Chains: {chain1} <-> {chain2}   Links: {len(pairs)}\n")
        fh.write("REMARK  CONECT records encode the suggested support bridges\n")

        for line in original_lines:
            if line[:6].strip() in ("CONECT", "END", "MASTER"):
                continue
            fh.write(line)

        fh.write("\n")
        for p in pairs:
            s1, s2 = p.atom1.serial, p.atom2.serial
            fh.write(f"CONECT{s1:5d}{s2:5d}\n")
            fh.write(f"CONECT{s2:5d}{s1:5d}\n")

        fh.write("END\n")
    print(f"PDB written  → {out_path}")


def write_cmm(pairs: List[Pair], out_path: str,
              chain1: str, chain2: str,
              marker_radius: float = 0.5,
              link_radius: float = 0.2) -> None:
    """
    Write a ChimeraX CMM file.

    Colour scheme reflects Z position (not chain identity):
      cyan    = lower-Z endpoint  (print-bed / support base side)
      orange  = upper-Z endpoint  (overhanging atom being supported)
      yellow-green = link rod

    Since atom1 is always lower-Z and atom2 always upper-Z (enforced at
    search time), cyan always maps to atom1 and orange to atom2.

    Load in ChimeraX:
      open <file>.cmm
    """
    COLOUR_LOWER = (0.18, 0.80, 0.90)   # cyan   — bed side
    COLOUR_UPPER = (1.00, 0.55, 0.15)   # orange — overhang side
    COLOUR_LINK  = (0.60, 0.90, 0.20)   # yellow-green

    def to255(c: tuple) -> Tuple[int, int, int]:
        return tuple(round(v * 255) for v in c)

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<marker_sets>',
        '<!-- cyan = lower-Z (bed side)  orange = upper-Z (overhang side) -->',
    ]

    marker_id = 1

    for idx, p in enumerate(pairs, 1):
        lo, up = p.atom1, p.atom2   # atom1=lower-Z, atom2=upper-Z

        id_lo = marker_id
        id_up = marker_id + 1
        marker_id += 2

        rl, gl, bl = to255(COLOUR_LOWER)
        ru, gu, bu = to255(COLOUR_UPPER)
        rk, gk, bk = to255(COLOUR_LINK)

        label_lo = (f"link{idx}_lower(bed) "
                    f"/{lo.chain}:{lo.res_seq}({lo.res_name})@{lo.name}"
                    f" z={lo.z:.2f}")
        label_up = (f"link{idx}_upper(overhang) "
                    f"/{up.chain}:{up.res_seq}({up.res_name})@{up.name}"
                    f" z={up.z:.2f}")

        # Each link gets its own marker_set so draw_dual_tubes reads
        # markers[0] and markers[1] independently per link
        lines.append(
            f'  <marker_set name="2chain-Z-link{idx} ({chain1}{lo.res_seq if lo.chain==chain1 else up.res_seq}'
            f'→{chain2}{up.res_seq if up.chain==chain2 else lo.res_seq})">'
        )

        # marker1 — lower Z endpoint (cyan)
        lines.append(
            f'    <marker id="{id_lo}"'
            f' x="{lo.x:.3f}" y="{lo.y:.3f}" z="{lo.z:.3f}"'
            f' red="{rl}" green="{gl}" blue="{bl}"'
            f' radius="{marker_radius:.3f}"'
            f' label="{label_lo}"/>'
        )
        # marker2 — upper Z endpoint (orange)
        lines.append(
            f'    <marker id="{id_up}"'
            f' x="{up.x:.3f}" y="{up.y:.3f}" z="{up.z:.3f}"'
            f' red="{ru}" green="{gu}" blue="{bu}"'
            f' radius="{marker_radius:.3f}"'
            f' label="{label_up}"/>'
        )
        # link rod (yellow-green)
        lines.append(
            f'    <link id1="{id_lo}" id2="{id_up}"'
            f' red="{rk}" green="{gk}" blue="{bk}"'
            f' radius="{link_radius:.3f}"'
            f' label="Z-link{idx}'
            f' XY={p.xy_dist:.2f}A Z={p.z_gap:.2f}A 3D={p.dist_3d:.2f}A'
            f' ({lo.chain}{lo.res_seq}→{up.chain}{up.res_seq})"/>'
        )
        lines.append('  </marker_set>')

    lines.append('</marker_sets>')
    lines.append('')
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"CMM written  → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Find Z-direction inter-chain linkages between two protein chains "
            "in a PDB file for 3D-printing overhang support."
        )
    )
    p.add_argument("pdb",
                   help="Input PDB file path")
    p.add_argument("--chain1", required=True,
                   help="First  chain ID (e.g. A)")
    p.add_argument("--chain2", required=True,
                   help="Second chain ID (e.g. B)")
    p.add_argument("--atoms", type=str, default="CA",
                   help="Comma-separated atom names to consider (default: CA)")
    p.add_argument("--xy_threshold", type=float, default=2.5,
                   help="Max XY distance in Å for vertical alignment (default: 2.5)")
    p.add_argument("--z_min_gap", type=float, default=3.0,
                   help="Min Z separation in Å (default: 3.0)")
    p.add_argument("--z_max_gap", type=float, default=15.0,
                   help="Max Z separation in Å (default: 15.0)")
    p.add_argument("--dist_3d_max", type=float, default=20.0,
                   help="Max 3-D distance cap in Å (default: 20.0)")
    p.add_argument("--dist", type=int, default=10,
                   help=(
                       "Min residue-sequence separation (on its own chain) "
                       "between any two residues used by selected links. "
                       "Applied independently per chain. (default: 10)"
                   ))
    p.add_argument("--no_links", type=int, default=None,
                   help="Keep only the top N shortest links (default: all)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: same directory as PDB)")
    p.add_argument("--marker_radius", type=float, default=0.5,
                   help="CMM sphere radius in Å (default: 0.5)")
    p.add_argument("--link_radius", type=float, default=0.2,
                   help="CMM rod radius in Å (default: 0.2)")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.pdb):
        sys.exit(f"ERROR: File not found: {args.pdb}")

    if args.chain1 == args.chain2:
        sys.exit("ERROR: --chain1 and --chain2 must be different chain IDs.")

    atom_names = {n.strip() for n in args.atoms.split(",")}
    out_dir    = args.output_dir or os.path.dirname(os.path.abspath(args.pdb))
    stem       = os.path.splitext(os.path.basename(args.pdb))[0]

    print(f"\nPDB file         : {args.pdb}")
    print(f"Chains           : {args.chain1}  ↔  {args.chain2}")
    print(f"Atom types       : {', '.join(sorted(atom_names))}")
    print(f"XY threshold     : {args.xy_threshold} Å")
    print(f"Z gap range      : {args.z_min_gap} – {args.z_max_gap} Å")
    print(f"3-D dist cap     : {args.dist_3d_max} Å")
    print(f"Exclusion zone   : ±{args.dist} residues per chain")
    print(f"Top-N selection  : {args.no_links if args.no_links else 'all'}\n")

    # Parse both chains
    atoms1, atoms2 = parse_pdb_two_chains(
        args.pdb, atom_names, args.chain1, args.chain2
    )

    if not atoms1:
        sys.exit(f"ERROR: No matching atoms found for chain {args.chain1}. "
                 f"Check --chain1 and --atoms.")
    if not atoms2:
        sys.exit(f"ERROR: No matching atoms found for chain {args.chain2}. "
                 f"Check --chain2 and --atoms.")

    print(f"Chain {args.chain1} atoms loaded : {len(atoms1)}")
    print(f"Chain {args.chain2} atoms loaded : {len(atoms2)}")
    z_range_summary(atoms1, atoms2, args.chain1, args.chain2)
    print()

    # Find all geometry-passing inter-chain pairs
    pairs = find_z_linkages_2chains(
        atoms1, atoms2,
        xy_threshold=args.xy_threshold,
        z_min_gap=args.z_min_gap,
        z_max_gap=args.z_max_gap,
        dist_3d_max=args.dist_3d_max,
    )

    print(f"Inter-chain candidates (geometry filter) : {len(pairs)}")

    if not pairs:
        print("No candidates found. Try relaxing --xy_threshold, --z_min_gap, "
              "--z_max_gap, or --dist_3d_max.")
        sys.exit(0)

    # Apply top-N greedy selection with exclusion zones
    if args.no_links is not None:
        pairs_before = pairs
        pairs = select_top_links(pairs, args.no_links,
                                 exclusion_dist=args.dist)
        discarded = len(pairs_before) - len(pairs)
        print(f"After --no_links {args.no_links} + exclusion ±{args.dist} res : "
              f"{len(pairs)} kept  (discarded {discarded})")

    # Console table
    title = (f"Top {len(pairs)} inter-chain Z-linkages "
             f"(chain {args.chain1} ↔ {args.chain2}, ranked shortest 3-D first)")
    print_table(pairs, args.chain1, args.chain2, title=title)

    # Write outputs
    base     = f"{stem}_2chain_z_linkages"
    csv_path = os.path.join(out_dir, f"{base}.csv")
    pdb_out  = os.path.join(out_dir, f"{base}.pdb")
    cmm_out  = os.path.join(out_dir, f"{base}.cmm")

    write_csv(pairs, csv_path)
    write_pdb_conect(args.pdb, pairs, pdb_out, args.chain1, args.chain2)
    write_cmm(pairs, cmm_out, args.chain1, args.chain2,
              marker_radius=args.marker_radius,
              link_radius=args.link_radius)

    print(f"\nDone. {len(pairs)} inter-chain Z-support linkage(s) written.\n")


if __name__ == "__main__":
    main()
