#!/usr/bin/env python3
"""
add_linkage.py

Add physical support linkages to a protein PDB for 3D printing.
Outputs a ChimeraX .cmm marker file with individual marker_sets per link.
"""

import argparse
import math
import sys
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# DATA OBJECTS
# ─────────────────────────────────────────────────────────────────────────────

class Atom:
    __slots__ = ("serial", "name", "res_name", "chain_id", "res_seq", "x", "y", "z")
    def __init__(self, serial, name, res_name, chain_id, res_seq, x, y, z):
        self.serial   = serial
        self.name     = name.strip()
        self.res_name = res_name.strip()
        self.chain_id = chain_id
        self.res_seq  = res_seq
        self.x, self.y, self.z = x, y, z
    def __repr__(self):
        return f"{self.chain_id}:{self.res_name}{self.res_seq}({self.name})"

class PseudoAtom:
    __slots__ = ("chain_id", "res_seq", "res_name", "name", "x", "y", "z")
    def __init__(self, chain_id, res_seq, res_name, name, x, y, z):
        self.chain_id = chain_id
        self.res_seq  = res_seq
        self.res_name = res_name
        self.name     = name
        self.x, self.y, self.z = x, y, z
    def __repr__(self):
        return f"{self.chain_id}:{self.res_name}{self.res_seq}({self.name})"

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY & PARSING
# ─────────────────────────────────────────────────────────────────────────────

def dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def midpoint_xyz(a, b):
    return ((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)

def parse_pdb(filepath):
    atoms = []
    with open(filepath) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    atoms.append(Atom(
                        int(line[6:11]), line[12:16], line[17:20], line[21],
                        int(line[22:26]), float(line[30:38]), float(line[38:46]), float(line[46:54])
                    ))
                except ValueError: continue
    return atoms

def get_backbone_atoms(atoms):
    bb = defaultdict(dict)
    for a in atoms:
        if a.name in ("CA", "C"): bb[(a.chain_id, a.res_seq)][a.name] = a
    return bb

def group_by_chain(backbone):
    chains = defaultdict(dict)
    for (ch, seq), d in backbone.items(): chains[ch][seq] = d
    return {ch: sorted(d.items()) for ch, d in chains.items()}

def build_pseudo_ca_atoms(chains_dict):
    pseudo = {}
    for chain_id, residues in chains_dict.items():
        nres = len(residues)
        for idx, (res_seq, atom_dict) in enumerate(residues):
            pts = []
            if idx > 0 and "CA" in residues[idx-1][1]: pts.append(residues[idx-1][1]["CA"])
            if "CA" in atom_dict: pts.append(atom_dict["CA"])
            if idx < nres - 1 and "CA" in residues[idx+1][1]: pts.append(residues[idx+1][1]["CA"])
            if "C" in atom_dict: pts.append(atom_dict["C"])
            if not pts: continue
            avg_x = sum(p.x for p in pts) / len(pts)
            avg_y = sum(p.y for p in pts) / len(pts)
            avg_z = sum(p.z for p in pts) / len(pts)
            pseudo[(chain_id, res_seq)] = PseudoAtom(chain_id, res_seq, pts[0].res_name, "PS", avg_x, avg_y, avg_z)
    return pseudo

# ─────────────────────────────────────────────────────────────────────────────
# SELECTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def chain_terminal_keys(chains_dict, window=2):
    prot = set()
    for cid, res in chains_dict.items():
        seqs = [r for r, _ in res]
        if seqs:
            for r in seqs:
                if r <= seqs[0] + window or r >= seqs[-1] - window: prot.add((cid, r))
    return prot

def collect_all_fragments(chains_dict, pseudo_atoms):
    all_frags = []
    for cid, res in chains_dict.items():
        if not res: continue
        curr = [res[0]]
        for i in range(1, len(res)):
            if (res[i][0] - res[i-1][0]) > 1 or dist(res[i][1]["CA"], res[i-1][1]["CA"]) > 4.0:
                all_frags.append({"chain": cid, "ca_list": [pseudo_atoms[k] for r, d in curr if (k := (cid, r)) in pseudo_atoms]})
                curr = []
            curr.append(res[i])
        all_frags.append({"chain": cid, "ca_list": [pseudo_atoms[k] for r, d in curr if (k := (cid, r)) in pseudo_atoms]})
    return all_frags

def select_stabilization_links(chains_dict, pseudo_atoms, no_link, min_dist, terminal_keys):
    ca_atoms = [(cid, rs, pseudo_atoms[(cid, rs)]) for cid, res in chains_dict.items() 
                for rs, d in res if (cid, rs) in pseudo_atoms and (cid, rs) not in terminal_keys]
    cands = []
    for i in range(len(ca_atoms)):
        for j in range(i + 1, len(ca_atoms)):
            ci, ri, ai = ca_atoms[i]; cj, rj, aj = ca_atoms[j]
            d = dist(ai, aj)
            if (ci == cj and abs(rj - ri) < 8) or d > 20.0: continue
            score = d - 0.5 * min(abs(rj-ri) if ci==cj else 9999, 50)
            cands.append((score, d, ai, aj))
    cands.sort(key=lambda x: x[0])
    chosen, chosen_mp = [], []
    for _, d, ai, aj in cands:
        if len(chosen) >= no_link: break
        mp = midpoint_xyz(ai, aj)
        if any(math.dist(mp, e) < min_dist for e in chosen_mp): continue
        chosen.append((ai, aj, d)); chosen_mp.append(mp)
    return chosen

def find_fragment_anchor_links(all_frags, chains_dict, pseudo_atoms, no_link, no_frag_link, terminal_keys, min_dist):
    frag_cands = []
    for i, fi in enumerate(all_frags):
        for ai in fi["ca_list"]:
            if (ai.chain_id, ai.res_seq) in terminal_keys: continue
            for j, fj in enumerate(all_frags[i+1:], i+1):
                for aj in fj["ca_list"]:
                    if (aj.chain_id, aj.res_seq) in terminal_keys: continue
                    d = dist(ai, aj)
                    if d > 20.0 or (ai.chain_id == aj.chain_id and abs(ai.res_seq - aj.res_seq) < 8): continue
                    frag_cands.append((d, ai, aj))
    frag_cands.sort(key=lambda x: x[0])
    selected, selected_mp, used = [], [], set()
    budget = min(no_link, no_frag_link)
    for d, ai, aj in frag_cands:
        if len(selected) >= budget: break
        pair = tuple(sorted([(ai.chain_id, ai.res_seq), (aj.chain_id, aj.res_seq)]))
        mp = midpoint_xyz(ai, aj)
        if pair in used or any(math.dist(mp, e) < min_dist for e in selected_mp): continue
        selected.append((ai, aj, d)); selected_mp.append(mp); used.add(pair)
    
    rem = no_link - len(selected)
    if rem > 0:
        extra = select_stabilization_links(chains_dict, pseudo_atoms, rem, min_dist, terminal_keys)
        for ai, aj, d in extra:
            pair = tuple(sorted([(ai.chain_id, ai.res_seq), (aj.chain_id, aj.res_seq)]))
            if pair not in used: selected.append((ai, aj, d)); used.add(pair)
    return selected[:no_link]

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT & REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def report_links(links):
    if not links:
        print("    No links placed.")
        return
    print(f"    {'Atom A':<30} {'Atom B':<30} {'Dist':>7}")
    print("    " + "─" * 70)
    for a, b, d in links:
        print(f"    {str(a):<30} {str(b):<30} {d:>7.2f} Å")

def write_cmm(output_path, frag_links, stab_links):
    lines = ['<?xml version="1.0" encoding="utf-8"?>', '<marker_sets>']
    
    def add_set(links, prefix, col_m):
        for i, (a, b, _) in enumerate(links, 1):
            lines.append(f'  <marker_set name="{prefix}_{i}">')
            for mid, at in enumerate((a, b), 1):
                note = f"{at.chain_id}:{at.res_name}{at.res_seq}({at.name})"
                lines.append(f'    <marker id="{mid}" x="{at.x:.3f}" y="{at.y:.3f}" z="{at.z:.3f}" r="{col_m[0]}" g="{col_m[1]}" b="{col_m[2]}" radius="1.0" note="{note}"/>')
            lines.append(f'    <link id1="1" id2="2" r="1.0" g="1.0" b="1.0" radius="0.400"/>')
            lines.append('  </marker_set>')

    add_set(frag_links, "frag_link", (1.0, 0.55, 0.0))
    add_set(stab_links, "stab_link", (0.2, 0.65, 1.0))
    lines.append('</marker_sets>')
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(frag_links), len(stab_links)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_input")
    parser.add_argument("cmm_output")
    parser.add_argument("--no_link", type=int, default=6)
    parser.add_argument("--no_frag_link", type=int, default=3)
    parser.add_argument("--frag_chain", action="store_true")
    parser.add_argument("--stabilize", action="store_true")
    parser.add_argument("--dist", type=float, default=15.0)
    args = parser.parse_args()

    if not args.frag_chain and not args.stabilize:
        sys.exit("ERROR: Specify --frag_chain or --stabilize")

    print(f"\n[1] Parsing {args.pdb_input}...")
    atoms = parse_pdb(args.pdb_input)
    chains = group_by_chain(get_backbone_atoms(atoms))
    pseudo_atoms = build_pseudo_ca_atoms(chains)
    terminals = chain_terminal_keys(chains)
    fl, sl = [], []

    if args.frag_chain:
        print(f"\n[2] Fragment-priority mode (Budget: {args.no_frag_link} frag links)")
        frags = collect_all_fragments(chains, pseudo_atoms)
        fl = find_fragment_anchor_links(frags, chains, pseudo_atoms, args.no_link, args.no_frag_link, terminals, args.dist)
        report_links(fl)
    
    if args.stabilize:
        print("\n[3] Global stabilization mode...")
        sl = select_stabilization_links(chains, pseudo_atoms, args.no_link, args.dist, terminals)
        report_links(sl)

    nf, ns = write_cmm(args.cmm_output, fl, sl)
    print(f"\n[✓] Written: {args.cmm_output}")
    print(f"    Final Count: {nf} fragment-priority links, {ns} stabilization links.")

if __name__ == "__main__":
    main()