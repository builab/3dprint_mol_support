#!/usr/bin/env python3
"""
TODO: Make Reverse another object for easy removal
draw_connector_tubes_v4.py
==========================
Draws connector tubes from CMM marker files in ChimeraX.

Changes from v3:
  - Every draw_dual_tubes() call groups ALL surfaces it creates under a
    single parent Model named after the CMM file stem.  The model tree
    becomes:
        contact_a_gtp          (group)
          ├─ link_0            (Surface)
          ├─ Reverse_link_0    (Surface)
          ├─ link_1            (Surface)
          └─ ...
  - CMM colour parsing now handles BOTH formats:
      old format : r / g / b  attributes (floats  0-1)
      new format : red / green / blue  attributes (ints 0-255)
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
from chimerax.core.models import Model, Surface
from chimerax.core.commands import run


# ---------------------------------------------------------------------------
# Geometry helpers  (unchanged from v3)
# ---------------------------------------------------------------------------

def create_solid_cylinder_geometry(p1, p2, radius, segments=32):
    """Calculates vertices and triangles for a solid cylinder (no hollow core)."""
    v = p2 - p1
    length = np.linalg.norm(v)
    if length <= 0:
        return np.array([]), np.array([])
    v_norm = v / length

    not_v = np.array([1, 0, 0]) if abs(v_norm[0]) < 0.9 else np.array([0, 1, 0])
    ortho1 = np.cross(v_norm, not_v)
    ortho1 /= np.linalg.norm(ortho1)
    ortho2 = np.cross(v_norm, ortho1)

    vertices = [p1, p2]  # indices 0, 1: cap centres
    triangles = []

    for i in range(segments):
        angle = 2 * np.pi * i / segments
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        radial_vec = cos_a * ortho1 + sin_a * ortho2
        vertices.append(p1 + radius * radial_vec)  # start ring
        vertices.append(p2 + radius * radial_vec)  # end ring

    for i in range(segments):
        curr_s = 2 + 2 * i
        curr_e = 2 + 2 * i + 1
        next_i = (i + 1) % segments
        next_s = 2 + 2 * next_i
        next_e = 2 + 2 * next_i + 1
        triangles.extend([[curr_s, next_s, curr_e], [next_s, next_e, curr_e]])
        triangles.append([0, next_s, curr_s])   # start cap
        triangles.append([1, curr_e, next_e])   # end cap

    return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int32)


def create_hollow_cylinder_geometry(p1, p2, outer_rad, inner_rad, segments=32):
    """Calculates vertices and triangles for a hollow cylinder (manifold surface)."""
    v = p2 - p1
    length = np.linalg.norm(v)
    if length <= 0:
        return np.array([]), np.array([])
    v_norm = v / length

    not_v = np.array([1, 0, 0]) if abs(v_norm[0]) < 0.9 else np.array([0, 1, 0])
    ortho1 = np.cross(v_norm, not_v)
    ortho1 /= np.linalg.norm(ortho1)
    ortho2 = np.cross(v_norm, ortho1)

    vertices, triangles = [], []
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        radial_vec = cos_a * ortho1 + sin_a * ortho2
        vertices.append(p1 + inner_rad * radial_vec)  # 0: inner_start
        vertices.append(p1 + outer_rad * radial_vec)  # 1: outer_start
        vertices.append(p2 + outer_rad * radial_vec)  # 2: outer_end
        vertices.append(p2 + inner_rad * radial_vec)  # 3: inner_end

    for i in range(segments):
        curr  = i * 4
        next_r = ((i + 1) % segments) * 4
        triangles.extend([[curr+1, next_r+1, curr+2], [next_r+1, next_r+2, curr+2]])  # outer
        triangles.extend([[curr,   next_r,   curr+3], [next_r,   next_r+3, curr+3]])  # inner
        triangles.extend([[curr,   curr+1,   next_r], [next_r,   curr+1,   next_r+1]])  # start cap
        triangles.extend([[curr+2, curr+3, next_r+2], [next_r+2, curr+3, next_r+3]])  # end cap

    return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int32)


# ---------------------------------------------------------------------------
# Colour parser — handles both old (r/g/b float) and new (red/green/blue int)
# ---------------------------------------------------------------------------

def _parse_marker_colour(marker_elem):
    """
    Return [R, G, B, 255] as uint8 from a <marker> XML element.

    Supports two CMM colour formats:
      New (find_z_linkages scripts):  red="153" green="230" blue="51"   (0-255 ints)
      Old (ChimeraX native):          r="0.6"   g="0.9"    b="0.2"     (0-1 floats)
    """
    # Try new format first (integer attributes red/green/blue)
    if marker_elem.get('red') is not None:
        r = int(float(marker_elem.get('red',   '0')))
        g = int(float(marker_elem.get('green', '0')))
        b = int(float(marker_elem.get('blue',  '0')))
    # Fall back to old format (float attributes r/g/b)
    else:
        r = int(float(marker_elem.get('r', '1')) * 255)
        g = int(float(marker_elem.get('g', '1')) * 255)
        b = int(float(marker_elem.get('b', '1')) * 255)

    # Clamp to valid range
    return [max(0, min(255, c)) for c in (r, g, b)] + [255]


# ---------------------------------------------------------------------------
# Main drawing function
# ---------------------------------------------------------------------------

def draw_dual_tubes(session, cmm_path, outer_rad=1.0, inner_rad=0.5,
                    fraction=0.5, minus_dist=None,
                    reverse_mode="hollow", tolerance=0.1):
    """
    Read a CMM file and draw connector tubes for every marker pair.

    All surfaces created by this call are placed as children of a single
    parent Model group named after the CMM file (without extension).

    reverse_mode:
      "solid"  -> One solid cylinder per marker pair (P1 → P2).
      "hollow" -> Two hollow half-tubes meeting in the middle.
      "thin"   -> Primary hollow tube + thin solid inner tube full-length.

    CMM colour format:
      Accepts both  red/green/blue (0-255 int, new format)
      and           r/g/b          (0-1 float, old ChimeraX format).
    """
    if not os.path.exists(cmm_path):
        print(f"[draw_dual_tubes] File not found: {cmm_path}")
        return

    # Derive a clean group name from the filename
    group_name = os.path.splitext(os.path.basename(cmm_path))[0]

    tree = ET.parse(cmm_path)
    root = tree.getroot()

    # Collect every surface created in this call
    all_surfaces = []

    for i, marker_set in enumerate(root.findall('marker_set')):
        markers = marker_set.findall('marker')
        if len(markers) < 2:
            continue

        m1 = np.array([float(markers[0].get('x')),
                       float(markers[0].get('y')),
                       float(markers[0].get('z'))])
        m2 = np.array([float(markers[1].get('x')),
                       float(markers[1].get('y')),
                       float(markers[1].get('z'))])

        color1 = _parse_marker_colour(markers[0])
        color2 = _parse_marker_colour(markers[1])

        set_name = marker_set.get('name') or f"link_{i}"

        # ---- MODE: SOLID ------------------------------------------------
        if reverse_mode == "solid":
            v1, t1 = create_solid_cylinder_geometry(m1, m2, outer_rad)
            if v1.size == 0:
                continue
            s1 = Surface(set_name, session)
            s1.set_geometry(v1, None, t1)
            s1.color = np.array(color1, dtype=np.uint8)
            all_surfaces.append(s1)
            continue  # no reverse tube in solid mode

        # ---- DUAL MODES: HOLLOW / THIN ----------------------------------
        # colour1 → forward tube  (m1 side, fraction of the way toward m2)
        # colour2 → reverse tube  (m2 side, fraction of the way toward m1)
        # They meet at the midpoint so the join shows a colour boundary.
        vec       = m2 - m1
        total_len = np.linalg.norm(vec)
        unit_vec  = vec / total_len

        if minus_dist is not None:
            p1_end = m1 + (unit_vec * max(0, total_len - minus_dist))
        else:
            p1_end = m1 + (vec * fraction)

        v1, t1 = create_hollow_cylinder_geometry(m1, p1_end, outer_rad, inner_rad)
        if v1.size > 0:
            s1 = Surface(set_name, session)
            s1.set_geometry(v1, None, t1)
            s1.color = np.array(color1, dtype=np.uint8)   # m1-side colour
            all_surfaces.append(s1)

        name_rev = f"Reverse_{set_name}"

        if reverse_mode == "hollow":
            if minus_dist is not None:
                p2_end = m2 - (unit_vec * max(0, total_len - minus_dist))
            else:
                p2_end = m2 - (vec * fraction)
            v2, t2 = create_hollow_cylinder_geometry(m2, p2_end, outer_rad, inner_rad)

        elif reverse_mode == "thin":
            v2, t2 = create_hollow_cylinder_geometry(m2, m1, inner_rad - tolerance, 0.0)

        else:
            continue  # unknown mode — skip reverse tube

        if v2.size > 0:
            s2 = Surface(name_rev, session)
            s2.set_geometry(v2, None, t2)
            s2.color = np.array(color2, dtype=np.uint8)   # m2-side colour  ← always different from s1
            all_surfaces.append(s2)

    # ------------------------------------------------------------------
    # Group all surfaces under one parent Model and add to session once
    # ------------------------------------------------------------------
    if not all_surfaces:
        print(f"[draw_dual_tubes] No surfaces created for: {cmm_path}")
        return

    group = Model(group_name, session)
    session.models.add([group])          # add group first so it gets an ID
    session.models.add(all_surfaces, parent=group)  # then add children

    print(f"[draw_dual_tubes] '{group_name}'  →  {len(all_surfaces)} surface(s) grouped.")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

draw_dual_tubes(session, 'contact_a_gtp.cmm',
                outer_rad=1, inner_rad=0.6, fraction=0.75, reverse_mode="thin")

draw_dual_tubes(session, 'contact_b_gdp.cmm',
                outer_rad=1, inner_rad=0.6, fraction=0.75, reverse_mode="thin")

draw_dual_tubes(session, 'achain_z_linkages.cmm',
                outer_rad=0.7, reverse_mode="solid")

draw_dual_tubes(session, 'achain_frag.cmm',
                outer_rad=0.7, reverse_mode="solid")
                
draw_dual_tubes(session, 'bchain_z_linkages.cmm',
                outer_rad=0.7, reverse_mode="solid")

draw_dual_tubes(session, 'bchain_frag.cmm',
                outer_rad=0.7, reverse_mode="solid")

draw_dual_tubes(session, 'tubulin_2chain_z_linkages.cmm',
                fraction=0.5, reverse_mode="hollow", outer_rad=1, inner_rad=0.6)
