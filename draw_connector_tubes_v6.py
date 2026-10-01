#!/usr/bin/env python3
"""
draw_connector_tubes_v6.py
==========================
Draws connector tubes from CMM marker files in ChimeraX.

Changes from v5:
  - Added `thin_shorten_fraction` parameter to customize the starting 
    point of the thin rod in "thin" mode from the m1 (first point) side.
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
from chimerax.core.models import Model, Surface
from chimerax.core.commands import run


# ---------------------------------------------------------------------------
# Dome cap helper
# ---------------------------------------------------------------------------

def _dome_cap_vertices_triangles(centre, axis_dir, radius, segments, rings,
                                  base_index):
    """
    Build a hemispherical dome cap.

    centre    : 3-D point at the centre of the dome's base circle.
    axis_dir  : unit vector pointing *outward* (away from the tube body).
    radius    : dome base radius (== cylinder outer radius).
    segments  : circumferential divisions (matches the cylinder).
    rings     : number of latitude stacks (more → smoother).
    base_index: offset to add to all triangle indices (len of existing verts).

    Returns (vertices, triangles) as lists (not yet np arrays).
    The first `segments` vertices in the returned list are the base ring,
    matching the order used by the cylinder body so they can share edges.
    """
    # Two orthonormal vectors perpendicular to axis_dir
    not_a = np.array([1, 0, 0]) if abs(axis_dir[0]) < 0.9 else np.array([0, 1, 0])
    ortho1 = np.cross(axis_dir, not_a)
    ortho1 /= np.linalg.norm(ortho1)
    ortho2 = np.cross(axis_dir, ortho1)

    verts = []
    tris  = []

    # latitude rings:  ring 0 = base circle (phi=0), ring `rings` = apex (phi=π/2)
    for ring in range(rings + 1):
        phi = (np.pi / 2) * ring / rings          # 0 → π/2
        r   = radius * np.cos(phi)                # shrinks to 0 at apex
        h   = radius * np.sin(phi)                # height above base
        for seg in range(segments):
            angle = 2 * np.pi * seg / segments
            radial = np.cos(angle) * ortho1 + np.sin(angle) * ortho2
            verts.append(centre + axis_dir * h + radial * r)

    # apex point (last vertex)
    apex_idx = base_index + len(verts)
    verts.append(centre + axis_dir * radius)

    # stitch latitude bands
    for ring in range(rings - 1):
        for seg in range(segments):
            next_seg = (seg + 1) % segments
            a = base_index + ring       * segments + seg
            b = base_index + ring       * segments + next_seg
            c = base_index + (ring + 1) * segments + seg
            d = base_index + (ring + 1) * segments + next_seg
            tris.extend([[a, b, d], [a, d, c]])

    # top band connecting to apex
    top_ring_start = base_index + (rings - 1) * segments
    for seg in range(segments):
        next_seg = (seg + 1) % segments
        a = top_ring_start + seg
        b = top_ring_start + next_seg
        tris.append([a, b, apex_idx])

    return verts, tris


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def create_solid_cylinder_geometry(p1, p2, radius, segments=32, cap_rings=8):
    """
    Solid cylinder with smooth hemispherical dome caps at both ends.

    cap_rings : latitude subdivisions for each dome (higher = smoother).
    """
    v = p2 - p1
    length = np.linalg.norm(v)
    if length <= 0:
        return np.array([]), np.array([])
    v_norm = v / length

    not_v = np.array([1, 0, 0]) if abs(v_norm[0]) < 0.9 else np.array([0, 1, 0])
    ortho1 = np.cross(v_norm, not_v)
    ortho1 /= np.linalg.norm(ortho1)
    ortho2 = np.cross(v_norm, ortho1)

    vertices  = []
    triangles = []

    # ---- cylinder body ------------------------------------------------
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        radial_vec = np.cos(angle) * ortho1 + np.sin(angle) * ortho2
        vertices.append(p1 + radius * radial_vec)   # 2*i   : start ring
        vertices.append(p2 + radius * radial_vec)   # 2*i+1 : end ring

    for i in range(segments):
        curr_s = 2 * i
        curr_e = 2 * i + 1
        next_i = (i + 1) % segments
        next_s = 2 * next_i
        next_e = 2 * next_i + 1
        triangles.extend([[curr_s, next_s, curr_e], [next_s, next_e, curr_e]])

    # ---- dome caps ----------------------------------------------------
    d1_verts, d1_tris = _dome_cap_vertices_triangles(
        p1, -v_norm, radius, segments, cap_rings, base_index=len(vertices))
    triangles.extend(d1_tris)
    vertices.extend(d1_verts)

    d2_verts, d2_tris = _dome_cap_vertices_triangles(
        p2, +v_norm, radius, segments, cap_rings, base_index=len(vertices))
    triangles.extend(d2_tris)
    vertices.extend(d2_verts)

    return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int32)


def create_hollow_cylinder_geometry(p1, p2, outer_rad, inner_rad, segments=32,
                                    cap_rings=8, dome_end='p1', flat_end='p2'):
    """
    Hollow cylinder where the *tip* end gets a smooth dome cap and the
    *join* end stays flat (so two half-tubes mate cleanly).

    dome_end : which endpoint gets the dome — 'p1' or 'p2'.
    flat_end : the other endpoint keeps the original flat annular cap.
    """
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

    # ---- cylinder walls (outer + inner) + flat join-end annular cap ----
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        radial_vec = np.cos(angle) * ortho1 + np.sin(angle) * ortho2
        vertices.append(p1 + inner_rad * radial_vec)  # 0: inner_start
        vertices.append(p1 + outer_rad * radial_vec)  # 1: outer_start
        vertices.append(p2 + outer_rad * radial_vec)  # 2: outer_end
        vertices.append(p2 + inner_rad * radial_vec)  # 3: inner_end

    for i in range(segments):
        curr   = i * 4
        next_r = ((i + 1) % segments) * 4
        triangles.extend([[curr+1, next_r+1, curr+2], [next_r+1, next_r+2, curr+2]])  # outer wall
        triangles.extend([[curr,   next_r,   curr+3], [next_r,   next_r+3, curr+3]])  # inner wall
        # flat annular cap at the join end (p2 side)
        triangles.extend([[curr+2, curr+3, next_r+2], [next_r+2, curr+3, next_r+3]])  # p2 flat cap

    # ---- dome cap at the tip end (p1 side by default) ------------------
    dome_axis = -v_norm   # points away from the tube body
    d_verts, d_tris = _dome_cap_vertices_triangles(
        p1, dome_axis, outer_rad, segments, cap_rings, base_index=len(vertices))
    triangles.extend(d_tris)
    vertices.extend(d_verts)

    return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int32)


# ---------------------------------------------------------------------------
# Colour parser — handles both old (r/g/b float) and new (red/green/blue int)
# ---------------------------------------------------------------------------

def _parse_marker_colour(marker_elem):
    """
    Return [R, G, B, 255] as uint8 from a <marker> XML element.
    """
    if marker_elem.get('red') is not None:
        r = int(float(marker_elem.get('red',   '0')))
        g = int(float(marker_elem.get('green', '0')))
        b = int(float(marker_elem.get('blue',  '0')))
    else:
        r = int(float(marker_elem.get('r', '1')) * 255)
        g = int(float(marker_elem.get('g', '1')) * 255)
        b = int(float(marker_elem.get('b', '1')) * 255)

    return [max(0, min(255, c)) for c in (r, g, b)] + [255]


# ---------------------------------------------------------------------------
# Main drawing function
# ---------------------------------------------------------------------------

def draw_dual_tubes(session, cmm_path, outer_rad=1.0, inner_rad=0.5,
                    fraction=0.5, minus_dist=None,
                    reverse_mode="hollow", tolerance=0.03, cap_rings=8,
                    thin_shorten_fraction=0.0):
    """
    Read a CMM file and draw connector tubes for every marker pair.

    All surfaces created by this call are placed as children of a single
    parent Model group named after the CMM file (without extension).

    reverse_mode:
      "solid"  -> One solid cylinder per marker pair (P1 → P2).
      "hollow" -> Two hollow half-tubes meeting in the middle.
      "thin"   -> Primary hollow tube + thin solid rod with rounded domes.

    thin_shorten_fraction:
      Used only in "thin" mode. Fractions between 0.0 and 1.0 control how far 
      back from m1 the thin stick begins. 0.0 means it runs the full length.

    cap_rings : smoothness of the hemispherical dome caps (default 8).
    """
    if not os.path.exists(cmm_path):
        print(f"[draw_dual_tubes] File not found: {cmm_path}")
        return

    group_name = os.path.splitext(os.path.basename(cmm_path))[0]

    tree = ET.parse(cmm_path)
    root = tree.getroot()

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
            v1, t1 = create_solid_cylinder_geometry(m1, m2, outer_rad, cap_rings=cap_rings)
            if v1.size == 0:
                continue
            s1 = Surface(set_name, session)
            s1.set_geometry(v1, None, t1)
            s1.color = np.array(color1, dtype=np.uint8)
            all_surfaces.append(s1)
            continue

        # ---- DUAL MODES: HOLLOW / THIN ----------------------------------
        vec       = m2 - m1
        total_len = np.linalg.norm(vec)
        unit_vec  = vec / total_len

        if minus_dist is not None:
            p1_end = m1 + (unit_vec * max(0, total_len - minus_dist))
        else:
            p1_end = m1 + (vec * fraction)

        v1, t1 = create_hollow_cylinder_geometry(m1, p1_end, outer_rad, inner_rad,
                                                  cap_rings=cap_rings)
        if v1.size > 0:
            s1 = Surface(set_name, session)
            s1.set_geometry(v1, None, t1)
            s1.color = np.array(color1, dtype=np.uint8)
            all_surfaces.append(s1)

        name_rev = f"Reverse_{set_name}"

        if reverse_mode == "hollow":
            if minus_dist is not None:
                p2_end = m2 - (unit_vec * max(0, total_len - minus_dist))
            else:
                p2_end = m2 - (vec * fraction)
            v2, t2 = create_hollow_cylinder_geometry(m2, p2_end, outer_rad, inner_rad,
                                                     cap_rings=cap_rings)

        elif reverse_mode == "thin":
            thin_r  = inner_rad - tolerance
            
            # Linearly interpolate the new starting point based on the fraction
            m1_start = m1 + vec * thin_shorten_fraction
            
            # Recess the body end by thin_r so the dome apex hits exactly on m1_start
            m1_body = m1_start + unit_vec * thin_r
            m2_body = m2 - unit_vec * thin_r
            
            # Ensure safety constraint so extreme shorten values don't reverse geometry
            if np.dot(m2_body - m1_body, unit_vec) > 0:
                v2, t2 = create_solid_cylinder_geometry(m1_body, m2_body, thin_r,
                                                        cap_rings=cap_rings)
            else:
                v2, t2 = np.array([]), np.array([])

        else:
            continue

        if v2.size > 0:
            s2 = Surface(name_rev, session)
            s2.set_geometry(v2, None, t2)
            s2.color = np.array(color2, dtype=np.uint8)
            all_surfaces.append(s2)

    # ------------------------------------------------------------------
    # Group all surfaces under one parent Model and add to session once
    # ------------------------------------------------------------------
    if not all_surfaces:
        print(f"[draw_dual_tubes] No surfaces created for: {cmm_path}")
        return

    group = Model(group_name, session)
    session.models.add([group])
    session.models.add(all_surfaces, parent=group)

    print(f"[draw_dual_tubes] '{group_name}'  →  {len(all_surfaces)} surface(s) grouped.")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

# Example of customizing thin_shorten_fraction to pull back the internal stick:
draw_dual_tubes(session, 'contact_a_gtp.cmm',
                outer_rad=1, inner_rad=0.5, fraction=0.85, 
                reverse_mode="thin", thin_shorten_fraction=0.15)

draw_dual_tubes(session, 'contact_b_gdp.cmm',
                outer_rad=1, inner_rad=0.5, fraction=0.85, 
                reverse_mode="thin", thin_shorten_fraction=0.15)

draw_dual_tubes(session, 'achain_z_linkages.cmm',
                outer_rad=0.8, reverse_mode="solid")

draw_dual_tubes(session, 'achain_frag.cmm',
                outer_rad=0.8, reverse_mode="solid")
                
draw_dual_tubes(session, 'bchain_z_linkages.cmm',
                outer_rad=0.8, reverse_mode="solid")

draw_dual_tubes(session, 'bchain_frag.cmm',
                outer_rad=0.8, reverse_mode="solid")

draw_dual_tubes(session, 'tubulin_2chain_z_linkages_short.cmm',
                fraction=0.55, reverse_mode="thin", tolerance=0.05, 
                outer_rad=1.8, inner_rad=0.9, thin_shorten_fraction=0.15)
                
draw_dual_tubes(session, 'tubulin_2chain_z_linkages_long.cmm',
                fraction=0.7, reverse_mode="thin", tolerance=0.05, 
                outer_rad=1.8, inner_rad=0.9, thin_shorten_fraction=0.18)