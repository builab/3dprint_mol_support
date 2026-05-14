#!/usr/bin/env python3

# Need to group everything in 1 group per draw

import os
import numpy as np
import xml.etree.ElementTree as ET
from chimerax.core.models import Surface
from chimerax.core.commands import run

def create_solid_cylinder_geometry(p1, p2, radius, segments=32):
    """Calculates vertices and triangles for a solid cylinder (no hollow core)."""
    v = p2 - p1
    length = np.linalg.norm(v)
    if length <= 0: return np.array([]), np.array([])
    v_norm = v / length
    
    # Basis vectors for the cylinder cross-section
    not_v = np.array([1, 0, 0]) if abs(v_norm[0]) < 0.9 else np.array([0, 1, 0])
    ortho1 = np.cross(v_norm, not_v)
    ortho1 /= np.linalg.norm(ortho1)
    ortho2 = np.cross(v_norm, ortho1)

    vertices = [p1, p2] # Indices 0 and 1 (centers of the caps)
    triangles = []

    for i in range(segments):
        angle = 2 * np.pi * i / segments
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        radial_vec = cos_a * ortho1 + sin_a * ortho2
        vertices.append(p1 + radius * radial_vec) # Start ring
        vertices.append(p2 + radius * radial_vec) # End ring

    for i in range(segments):
        # Vertex mapping: 0=P1 center, 1=P2 center
        # 2+2i = start ring, 2+2i+1 = end ring
        curr_s = 2 + 2*i
        curr_e = 2 + 2*i + 1
        next_i = (i + 1) % segments
        next_s = 2 + 2*next_i
        next_e = 2 + 2*next_i + 1
        
        # Sides (two triangles per segment)
        triangles.extend([[curr_s, next_s, curr_e], [next_s, next_e, curr_e]])
        # Start Cap (Triangle fan piece from P1 center)
        triangles.append([0, next_s, curr_s])
        # End Cap (Triangle fan piece from P2 center)
        triangles.append([1, curr_e, next_e])

    return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int32)

def create_hollow_cylinder_geometry(p1, p2, outer_rad, inner_rad, segments=32):
    """Calculates vertices and triangles for a hollow cylinder (manifold surface)."""
    v = p2 - p1
    length = np.linalg.norm(v)
    if length <= 0: return np.array([]), np.array([])
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
        vertices.append(p1 + inner_rad * radial_vec) # 0: inner_start
        vertices.append(p1 + outer_rad * radial_vec) # 1: outer_start
        vertices.append(p2 + outer_rad * radial_vec) # 2: outer_end
        vertices.append(p2 + inner_rad * radial_vec) # 3: inner_end

    for i in range(segments):
        curr = i * 4
        next_r = ((i + 1) % segments) * 4
        triangles.extend([[curr+1, next_r+1, curr+2], [next_r+1, next_r+2, curr+2]]) # Outer
        triangles.extend([[curr, next_r, curr+3], [next_r, next_r+3, curr+3]])       # Inner
        triangles.extend([[curr, curr+1, next_r], [next_r, curr+1, next_r+1]])       # Start Cap
        triangles.extend([[curr+2, curr+3, next_r+2], [next_r+2, curr+3, next_r+3]]) # End Cap

    return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int32)

def draw_dual_tubes(session, cmm_path, outer_rad=1.0, inner_rad=0.5, 
                    fraction=0.5, minus_dist=None, 
                    reverse_mode="hollow", tolerance=0.1):
    """
    reverse_mode: 
      "solid"  -> Single solid cylinder from P1 to P2. No second part.
      "hollow" -> Two hollow tubes (P1-mid and P2-mid).
      "thin"   -> Primary hollow tube + one thin solid inner tube.
    """
    if not os.path.exists(cmm_path): return

    tree = ET.parse(cmm_path)
    root = tree.getroot()

    for i, marker_set in enumerate(root.findall('marker_set')):
        markers = marker_set.findall('marker')
        if len(markers) < 2: continue
            
        m1 = np.array([float(markers[0].get('x')), float(markers[0].get('y')), float(markers[0].get('z'))])
        m2 = np.array([float(markers[1].get('x')), float(markers[1].get('y')), float(markers[1].get('z'))])
        
        color1 = [int(float(markers[0].get(c)) * 255) for c in ['r', 'g', 'b']] + [255]
        
        # --- MODE: SOLID ---
        if reverse_mode == "solid":
            # Draw one continuous cylinder from marker 1 to marker 2
            v1, t1 = create_solid_cylinder_geometry(m1, m2, outer_rad)
            s1 = Surface(marker_set.get('name') or f"solid_link_{i}", session) 
            s1.set_geometry(v1, None, t1)
            s1.color = np.array(color1, dtype=np.uint8)
            session.models.add([s1])
            continue # Move to next marker set, skip reverse tube logic

        # --- DUAL MODES (Hollow or Thin) ---
        vec = m2 - m1
        total_len = np.linalg.norm(vec)
        unit_vec = vec / total_len
        
        if minus_dist is not None:
            p1_end = m1 + (unit_vec * max(0, total_len - minus_dist))
        else:
            p1_end = m1 + (vec * fraction)
            
        v1, t1 = create_hollow_cylinder_geometry(m1, p1_end, outer_rad, inner_rad)
        s1 = Surface(marker_set.get('name') or f"contact_{i}", session) 
        s1.set_geometry(v1, None, t1)
        s1.color = np.array(color1, dtype=np.uint8)
        session.models.add([s1])

        name_rev = f"Reverse_{marker_set.get('name') or f'contact_{i}'}"
        color2 = [int(float(markers[1].get(c)) * 255) for c in ['r', 'g', 'b']] + [255]
        
        if reverse_mode == "hollow":
            if minus_dist is not None:
                p2_end = m2 - (unit_vec * max(0, total_len - minus_dist))
            else:
                p2_end = m2 - (vec * fraction)
            v2, t2 = create_hollow_cylinder_geometry(m2, p2_end, outer_rad, inner_rad)
        
        elif reverse_mode == "thin":
            v2, t2 = create_hollow_cylinder_geometry(m2, m1, inner_rad - tolerance, 0.0)

        s2 = Surface(name_rev, session)
        s2.set_geometry(v2, None, t2)
        s2.color = np.array(color2, dtype=np.uint8)
        session.models.add([s2])

# --- EXECUTION EXAMPLES ---

# NEW SOLID MODE: Just a simple cylinder between markers
# For Reverse Option 2: Thin solid tube going full length
draw_dual_tubes(session, 'contact_a_gtp.cmm', outer_rad=1, inner_rad=0.6, fraction=0.75, reverse_mode="thin")

draw_dual_tubes(session, 'contact_b_gdp.cmm', outer_rad=1, inner_rad=0.6, fraction=0.75, reverse_mode="thin")

draw_dual_tubes(session, 'achain_z_linkages.cmm', outer_rad=0.7, reverse_mode="solid")

draw_dual_tubes(session, 'bchain_z_linkages.cmm', outer_rad=0.7, reverse_mode="solid")

draw_dual_tubes(session, 'achain_frag.cmm', outer_rad=0.7, reverse_mode="solid")

# Original Hollow Mode
draw_dual_tubes(session, 'tubulin_2chain_z_linkages.cmm', fraction=0.5, reverse_mode="hollow", outer_rad=1, inner_rad=0.6)