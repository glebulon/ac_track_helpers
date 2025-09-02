import bpy
import bmesh
import json
import os
from mathutils import Vector

# --- Static Script for Blender 3.6+ ---
# This script reads 'track_data.json' from the output directory.

def load_track_data():
    """Load track data from the JSON file."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Check for an 'output' directory relative to the blend file
    output_dir = os.path.join(script_dir, 'output')
    json_path = os.path.join(output_dir, 'track_data.json')

    if not os.path.exists(json_path):
        # Fallback to checking the same directory as the blend file
        json_path_fallback = os.path.join(script_dir, 'track_data.json')
        if os.path.exists(json_path_fallback):
            json_path = json_path_fallback
        else:
            print(f"Error: Cannot find track_data.json in {output_dir} or {script_dir}")
            return None

    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Successfully loaded track data from {json_path}")
    return data

def clear_scene():
    """Clear existing mesh objects."""
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    print("Cleared existing objects")

def create_track_mesh(name, vertices, faces, material):
    """Generic mesh creation helper."""
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    bm = bmesh.new()
    verts = [bm.verts.new(v) for v in vertices]
    for f_indices in faces:
        try:
            bm.faces.new([verts[i] for i in f_indices])
        except ValueError: # Handles degenerate faces
            pass
    
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    
    if material:
        obj.data.materials.append(material)
    return obj

def create_asphalt_material():
    """Create a basic asphalt material."""
    mat = bpy.data.materials.new(name="AsphaltMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.08, 0.08, 0.08, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.9
    return mat

def create_racing_line_curve(name, points, material):
    """Create a curve object for the racing line."""
    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'
    spline = curve_data.splines.new('NURBS')
    spline.points.add(len(points) - 1)
    for i, p in enumerate(points):
        spline.points[i].co = (p['x'], p['y'], p['z'] + 0.05, 1)
    
    curve_data.bevel_depth = 0.1
    curve_data.fill_mode = 'FULL'
    obj = bpy.data.objects.new(name, curve_data)
    if material:
        obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)
    return obj

def create_line_material(name, color, emission_strength=0.0):
    """Create a simple colored material."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color
        if emission_strength > 0:
            bsdf.inputs['Emission'].default_value = color
            bsdf.inputs['Emission Strength'].default_value = emission_strength
    return mat

def main():
    """Main execution function for Blender script."""
    TRACK_DATA = load_track_data()
    if not TRACK_DATA:
        print("Aborting script.")
        return

    clear_scene()
    
    # Prepare data
    left_verts = [(p['x'], p['y'], p['z']) for p in TRACK_DATA['leftEdge']]
    right_verts = [(p['x'], p['y'], p['z']) for p in TRACK_DATA['rightEdge']]
    track_verts = left_verts + right_verts
    
    num_pts = len(TRACK_DATA['centerline'])
    track_faces = []
    for i in range(num_pts - 1):
        l0, l1 = i, i + 1
        r0, r1 = i + num_pts, i + 1 + num_pts
        track_faces.append([l0, r0, r1, l1])

    # Create objects
    asphalt_mat = create_asphalt_material()
    track_obj = create_track_mesh("AutocrossTrack", track_verts, track_faces, asphalt_mat)
    
    line_mat = create_line_material("RacingLineMaterial", (1, 0.8, 0, 1), emission_strength=2.0)
    create_racing_line_curve("RacingLine", TRACK_DATA['centerline'], line_mat)
    
    # Create start/finish line
    start_point = TRACK_DATA['centerline'][0]
    bpy.ops.mesh.primitive_plane_add(
        size=TRACK_DATA['width'], 
        location=(start_point['x'], start_point['y'], 0.01)
    )
    sf_line = bpy.context.active_object
    sf_line.name = "StartFinishLine"
    sf_line.data.materials.append(create_line_material("StartFinishMaterial", (1,1,1,1)))

    print("\n--- Autocross Track Generation Complete ---")
    metadata = TRACK_DATA['metadata']
    print(f"Track Length: {metadata['length']:.1f}m")
    print(f"Max Speed: {metadata['maxSpeed']:.1f} km/h")
    print(f"Run Time: {metadata['runTime']:.1f}s")
    print("\nNext steps: Add cones, export to FBX for Assetto Corsa.")

if __name__ == "__main__":
    main()
