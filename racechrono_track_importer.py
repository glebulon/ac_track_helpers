bl_info = {
    "name": "RaceChrono Track Importer",
    "author": "Autocross Track Builder",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "File > Import > RaceChrono Track (.json)",
    "description": "Import autocross tracks from RaceChrono datalog exports",
    "category": "Import-Export",
}

import bpy
import bmesh
import json
from bpy.props import StringProperty, FloatProperty, BoolProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Vector
import os

class ImportRaceChronoTrack(bpy.types.Operator, ImportHelper):
    """Import RaceChrono track data"""
    bl_idname = "import_scene.racechrono_track"
    bl_label = "Import RaceChrono Track"
    
    # File browser settings
    filename_ext = ".json"
    filter_glob: StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,
    )
    
    # Import options
    track_width: FloatProperty(
        name="Track Width",
        description="Width of the track in meters",
        default=9.0,
        min=3.0,
        max=20.0,
    )
    
    surface_height: FloatProperty(
        name="Surface Thickness",
        description="Thickness of track surface in meters",
        default=0.1,
        min=0.01,
        max=1.0,
    )
    
    import_cones: BoolProperty(
        name="Import Turn Markers",
        description="Create cone objects at detected turn points",
        default=True,
    )
    
    curve_resolution: EnumProperty(
        name="Curve Resolution",
        description="Resolution of the track curve",
        items=[
            ('LOW', "Low (6)", "Low resolution for faster preview"),
            ('MEDIUM', "Medium (12)", "Medium resolution - good balance"),
            ('HIGH', "High (24)", "High resolution for final output"),
        ],
        default='MEDIUM',
    )
    
    create_ai_line: BoolProperty(
        name="Create AI Racing Line",
        description="Create a separate curve for AI racing line",
        default=True,
    )
    
    def execute(self, context):
        try:
            # Load track data
            with open(self.filepath, 'r') as f:
                track_data = json.load(f)
            
            # Clear existing selection
            bpy.ops.object.select_all(action='DESELECT')
            
            # Create materials first
            asphalt_material = self.create_asphalt_material()
            cone_material = self.create_cone_material()
            
            # Import racing line
            racing_line_curve = self.create_racing_line_curve(
                track_data['racing_line'], 
                track_data['metadata']
            )
            
            # Create track surface
            track_surface = self.create_track_surface(
                racing_line_curve, 
                asphalt_material
            )
            
            # Import turn markers if requested
            if self.import_cones and 'turn_points' in track_data:
                self.create_turn_markers(track_data['turn_points'], cone_material)
            
            # Create AI line if requested
            if self.create_ai_line:
                ai_line = self.create_ai_racing_line(track_data['racing_line'])
            
            # Center view
            bpy.ops.view3d.view_all()
            
            # Report success
            self.report({'INFO'}, 
                       f"Imported track: {track_data['metadata']['racing_line_analysis']['track_length_meters']:.1f}m, "
                       f"{len(track_data.get('turn_points', []))} turns")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to import track: {str(e)}")
            return {'CANCELLED'}
    
    def create_asphalt_material(self):
        """Create realistic asphalt material"""
        mat_name = "Asphalt_Autocross"
        
        # Remove existing material if it exists
        if mat_name in bpy.data.materials:
            bpy.data.materials.remove(bpy.data.materials[mat_name])
        
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
        # Clear default nodes
        mat.node_tree.nodes.clear()
        
        # Create nodes
        bsdf = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
        output = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
        
        # Set material properties for asphalt
        bsdf.inputs['Base Color'].default_value = (0.12, 0.12, 0.12, 1.0)  # Dark gray
        bsdf.inputs['Roughness'].default_value = 0.9
        bsdf.inputs['Specular'].default_value = 0.1
        
        # Add some texture variation
        noise = mat.node_tree.nodes.new(type='ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 50.0
        noise.inputs['Detail'].default_value = 15.0
        noise.inputs['Roughness'].default_value = 0.5
        
        # Mix with base color
        mix = mat.node_tree.nodes.new(type='ShaderNodeMixRGB')
        mix.blend_type = 'MULTIPLY'
        mix.inputs['Fac'].default_value = 0.1
        mix.inputs['Color1'].default_value = (0.12, 0.12, 0.12, 1.0)
        mix.inputs['Color2'].default_value = (0.08, 0.08, 0.08, 1.0)
        
        # Connect nodes
        mat.node_tree.links.new(noise.outputs['Fac'], mix.inputs['Fac'])
        mat.node_tree.links.new(mix.outputs['Color'], bsdf.inputs['Base Color'])
        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        # Position nodes for organization
        bsdf.location = (0, 0)
        output.location = (300, 0)
        noise.location = (-600, 0)
        mix.location = (-300, 0)
        
        return mat
    
    def create_cone_material(self):
        """Create orange cone material"""
        mat_name = "Cone_Orange"
        
        if mat_name in bpy.data.materials:
            return bpy.data.materials[mat_name]
        
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
        # Set bright orange color
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Base Color'].default_value = (1.0, 0.4, 0.05, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.3
        bsdf.inputs['Specular'].default_value = 0.8
        
        return mat
    
    def create_racing_line_curve(self, racing_line_points, metadata):
        """Create the main racing line curve"""
        curve_data = bpy.data.curves.new(name="Racing_Line", type='CURVE')
        curve_data.dimensions = '3D'
        
        # Set resolution based on user choice
        resolution_map = {'LOW': 6, 'MEDIUM': 12, 'HIGH': 24}
        curve_data.resolution_u = resolution_map[self.curve_resolution]
        
        # Create spline
        spline = curve_data.splines.new('BEZIER')
        spline.bezier_points.add(len(racing_line_points) - 1)
        
        # Set points with elevation (flat for autocross)
        for i, point in enumerate(racing_line_points):
            bp = spline.bezier_points[i]
            bp.co = Vector((point['x'], point['y'], 0))
            bp.handle_left_type = 'AUTO'
            bp.handle_right_type = 'AUTO'
        
        # Create object
        curve_obj = bpy.data.objects.new("Racing_Line", curve_data)
        bpy.context.collection.objects.link(curve_obj)
        
        return curve_obj
    
    def create_track_surface(self, racing_line_curve, material):
        """Create track surface from racing line"""
        # Select the curve
        bpy.context.view_layer.objects.active = racing_line_curve
        racing_line_curve.select_set(True)
        
        # Duplicate for track surface
        bpy.ops.object.duplicate()
        track_surface = bpy.context.active_object
        track_surface.name = "Track_Surface"
        
        # Add bevel to create width
        track_surface.data.bevel_depth = self.track_width / 2
        track_surface.data.bevel_resolution = 4
        track_surface.data.fill_mode = 'BOTH'
        
        # Convert to mesh for better control
        bpy.ops.object.convert(target='MESH')
        
        # Add surface thickness
        solidify = track_surface.modifiers.new(name="Solidify", type='SOLIDIFY')
        solidify.thickness = -self.surface_height
        solidify.offset = 1.0
        
        # Add subdivision for smoother surface
        subsurf = track_surface.modifiers.new(name="Subdivision", type='SUBSURF')
        subsurf.levels = 1
        
        # Apply material
        track_surface.data.materials.append(material)
        
        # Move slightly down to ground level
        track_surface.location.z = -self.surface_height / 2
        
        return track_surface
    
    def create_turn_markers(self, turn_points, material):
        """Create cone markers at turn points"""
        cone_collection = bpy.data.collections.new("Turn_Cones")
        bpy.context.scene.collection.children.link(cone_collection)
        
        for i, turn in enumerate(turn_points):
            # Create cone
            bpy.ops.mesh.primitive_cone_add(
                radius1=0.25,
                radius2=0.05,
                depth=0.6,
                location=(turn['x'], turn['y'], 0.3)
            )
            
            cone = bpy.context.active_object
            cone.name = f"Cone_{i+1:02d}"
            
            # Move to cone collection
            bpy.context.scene.collection.objects.unlink(cone)
            cone_collection.objects.link(cone)
            
            # Apply material
            cone.data.materials.append(material)
            
            # Add slight random rotation for realism
            import random
            cone.rotation_euler.z = random.uniform(-0.1, 0.1)
    
    def create_ai_racing_line(self, racing_line_points):
        """Create AI racing line curve for Assetto Corsa export"""
        curve_data = bpy.data.curves.new(name="AI_Racing_Line", type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 12
        
        # Create spline
        spline = curve_data.splines.new('POLY')
        spline.points.add(len(racing_line_points) - 1)
        
        # Set points (POLY spline uses 4D coordinates)
        for i, point in enumerate(racing_line_points):
            spline.points[i].co = (point['x'], point['y'], 0.05, 1)  # Slightly above track
        
        # Create object
        curve_obj = bpy.data.objects.new("AI_Racing_Line", curve_data)
        bpy.context.collection.objects.link(curve_obj)
        
        # Set display properties
        curve_obj.show_in_front = True
        
        # Create material for AI line
        ai_mat = bpy.data.materials.new(name="AI_Line_Material")
        ai_mat.use_nodes = True
        bsdf = ai_mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Base Color'].default_value = (0.0, 1.0, 0.0, 1.0)  # Bright green
        bsdf.inputs['Emission'].default_value = (0.0, 0.5, 0.0, 1.0)
        
        curve_obj.data.materials.append(ai_mat)
        
        return curve_obj

# Panel for track import settings
class RACECHRONO_PT_TrackImportPanel(bpy.types.Panel):
    """Panel for RaceChrono track import settings"""
    bl_label = "RaceChrono Track Import"
    bl_idname = "RACECHRONO_PT_track_import"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "RaceChrono"
    
    def draw(self, context):
        layout = self.layout
        
        col = layout.column()
        col.label(text="Import RaceChrono Track Data:")
        col.operator("import_scene.racechrono_track", text="Import Track JSON")
        
        col.separator()
        col.label(text="Workflow:")
        col.label(text="1. Process datalog with Python script")
        col.label(text="2. Import JSON file here")
        col.label(text="3. Export to Assetto Corsa")

# Menu integration
def menu_func_import(self, context):
    self.layout.operator(ImportRaceChronoTrack.bl_idname, text="RaceChrono Track (.json)")

# Registration
classes = [
    ImportRaceChronoTrack,
    RACECHRONO_PT_TrackImportPanel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

if __name__ == "__main__":
    register()