import bpy
from .helpers import *

class MyProperties(bpy.types.PropertyGroup):

    distance_from_camera : bpy.props.FloatProperty(name= "Camera Distance"
        , default= InitVal.distance_from_camera
        , update= update_scene
        , soft_min= 1.0E-2
        , min= 1.0E-2
        , soft_max= 100
        , max= 100)
    fov : bpy.props.FloatProperty(name= "FOV"
        , default= InitVal.fov
        , update= update_scene
        , soft_min= 1.0E-3
        , min= 1.0E-3
        , soft_max= pi
        , max= pi)
    angleRes : bpy.props.IntProperty(name= "Angle Resolution"
        , default= InitVal.angleRes
        , update= update_scene
        , step= 1
        , soft_min= 2
        , min= 2
        , soft_max= 32
        , max= 32)
    max_resolution : bpy.props.IntProperty(name= "Maximum Resolution"
        , default= InitVal.max_resolution
        , update= update_scene
        , soft_min= 100
        , min= 100
        , step= 1
        , soft_max= 2048
        , max= 2048)
    imgName: bpy.props.StringProperty(name="File Name"
        , default= InitVal.imgName
        , update= when_imgName_update)

    # filePath: bpy.props.StringProperty(name="File Path"
    #     , default= InitVal.current_path
    #     , update= update_scene)

    attn_width : bpy.props.FloatProperty(name= "Layers Width"
        , default= InitVal.attn_width
        , update= update_scene
        , soft_min= 1.0E-2
        , min= 1.0E-2
        , soft_max= 100
        , max= 100)
    attn_height : bpy.props.FloatProperty(name= "Layers Height"
        , default= InitVal.attn_height
        , update= update_scene
        , soft_min= 1.0E-2
        , min= 1.0E-2
        , soft_max= 100
        , max= 100)
    attn_thick : bpy.props.FloatProperty(name= "Layers Thickness"
        , default= InitVal.attn_thick
        , update= update_scene
        , soft_min= 1.0E-3
        , min= 1.0E-3
        , soft_max= 20
        , max= 20)
    attn_layers : bpy.props.IntProperty(name= "Layers Counts"
        , default= InitVal.attn_layers
        , update= update_scene
        , soft_min= 1
        , min= 1
        , soft_max= 20
        , max= 20)
    attn_z : bpy.props.FloatProperty(name= "Layers distance (x-axis)"
        , default= InitVal.attn_z
        , update= update_scene
        , soft_min= -10
        , min= -10
        , soft_max= 10
        , max= 10)
    camera_name : bpy.props.StringProperty(name="camera_name"
        , default= "main_camera")
    fov_name : bpy.props.StringProperty(name="fov_name"
        , default= "fov_bounding_box")
    attn_name : bpy.props.StringProperty(name="attn_name"
        , default= "attenuation_layer_bounding_box")
    current_path : bpy.props.StringProperty(name="current_path"
        , default= r"C:\\"
        , update= when_current_path_update)

    select_config : bpy.props.EnumProperty(name= "Select Configuration"
        , items= select_config_callback
        , update= when_select_config_update)
    

    real_width : bpy.props.FloatProperty(name= "Real Width"
        , default= InitVal.attn_width
        , update= lambda self, context: update_realcalc(self, context, 'real_width')
        , soft_min= 0
        , min= 0
        , soft_max= 10**4
        , max= 10**4)
    real_height : bpy.props.FloatProperty(name= "Real Height"
        , default= InitVal.attn_height
        , update= lambda self, context: update_realcalc(self, context, 'real_height')
        , soft_min= 0
        , min= 0
        , soft_max= 10**4
        , max= 10**4)
    real_thick : bpy.props.FloatProperty(name= "Real Thickness"
        , default= InitVal.attn_thick
        , update= lambda self, context: update_realcalc(self, context, 'real_thick')
        , soft_min= 0
        , min= 0
        , soft_max= 10**4
        , max= 10**4)

    current_setting : bpy.props.StringProperty(name="Setting Save As"
        , default= 'default'
        , update= when_current_settings_update)



classes = [MyProperties]
def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.my_variables = bpy.props.PointerProperty(type= MyProperties)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)