import bpy
from .helpers import *


class VIEW3D_PT_setScenePanel(bpy.types.Panel):
    bl_label = "Set Scene"
    bl_id_name = "VIEW3D_PT_setScenePanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Camera Table"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        myVariables = scene.my_variables
        StateVal.if_initialized = True

        if StateVal.if_testRendering:
            layout.operator("test_render.op", text="Next Shot")
            return None

        if scene.objects.get(myVariables.fov_name) == None \
            or  scene.objects.get(myVariables.attn_name) == None \
            or scene.objects.get(myVariables.camera_name) == None:
            layout.operator("set_scene.op")
        elif not StateVal.if_rendered:
            if context.active_object.type == 'CAMERA' \
                and context.active_object.name_full != myVariables.camera_name:
                layout.operator("set_scene.op")
                layout.row()
            row = layout.row()
            row.label(text= "Scene Settings")
            layout.prop(myVariables, "distance_from_camera")
            layout.prop(myVariables, "fov")
            layout.prop(myVariables, "angleRes")

            layout.prop(myVariables, "max_resolution")
            row = layout.row()
            row.label(text= "x Resolution: "+str(scene.render.resolution_x))
            row = layout.row()
            row.label(text= "y Resolution: "+str(scene.render.resolution_y))
            layout.row()
            layout.prop(myVariables, "attn_width")
            layout.prop(myVariables, "attn_height")
            layout.prop(myVariables, "attn_thick")
            layout.prop(myVariables, "attn_z")
            layout.prop(myVariables, "attn_layers")

            row = layout.row()
            row.label(text= "Rendering")
            layout.prop(myVariables, "imgName")
            layout.prop(myVariables, "select_config")
            layout.prop(myVariables, "current_path")
            layout.operator("configure.op")
            layout.operator("render.op")
            layout.operator("test_render.op")
        else:
            if context.active_object.type == 'CAMERA' \
                and context.active_object.name_full != myVariables.camera_name:
                layout.operator("set_scene.op")
                layout.row()
            row = layout.row()
            row.label(text= "Scene Settings")

            layout.label(text= "x Resolution: "+str(scene.render.resolution_x))
            layout.label(text= "y Resolution: "+str(scene.render.resolution_y))
            layout.prop(myVariables, "attn_thick")

            row = layout.row()
            row.label(text='Width')
            row.label(text='Height')
            row.label(text='thickness')

            row = layout.row()
            row.prop(myVariables, "real_width", slider=False)
            row.prop(myVariables, "real_height", slider=False)
            row.prop(myVariables, "real_thick", slider=False)

            row = layout.row()
            row.prop(myVariables, "attn_z")      
            layout.prop(myVariables, "attn_layers")

            layout.label(text="Image Generation")
            layout.prop(myVariables, "current_setting")
            current_setting = myVariables.current_setting
            # StateVal.is_generated[current_setting] = StateVal.is_generated.get(current_setting, False)
            layout.operator("gen.op")
            imgName = myVariables.imgName
            try:
                cond = StateVal.configs[imgName]['is_generated'][current_setting]
            except:
                print("Panel: al.configs[imgName]['is_generated'][current_setting] not initialized")
                cond = False
            layout.operator("prev.op") if cond else None

            row = layout.row()
            row.label(text= "Rendering")
            layout.prop(myVariables, "imgName")
            layout.prop(myVariables, "select_config")
            layout.prop(myVariables, "current_path")
            layout.operator("configure.op")
            layout.operator("render.op", text= "Re-render")
            layout.operator("test_render.op")


classes = [VIEW3D_PT_setScenePanel]
def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    InitVal.initialize_properties(bpy.context.scene.my_variables)
    StateVal.update_configs(bpy.context)
    StateVal.if_initialized = True

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)


