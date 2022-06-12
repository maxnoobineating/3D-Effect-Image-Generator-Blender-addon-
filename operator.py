
import bpy
from .helpers import *

class RENDER_OT_op(bpy.types.Operator):
    bl_label = "Render"
    bl_idname = "render.op"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        # try:
        myVariables = context.scene.my_variables
        cam = context.scene.objects.get(myVariables.camera_name)
        if StateVal.if_rendered:
            StateVal.if_rendered = False
            saveConfig(context)
        elif cam != None:
            StateVal.if_rendered = True
            render(cam, context)
            saveConfig(context)
            imgPostProcessing(context)

        else:
            raise Exception('SET_CAM_OP: Scene Setting Not Properly Initialized')
        return {'FINISHED'}

from .attenuationLayersGeneration import main_generate
class GEN_OT_op(bpy.types.Operator):
    bl_label = "Generate"
    bl_idname = "gen.op"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        # try:
        myVariables = context.scene.my_variables
        current_setting = myVariables.current_setting
        filePath = StateVal.filePath
        imgName = myVariables.imgName
        config = StateVal.configs[imgName]
        StateVal.current_setting = current_setting
        StateVal.subPaths[current_setting] = config.get('subPaths', {})
        StateVal.subPaths[current_setting] = os.path.join(filePath, current_setting)
        StateVal.is_generated[current_setting] = True
        saveConfig(context)
        StateVal.update_configs(context)

        main_generate(StateVal.configs[imgName], imgName, current_setting)

        return {'FINISHED'}

from .attenuationLayersGeneration import main_preview
class PREVIEW_OT_op(bpy.types.Operator):
    bl_label = "Preview Images"
    bl_idname = "prev.op"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        # try:
        myVariables = context.scene.my_variables
        current_setting = myVariables.current_setting
        filePath = StateVal.filePath

        imgName = myVariables.imgName
        assert StateVal.is_generated[current_setting], "PREVIEW_OT_op: not yet generated"

        subPath = os.path.join(filePath, current_setting)
        main_preview(imgName, subPath)

        return {'FINISHED'}


class SET_SCENE_OT_op(bpy.types.Operator):
    bl_label = "Set Scene"
    bl_idname = "set_scene.op"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        scene = context.scene
        myVariables = scene.my_variables
        if context.scene.camera == None:
            cam = create_main_camera(context)
            myVariables.camera_name = cam.name_full
            cam.hide_select = True
            initializeScene(context)
        elif context.active_object != None and context.active_object.type == 'CAMERA':
            cam = context.active_object
            myVariables.camera_name = cam.name_full
            cam.hide_select = True
            initializeScene(context)
        else:
            raise Exception("No Camera: SET_SCENE_OT_op")

        return {'FINISHED'}



class CONFIGURE_OT_op(bpy.types.Operator):
    bl_label = "Save Config"
    bl_idname = "configure.op"
    context = bpy.context

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        saveConfig(context)
        return {'FINISHED'}

class TEST_RENDER_OT_op(bpy.types.Operator):
    bl_label = "Test Rendering"
    bl_idname = "test_render.op"

    def execute(self, context):
        if StateVal.if_testRendering:
            try:
                next(StateVal.generator)
            except StopIteration:
                bl_label = "Test Rendering"
                StateVal.if_testRendering = False
                pass
        else:
            StateVal.if_testRendering = True
            bl_label = "Next Shot"
        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        return True


classes = [RENDER_OT_op, GEN_OT_op, PREVIEW_OT_op, SET_SCENE_OT_op, CONFIGURE_OT_op, TEST_RENDER_OT_op]
def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

