from pip._internal import main

# main(['install','scipy'])
# main(['install','imageio'])
# main(['install', 'bidict'])
# main(['install', 'matplotlib'])
# main(['install', 'numba'])


import bpy
from math import pi, sin, cos, acos, asin, tan, radians, sqrt, log, exp, atan
from bpy_extras.io_utils import ExportHelper
import numpy as np
from numpy import array, cross
import os
import os.path
from mathutils import Vector, Matrix, Euler
import json
import scipy
import scipy.io


import sys
from scipy import ndimage as ndimg
from math import sin, cos, asin, acos, tan, atan, pi, e, sqrt, radians, degrees
import numpy as np
from numpy import array
import json
from functools import reduce
import scipy.io

import os

from scipy.spatial.transform import Rotation as Rot

# img = mpimg.imread('default-00.png')
# imgplot = plt.imshow(img)
# plt.show()

import imageio.v3 as iio
# m = iio.imread("default-00.png")
# the two above both return numpy array


# scene objects includes:
# "main_camera"
# "fov_bounding_box"
# "attenuation_layer_bounding_box"

class StateVal:
    exportVal = ["filePath", "imgRes", "if_rendered", "factor", "channels", "is_generated", 'subPaths']
    importVal = ["imgRes", "if_rendered", "factor", "is_generated", "subPaths"]
    channels = 1
    imgRes = (512, 384)
    configs = {}
    if_rendered = False
    factor = 10
    if_testRendering = False
    if_initialized = False
    generator = None
    is_generated = {'default':False}
    current_setting = 'default'
    subPaths = {}

    @classmethod
    @property
    def filePath(cls):
        myVariables = bpy.context.scene.my_variables
        imgName = myVariables.imgName
        current_path = myVariables.current_path
        return os.path.join(current_path, "data", imgName)

    @classmethod
    def update_configs(cls, context):
        myVar = context.scene.my_variables
        current_path = myVar.current_path
        _, cls.configs = get_configs(context, current_path)

    @classmethod
    def getitems(cls):
        for k in cls.exportVal:
            yield (k, getattr(cls, k))

    @classmethod
    def setitems(cls, dic):
        for k in cls.importVal:
            setattr(cls, k, dic[k])


class InitVal:
    # main camera
    fov = radians(30)
    distance_from_camera = 8
    angleRes = 7 # n -> n*n
    max_resolution = 512
    imgName = "default"
    current_path = r"C:\\"

    # attenuation layer bounding box
    attn_width = 10
    attn_height = 384/512*10
    attn_thick = 1.6
    attn_layers = 7
    attn_z = 0.0

    @classmethod
    def initialize_properties(cls, obj):
        # main camera
        obj.fov = cls.fov
        obj.distance_from_camera = cls.distance_from_camera
        obj.angleRes = cls.angleRes # n -> n*n
        obj.max_resolution = cls.max_resolution
        obj.imgName = cls.imgName
        obj.current_path = cls.current_path
        obj.camera_name = "main_camera"

        obj.attn_name = "attenuation_layer_bounding_box"
        obj.fov_name = "fov_bounding_box"

        # attenuation layer bounding box
        obj.attn_width = cls.attn_width
        obj.attn_height = cls.attn_height
        obj.attn_thick = cls.attn_thick
        obj.attn_layers = cls.attn_layers
        obj.attn_z = cls.attn_z



def create_fov_bounding_box(context):
    scene = context.scene
    myVariables = scene.my_variables
    try:
        cam = scene.camera
    except:
        raise Exception("SCENE.CAMERA DOESN'T EXIST: create_fov_bounding_box()")

    distance_from_camera = myVariables.distance_from_camera
    fov = myVariables.fov

    # fov bounding box
    vertices = [(0, 0, 0),
        (distance_from_camera, -distance_from_camera*tan(fov/2), distance_from_camera/cos(fov/2)*tan(fov/2)),
        (distance_from_camera, -distance_from_camera*tan(fov/2), -distance_from_camera/cos(fov/2)*tan(fov/2)),
        (distance_from_camera, distance_from_camera*tan(fov/2), -distance_from_camera/cos(fov/2)*tan(fov/2)),
        (distance_from_camera, distance_from_camera*tan(fov/2), distance_from_camera/cos(fov/2)*tan(fov/2))]
    edges = [(0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1)]
    # faces = [(0, 1, 2),
    #     (0, 2, 3),
    #     (0, 3, 4),
    #     (0, 4, 1)]
    faces = []

    #data
    for key in scene.objects.keys():
        if "fov_bounding_box" in key:
            bpy.data.objects.remove(scene.objects[key])

    if "fov_bounding_box" in bpy.data.meshes.keys():
        bpy.data.meshes.remove(bpy.data.meshes["fov_bounding_box"])

    fov_bounding_box_data = \
        bpy.data.meshes.new("fov_bounding_box")

    # # material
    # if "transparent_mesh_fov" in bpy.data.materials.keys():
    #     bpy.data.materials.remove(bpy.data.materials["transparent_mesh_fov"])
    #
    # transparent_mesh = bpy.data.materials.new("transparent_mesh_fov")
    # transparent_mesh.diffuse_color = (1.0,0.0,1.0, 0.05)
    # transparent_mesh.blend_method = 'BLEND'
    # transparent_mesh.use_nodes = True
    # transparent_mesh.node_tree.nodes["Principled BSDF"].inputs['Alpha'].default_value = 0.3
    #
    # fov_bounding_box_data.materials.append(transparent_mesh)

    # object
    fov_bounding_box_data.from_pydata(vertices, edges, faces)
    fov_bounding_box_data.update()
    fov_bounding_box_obj = \
        bpy.data.objects.new("fov_bounding_box", fov_bounding_box_data)


    view_layer = context.view_layer
    view_layer.active_layer_collection.collection.objects.link(fov_bounding_box_obj)

    camera_centre_location(context, distance_from_camera, fov_bounding_box_obj)
    camera_centre_rotation(context, fov_bounding_box_obj)
    fov_bounding_box_obj.hide_select = True
    fov_bounding_box_obj.hide_render = True
    myVariables.fov_name = fov_bounding_box_obj.name_full

    return fov_bounding_box_obj

def create_attenuation_layer_bounding_box(context):
    scene = context.scene
    myVariables = scene.my_variables
    distance_from_camera = myVariables.distance_from_camera
    width = myVariables.attn_width
    height = myVariables.attn_height
    thick = myVariables.attn_thick
    layers = myVariables.attn_layers
    x_axis = myVariables.attn_z

    vertices = [(-thick/2, -width/2, -height/2),
        (thick/2, -width/2, -height/2),
        (thick/2, -width/2, height/2),
        (-thick/2, -width/2, height/2),
        (-thick/2, width/2, -height/2),
        (thick/2, width/2, -height/2),
        (thick/2, width/2, height/2),
        (-thick/2, width/2, height/2)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 5), (5, 6), (6, 7), (7, 4)]
    faces = [(0, 1, 2, 3),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
        (4, 5, 6, 7)]
    #data
    for key in scene.objects.keys():
        if "attenuation_layer_bounding_box" in key:
            bpy.data.objects.remove(scene.objects[key])

    if "attenuation_layer_bounding_box" in bpy.data.meshes.keys():
        bpy.data.meshes.remove(bpy.data.meshes["attenuation_layer_bounding_box"])

    attenuation_layer_bounding_box_data = \
        bpy.data.meshes.new("attenuation_layer_bounding_box")

    # material
    if "transparent_mesh_attn" in bpy.data.materials.keys():
        bpy.data.materials.remove(bpy.data.materials["transparent_mesh_attn"])

    transparent_mesh = bpy.data.materials.new("transparent_mesh_attn")
    transparent_mesh.diffuse_color = (1.0,0.0,1.0, 0.05)
    transparent_mesh.blend_method = 'BLEND'
    transparent_mesh.use_nodes = True
    transparent_mesh.node_tree.nodes["Principled BSDF"].inputs['Alpha'].default_value = 0.3

    attenuation_layer_bounding_box_data.materials.append(transparent_mesh)

    # object
    attenuation_layer_bounding_box_data.from_pydata(vertices, edges, faces)
    attenuation_layer_bounding_box_data.update()
    attenuation_layer_bounding_box_obj = \
        bpy.data.objects.new("attenuation_layer_bounding_box", attenuation_layer_bounding_box_data)

    view_layer = context.view_layer
    view_layer.active_layer_collection.collection.objects.link(attenuation_layer_bounding_box_obj)

    camera_centre_location(context, distance_from_camera+x_axis, attenuation_layer_bounding_box_obj)
    camera_centre_rotation(context, attenuation_layer_bounding_box_obj)
    attenuation_layer_bounding_box_obj.hide_select = True
    attenuation_layer_bounding_box_obj.hide_render = True
    myVariables.attn_name = attenuation_layer_bounding_box_obj.name_full

    return attenuation_layer_bounding_box_obj

def create_main_camera(context):
    scene = context.scene
    myVariables = scene.my_variables
    distance_from_camera = myVariables.distance_from_camera

    # we first create the camera object
    if "main_camera" in scene.objects.keys():
        bpy.data.objects.remove(scene.objects["main_camera"])

    main_camera_data = bpy.data.cameras.new('main_camera')
    main_camera_obj = bpy.data.objects.new('main_camera', main_camera_data)
    context.collection.objects.link(main_camera_obj)
    # add camera to scene
    scene = context.scene
    scene.camera = main_camera_obj

    main_camera_obj.rotation_mode = "XYZ"
    main_camera_obj.rotation_euler = (radians(90), 0, radians(90))
    main_camera_obj.location = (distance_from_camera, 0, 0)
    main_camera_obj.hide_select = True

    return main_camera_obj

# only called when main_camera exists
def initializeScene(context):
    # InitVal.update_all(context.scene.my_variables)
    create_attenuation_layer_bounding_box(context)
    create_fov_bounding_box(context)

# Modifying
def camera_centre_location(context, distance_from_camera, obj):
    cam_name = context.scene.my_variables.camera_name
    cam = context.scene.objects.get(cam_name)
    if cam == None:
        raise Exception("CAMERA DOESN'T EXIST: camera_centre_location()")
    cam.rotation_mode = "XYZ"
    if cam.rotation_mode != "XYZ":
        raise Exception("SCENE.CAMERA ROTATION_MODE ERROR: camera_centre_location()")
    (ang_x, ang_y, ang_z) = cam.rotation_euler
    facing = Vector([0, 0, -1])
    facing = Matrix.Rotation(ang_x, 3, [1, 0, 0]) @ facing
    facing = Matrix.Rotation(ang_y, 3, [0, 1, 0]) @ facing
    facing = Matrix.Rotation(ang_z, 3, [0, 0, 1]) @ facing
    obj.location = facing.normalized()*distance_from_camera + Vector(cam.location)

# rotation_euler ("XYZ")
def camera_centre_rotation(context, obj):
    cam_name = context.scene.my_variables.camera_name
    cam = context.scene.objects.get(cam_name)
    if cam == None:
        raise Exception("CAMERA DOESN'T EXIST: camera_centre_rotation()")
    else:
        cam.rotation_mode = "XYZ"
        if cam.rotation_mode != "XYZ":
            raise Exception("SCENE.CAMERA ROTATION_MODE ERROR: camera_centre_location()")
        obj.rotation_euler = (-radians(90), -radians(90), 0)
        obj.delta_rotation_euler = cam.rotation_euler

# matlab implementation version: deltaX/deltaY rotation - rotating on fixed z/y axis
def spherical_rotation_matlab(mat, y_axis, z_axis, pivot, beta, phi):
    M = Matrix.Translation(pivot) \
        @ Matrix.Rotation(-beta, 4, y_axis) \
        @ Matrix.Rotation(phi, 4, z_axis) \
        @ Matrix.Translation(-pivot)
    return M @ mat

# shperical coordinate rotation: theta/phi
def spherical_rotation(mat, y_axis, z_axis, pivot, theta, phi):
    rotated_y = Matrix.Rotation(phi, 3, z_axis) @ y_axis
    M = Matrix.Translation(pivot) \
        @ Matrix.Rotation(theta, 4, rotated_y) \
        @ Matrix.Rotation(phi, 4, z_axis) \
        @ Matrix.Translation(-pivot)
    return M @ mat

# Setting
rot_func = [spherical_rotation, spherical_rotation_matlab][1]
def edit_main_camera(context, center_matrix, beta, phi):
    myVariables = context.scene.my_variables
    pivot = context.scene.objects[myVariables.fov_name].location
    delta_r = pivot - center_matrix.decompose()[0]
    z_axis = (center_matrix.to_euler().to_matrix() @ Vector((0, 1, 0))).normalized()
    y_axis = delta_r.cross(z_axis).normalized()
    print("z-axis: ", z_axis)
    print("y-axis: ", y_axis)
    return rot_func(center_matrix, y_axis, z_axis, pivot, beta, phi)


# def orthogonal_rotation(mat, y_axis, z_axis, delta_y, delta_z):


# def edit_main_camera(context, center_matrix, index_y, index_z):
#     myVariables = context.scene.my_variables
#     pivot = context.scene.objects[myVariables.fov_name].location
#     ortho_scale = context.scene.objects[myVariables.camera_name].data.ortho_scale
#     camera_distance = myVariables.distance_from_camera
#     fov = myVariables.fov
#     angleRes = myVariables.angleRes
#
#     unit_y = 2*camera_distance*tan(fov/2)/(angleRes-1)
#     unit_z = 2*camera_distance*tan(fov/2)/(angleRes-1)
#     delta_y = camera_distance*tan(fov/2) - unit_y
#     delta_z =
#
#     delta_r = pivot - center_matrix.decompose()[0]
#     z_axis = (center_matrix.to_euler().to_matrix() @ Vector((0, 1, 0))).normalized()
#     y_axis = delta_r.cross(z_axis).normalized()
#
#     return orthogonal_rotation(center_matrix, y_axis, z_axis, pivot, theta, phi)

def edit_fov_bounding_box(context, obj, distance_from_camera, fov):
    initFov = InitVal.fov
    initCameraDistance = InitVal.distance_from_camera

    x_scale = distance_from_camera/initCameraDistance
    y_scale = tan(fov/2)*distance_from_camera/(tan(initFov/2)*initCameraDistance)
    z_scale = (tan(fov/2)*distance_from_camera/cos(fov/2))\
        /(tan(initFov/2)*initCameraDistance/cos(initFov/2))

    camera_centre_location(context, distance_from_camera, obj)
    camera_centre_rotation(context, obj)

    obj.scale = (x_scale, y_scale, z_scale)

def edit_attenuation_layer_bounding_box(context, obj, width, height, thick, x, distance_from_camera):
    initWidth = InitVal.attn_width
    initHeight = InitVal.attn_height
    initThick = InitVal.attn_thick
    initLayers = InitVal.attn_layers
    initX = InitVal.attn_z

    x_scale = thick/initThick
    y_scale = width/initWidth
    z_scale = height/initHeight

    camera_centre_location(context, distance_from_camera+x, obj)
    camera_centre_rotation(context, obj)

    obj.scale = (x_scale, y_scale, z_scale)


def update_scene(self, context):
    if not StateVal.if_initialized:
        return None
    print()
    print("update_scene: self=", self)
    scene = context.scene
    myVariables = context.scene.my_variables

    distance_from_camera = myVariables.distance_from_camera
    fov = myVariables.fov
    width = myVariables.attn_width
    height = myVariables.attn_height
    thick = myVariables.attn_thick
    x = myVariables.attn_z

    fov_name = myVariables.fov_name
    attn_name = myVariables.attn_name
    fov_box = scene.objects.get(fov_name)
    attn_box = scene.objects.get(attn_name)
    if fov_box == None or attn_box == None:
        raise Exception("#####ERROR##### update_scene???")

    max_resolution = myVariables.max_resolution
    cam = scene.objects[myVariables.camera_name]
    cam.data.type = 'ORTHO'
    max_dim = max(width, height)
    cam.data.ortho_scale = max_dim
    res_x = int(max_resolution*width/max_dim)
    res_y = int(max_resolution*height/max_dim)
    StateVal.imgRes = (res_x, res_y)
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y

    current_path = myVariables.current_path
    imgName = myVariables.imgName

    edit_fov_bounding_box(context, fov_box, distance_from_camera, fov)
    edit_attenuation_layer_bounding_box(context, attn_box, width, height, thick, x, distance_from_camera)

    # Test
    StateVal.generator = test_render(context)


def pack_config(var, *clss):
    imgName = var.imgName
    dic = {k: var.get(k) for k in var.keys()}
    # extra configuration
    print()
    print("dic:", dic)
    for cls in clss:
        print()
        print("cls:", cls)
        for k, v in cls.getitems():
            print(f"(k: {k}, v: {v})")
            dic[k] = v
    return imgName, dic

def saveConfig(context, subPath=None):
    # Json
    myVariables = context.scene.my_variables
    current_path = myVariables.current_path
    if subPath:
        current_path = subPath

    imgName, config = pack_config(myVariables, StateVal)

    config_path, configs = get_configs(context, current_path)
    configs[imgName] = config

    with open(config_path, 'w') as fh:
        try:
            fh.write(json.dumps(configs))
        except:
            print("!!!!!!!!!!!!!!")
            print(configs)
            raise Exception("json saving failed!")

    printSuccess("Save to light_field_config.json")

    StateVal.update_configs(context)
    # .mat
    factor = StateVal.factor
    fov = config["fov"]
    attn_height= config["attn_height"]*factor
    attn_width= config["attn_width"]*factor
    attn_thick= config["attn_thick"]*factor
    attn_z= config["attn_z"]*factor
    attn_layers= config["attn_layers"]
    res_x = config["imgRes"][0]
    res_y = config["imgRes"][1]
    angleResX = config["angleRes"]
    angleResY = config["angleRes"]

    tomat = {"P_fov" : fov,
    "P_lightFieldSize" : [attn_height, attn_width],
    "P_lightFieldResolution" : [angleResY, angleResX, res_y, res_x, 3],
    "P_lightFieldImageResolution" : [res_y, res_x],
    "P_numLayers" : attn_layers,
    "P_layerDistance" : attn_thick/(attn_layers - 1),
    "P_layerOrigin" : [0, 0, -attn_z],
    "P_if_rendered" : StateVal.if_rendered,
    "P_outPath" : imgName}
    scipy.io.savemat(os.path.join(current_path, "data", "LightFieldImage_to_Matlab.mat"), tomat)

    printSuccess("Save to LightFieldImage_to_Matlab.mat")

def printSuccess(s):
    print(f'''

    ############################################################################
        {s} successful!
    ############################################################################

    ''')

# all the extra class values packed into config should have .getitems() and __setitem__!
def unpack_config(config, myVar, *clss):
    for k in myVar.keys():
        myVar[k] = config[k]
    for cls in clss:
        cls.setitems(config)

def get_configs(context, current_path):
    myVariables = context.scene.my_variables

    config_path = os.path.join(current_path, "light_field_config.json")
    try:
        with open(config_path, 'r') as fh:
            text = fh.read()
            configs = json.loads(text if text != '' else "{}")
    except FileNotFoundError:
        print("!!!!!!!!!!!!!!")
        print("get_configs: FileNotFoundError")
        configs = {}
    except json.decoder.JSONDecodeError:
        print("!!!!!!!!!!!!!!")
        print("get_configs: JSONDecodeError")
        configs = {}

    return config_path, configs

def configuration_exists(context):
    scene = context.scene
    myVariables = scene.my_variables
    current_path = myVariables.current_path
    _, configs = get_configs(context, current_path)

    return myVariables.imgName in configs


def select_config_callback(scene, context):
    myVariables = context.scene.my_variables
    imgName = myVariables.imgName
    # items: list of (identifier, name, description, icon, number)
    blank_item1 = ("", "", "", "", 0)
    blank_item2 = ("", "", "", "", 1)

    configs = StateVal.configs
    items = []
    if imgName not in configs:
        items.append(blank_item1)
        items.append(blank_item2)
    else:
        items.append(blank_item1)
        items.append((imgName, imgName, "export configuration", "COPY_ID", 1))
    tags = enumerate(filter(lambda k: k!= imgName, configs.keys()))
    # tags = enumerate(configs.keys())
    for i, k in tags:
        items.append((k, k, "export configuration", "COPY_ID", i+2))

    return items

def when_select_config_update(self, context):
    myVar = context.scene.my_variables
    imgName = myVar.select_config
    # assert type(self.select_config) == str, "when_select_config_update: Enum type mismatch!"
    myVar.imgName = imgName
    when_imgName_update(self, context)

def when_imgName_update(self, context):
    myVar = context.scene.my_variables
    # assert self is context.scene.my_variables, "when_imgName_update: self is not myVar!"
    imgName = myVar.imgName
    current_path = myVar.current_path

    configs = StateVal.configs
    if imgName in configs:
        unpack_config(configs[imgName], myVar, StateVal)
        update_scene(self, context)
    else:
        StateVal.if_rendered = False

def when_current_path_update(self, context):
    # order matters!
    StateVal.update_configs(context)
    when_imgName_update(self, context)


def render(cam, context):
    update_scene(None, context)
    scene = context.scene
    myVariables = scene.my_variables

    angleRes = myVariables.angleRes
    fov = myVariables.fov
    distance_from_camera = myVariables.distance_from_camera
    imgName = myVariables.imgName
    filePath = StateVal.filePath

    delta = fov/(angleRes-1)
    lightField_path = os.path.join(filePath, "img")
    try:
        os.mkdirs(lightField_path)
    except: pass

    scene.render.image_settings.file_format='PNG'

    center_matrix = cam.matrix_world.copy()
    for i in range(angleRes):
        for j in range(angleRes):
            delta_y = 2*tan(fov/2)/(angleRes-1)
            delta_z = 2*tan(fov/2)/(angleRes-1)
            cam.matrix_world = \
                edit_main_camera(context, center_matrix
                    , atan(tan(fov/2)-i*delta_z)
                    , atan(-(tan(fov/2)-j*delta_y)))
            scene.render.filepath = \
                f"{lightField_path}\\{imgName}-{i*angleRes+j}.png"
            bpy.ops.render.render(write_still=True)
            print("coord({0}, {1})".format(i, j))
    cam.matrix_world = center_matrix
    printSuccess("Render Image")

def matProj(beta, phi):
    z0 = array((0, 0, 1))
    y0 = array((0, 1, 0))
    x0 = array((1, 0, 0))
    P_rot = Rot.from_rotvec(-phi*x0)
    B_rot = Rot.from_rotvec(-beta*y0)
    zh = B_rot.apply(P_rot.apply(z0))
    yh = B_rot.apply(P_rot.apply(y0))
    xh = B_rot.apply(P_rot.apply(x0))

    # zh = array(
    #     (-sin(theta),
    #     cos(theta)*sin(phi),
    #     cos(theta)*cos(phi)))
    # yh = array(
    #     (0,
    #     cos(phi),
    #     -sin(phi)))
    # xh = array(
    #     (cos(theta),
    #     sin(theta)*sin(phi),
    #     sin(theta)*cos(phi),
    #     ))

    x0Pzh = (x0 @ zh)*zh
    y0Pzh = (y0 @ zh)*zh
    x0Pzh_xh = (x0 - x0Pzh) @ xh
    x0Pzh_yh = (x0 - x0Pzh) @ yh
    y0Pzh_xh = (y0 - y0Pzh) @ xh
    y0Pzh_yh = (y0 - y0Pzh) @ yh

    mat_proj = array([[x0Pzh_xh, y0Pzh_xh],
        [x0Pzh_yh, y0Pzh_yh]])
    print("beta: {}, phi: {}".format(beta, phi))
    print(mat_proj)
    return mat_proj

def imgPostProcessing(context):
    myVar = context.scene.my_variables
    imgName = myVar.imgName
    current_path = myVar.current_path
    config_path, configs = get_configs(context, current_path)
    cfg = configs[imgName]
    img_name = cfg["imgName"]
    img_dir = os.path.join(cfg["filePath"], "img")

    fovX = cfg["fov"]
    fovY = cfg["fov"]
    angleResX = cfg["angleRes"]
    angleResY = cfg["angleRes"]

    print(cfg)
    # We treat the verticle axis of an image as x, horizontal as y, for conserving
    # rotational charateristic of a right hand coordinate
    delta_Y = 2*tan(fovY/2)/(angleResY - 1)
    delta_X = 2*tan(fovX/2)/(angleResX - 1)

    # decompose .png img into R, G, B (n, m) + alpha (n, m)
    def decompose_img(img):
        RGB = img[...,:3]
        try:
            alpha = img[...,3]
        except IndexError:
            print("image file no alpha channel")
            alpha = None
        return RGB[..., 0], RGB[..., 1], RGB[..., 2], alpha

    for i in range(angleResY):
        for j in range(angleResX):
            # Create interpolation matrix
            beta = atan(tan(fovY/2) - i*delta_Y)
            phi = atan(-(tan(fovX/2) - j*delta_X))
            # mat_backproj = array([[cos(theta), 0],
            #      [-2*sin(theta)*cos(theta)**2*sin(phi)\
            #      , 2*cos(phi)*sin(phi)**2*cos(theta)**2]])/(2*cos(phi)*sin(phi)**2*cos(theta)**3)
            mat_proj = matProj(beta, phi)

            # Load file
            path =\
                os.path.join(img_dir, "{0}-{1}.png".format(img_name, i*angleResX+j))
            img = iio.imread(path)
            res_x, res_y, _ = img.shape

            # plt.subplot(1, 2, 1)
            # plt.imshow(img)

            # c = array((0, res_z))
            # Iminus = array([[1, 0], [0, -1]])
            # O = array((res_y/2, res_z/2))
            # Inv = array([[0, 1], [1, 0]])
            O = array((res_x/2, res_y/2))
            # scale = min(res_x, res_y) # the fuck is this coordinate system, it's WACK!
            # S = array([[res_x/scale, 0], [0, res_y/scale]])
            # Sinv = array([[scale/res_x, 0], [0, scale/res_y]])

            # img.shape = (y, z, 4): .png has an extra alpha channel -> RGB+alpha = 4
            R, G, B, alpha = decompose_img(img)
            # offset_img_origin = Inv @ (Iminus @ mat_proj @ (c - O) + Iminus @ O - c)
            # M = Inv @ Iminus @ mat_proj @ Iminus @ Inv
            M = mat_proj
            offset_img_origin = (O - M @ O)

            R = ndimg.affine_transform(R, M, offset= offset_img_origin, mode= 'nearest')
            G = ndimg.affine_transform(G, M, offset= offset_img_origin, mode= 'nearest')
            B = ndimg.affine_transform(B, M, offset= offset_img_origin, mode= 'nearest')
            img[..., 0] = R
            img[..., 1] = G
            img[..., 2] = B
            img = img[...,:3]

            # plt.subplot(1, 2, 2)
            # plt.imshow(img)
            # plt.show()
            print("Shape", i*angleResX+j, ":", img.shape)
            iio.imwrite(os.path.join(img_dir, "{0}-{1}.png".format(img_name, i*angleResX+j)), img)
            print()
            print("#####################")
            print("Projection Corrected!")
            print("#####################")


def test_render(context):
    scene = context.scene
    myVariables = scene.my_variables
    cam = context.scene.objects.get(myVariables.camera_name)

    angleRes = myVariables.angleRes
    fov = myVariables.fov
    distance_from_camera = myVariables.distance_from_camera
    imgName = myVariables.imgName
    filePath = StateVal.filePath

    delta = 2*tan(fov/2)/(angleRes-1)

    center_matrix = cam.matrix_world.copy()
    for i in range(angleRes):
        for j in range(angleRes):
            cam.matrix_world = \
                edit_main_camera(context
                    , center_matrix
                    , atan(tan(fov/2)-i*delta)
                    , atan(-(tan(fov/2)-j*delta)))
            yield 0
    cam.matrix_world = center_matrix

def update_realcalc(self, context, updtVar):
    myVariables = context.scene.my_variables
    if updtVar == 'real_width':
        real_width = self.real_width
        attn_width = self.attn_width
        factor = real_width/attn_width
    if updtVar == 'real_height':
        real_height = self.real_height
        attn_height = self.attn_height
        factor = real_height/attn_height
    if updtVar == 'real_thick':
        real_thick = self.real_thick
        attn_thick = self.attn_thick
        factor = real_thick/attn_thick

    if updtVar != 'real_width':
        attn_width = myVariables.attn_width
        myVariables['real_width'] = attn_width*factor
    if updtVar != 'real_height':
        attn_height = myVariables.attn_height
        myVariables['real_height'] = attn_height*factor
    if updtVar != 'real_thick':
        attn_thick = myVariables.attn_thick
        myVariables['real_thick'] = attn_thick*factor

def when_current_settings_update(self, context):
    myVar = context.scene.my_variables
    current_setting = myVar.current_setting
    imgName = myVar.imgName
    print(f'''
    ##################
    {current_setting}
    ##################
    ''')
    config = StateVal.configs[imgName]
    config['is_generated'] = config.get('is_generated', {})
    config['is_generated'][current_setting] = config['is_generated'].get(current_setting, False)

# def select_setting_callback(self, context):
#     ...

# def when_select_setting_update(self, context):
