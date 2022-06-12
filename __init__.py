# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name" : "CameraTable",
    "author" : "Optical Project Group",
    "description" : "Render 3D light field image for 3D effect pictures",
    "blender" : (2, 80, 0),
    "version" : (0, 0, 1),
    "location" : "3D View > Sidebar > Camera Table",
    "warning" : "",
    "category" : "Generic"
}
import bpy
from .helpers import *

from . import panels
from . import preferences
from . import property
from . import operator

from .panels import VIEW3D_PT_setScenePanel
from .property import MyProperties
from .operator import RENDER_OT_op, GEN_OT_op, PREVIEW_OT_op, SET_SCENE_OT_op, CONFIGURE_OT_op, TEST_RENDER_OT_op

classes = [MyProperties
    , RENDER_OT_op
    , GEN_OT_op
    , PREVIEW_OT_op
    , SET_SCENE_OT_op
    , CONFIGURE_OT_op
    , TEST_RENDER_OT_op
    , VIEW3D_PT_setScenePanel]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.my_variables = bpy.props.PointerProperty(type= MyProperties)
    StateVal.if_initialized = True

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
