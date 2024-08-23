import os
import re
import xml.etree.ElementTree as ET
from types import SimpleNamespace

from lxml import etree
from pptx.dml.fill import _NoFill, _NoneFill
from pptx.enum.shapes import MSO_CONNECTOR_TYPE
from pptx.shapes.base import BaseShape
from pptx.shapes.group import GroupShape
from pptx.util import Length
from rich import print

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


def filename_normalize(filename: str):
    return re.sub(r"[\/\0]", "_", filename)


def set_proxy(proxy_url: str):
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url


def get_text_inlinestyle(para: dict):
    font = SimpleNamespace(**para["font"])
    font_size = f"font-size: {Length(font.size).pt}pt;" if font.size else ""
    font_family = f"font-family: {font.name};" if font.name else ""
    font_color = f"color={font.color};" if font.color else ""
    font_bold = "font-weight: bold;" if font.bold else ""
    return 'style="{}"'.format("".join([font_size, font_family, font_color, font_bold]))


def extract_fill(shape: BaseShape):
    if "fill" not in dir(shape):
        return None
    fill_dict = {
        "fill_xml": shape.fill._xPr.xml,
    } | {k: v for k, v in object_to_dict(shape.fill).items() if "color" in k}
    if not isinstance(shape.fill._fill, (_NoneFill, _NoFill)):
        fill_dict["type"] = shape.fill.type.name.lower()
    return fill_dict


def apply_fill(shape: BaseShape, fill: dict):
    if fill is None:
        return
    replace_xml_node(shape.fill._xPr, fill["fill_xml"])


def save_xml(xml_string: str, filename: str = "output.xml"):
    root = ET.fromstring(xml_string)
    tree = ET.ElementTree(root)
    tree.write(filename, encoding="utf-8", xml_declaration=True)


def parse_groupshape(groupshape: GroupShape):
    assert isinstance(groupshape, GroupShape)
    group_top_left_x = groupshape.left
    group_top_left_y = groupshape.top
    group_width = groupshape.width
    group_height = groupshape.height
    # false size and xy
    shape_top_left_x = min([sp.left for sp in groupshape.shapes])
    shape_top_left_y = min([sp.top for sp in groupshape.shapes])
    shape_width = (
        max([sp.left + sp.width for sp in groupshape.shapes]) - shape_top_left_x
    )
    shape_height = (
        max([sp.top + sp.height for sp in groupshape.shapes]) - shape_top_left_y
    )
    # scale xy
    group_shape_xy = []
    for sp in groupshape.shapes:
        group_shape_left = (
            sp.left - shape_top_left_x
        ) * group_width / shape_width + group_top_left_x
        group_shape_top = (
            sp.top - shape_top_left_y
        ) * group_height / shape_height + group_top_left_y
        group_shape_width = sp.width * group_width / shape_width
        group_shape_height = sp.height * group_height / shape_height
        group_shape_xy.append(
            {
                "left": Length(group_shape_left),
                "top": Length(group_shape_top),
                "width": Length(group_shape_width),
                "height": Length(group_shape_height),
            }
        )
    return group_shape_xy


# 这个xpr中包含了x:left, y:top, cx:widht, cy:height四个属性，用于描述矩形的位置和大小
def replace_xml_node(old_element, new_xml):
    # 用子节点替换旧节点
    new_element = etree.fromstring(new_xml)
    old_element.getparent().replace(old_element, new_element)


def xml_print(xml_str):
    import xml.dom.minidom

    print(xml.dom.minidom.parseString(xml_str).toprettyxml())


def output_obj(obj):
    for attr in dir(obj):
        if not attr.startswith("_"):
            try:
                if not callable(getattr(obj, attr)):
                    print("obj.%s = %s" % (attr, getattr(obj, attr)))
                else:
                    print("obj.%s is a method" % attr)
            except Exception as e:
                print("obj.%s error: %s" % (attr, e))
            print("---***---")


def is_primitive(obj):
    """
    判断对象或该集合包含的所有对象是否是基本类型。

    参数:
    obj: 要判断的对象

    返回:
    如果对象是基本类型，返回True，否则返回False
    """
    if isinstance(obj, (list, tuple, set, frozenset)):
        return all(is_primitive(item) for item in obj)
    return isinstance(
        obj, (int, float, complex, bool, str, bytes, bytearray, type(None))
    )


DEFAULT_EXCLUDE = set(["element", "language_id", "ln", "placeholder_format"])


def object_to_dict(obj, result=None, exclude=None):
    """
    将对象的非隐藏属性拷贝到一个字典中。

    参数:
    obj: 要拷贝属性的对象

    返回:
    包含对象非隐藏属性的字典
    """
    if result is None:
        result = {}
    exclude = DEFAULT_EXCLUDE.union(exclude or set())
    for attr in dir(obj):
        if attr in exclude:
            continue
        try:
            if not attr.startswith("_") and not callable(getattr(obj, attr)):
                attr_value = getattr(obj, attr)
                if "real" in dir(attr_value):
                    attr_value = attr_value.real
                if is_primitive(attr_value):
                    result[attr] = attr_value
        except:
            pass
    return result


def dict_to_object(dict: dict, obj: object, exclude=None):
    """
    从字典中恢复对象的属性。

    参数:
    d: 包含对象属性的字典
    obj: 要恢复属性的对象

    返回:
    恢复属性后的对象
    """
    if exclude is None:
        exclude = set()
    for key, value in dict.items():
        if key not in exclude:
            setattr(obj, key, value)


class Config:
    def __init__(self, session_id=None):
        if session_id is not None:
            self.set_session(session_id)

    def set_session(self, session_id):
        self.session_id = session_id
        self.RUN_DIR = f"./runs/{session_id}"
        self.IMAGE_DIR = pjoin(self.RUN_DIR, "images")
        self.TEST_PPT = "./resource/陆垚杰_博士论文答辩PPT_0530.pptx"
        for the_dir in [self.RUN_DIR, self.IMAGE_DIR]:
            if not pexists(the_dir):
                os.makedirs(the_dir)


pjoin = os.path.join
pexists = os.path.exists
app_config = Config("test")

if __name__ == "__main__":
    config = Config()
    print(config)
