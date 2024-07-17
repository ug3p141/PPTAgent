import os
from lxml import etree
from pptx.oxml.ns import _nsmap as namespaces
from pptx.shapes.group import GroupShape
import xml.etree.ElementTree as ET
from rich import print

# def clone_shape(shape):
#     """Add a duplicate of `shape` to the slide on which it appears."""
#     # ---access required XML elements---
#     sp = shape._sp
#     spTree = sp.getparent()
#     # ---clone shape element---
#     new_sp = copy.deepcopy(sp)
#     # ---add it to slide---
#     spTree.append(new_sp)
#     # ---create a proxy object for the new sp element---
#     new_shape = Shape(new_sp, None)
#     # ---give it a unique shape-id---
#     new_shape.shape_id = shape.shape_id + 1000
#     # ---return the new proxy object---
#     return new_shape


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
                "left": int(group_shape_left),
                "top": int(group_shape_top),
                "width": int(group_shape_width),
                "height": int(group_shape_height),
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
        except Exception as e:
            print(f"Error while processing attribute {attr}: {e}")
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
    def __init__(self):
        # 当前运行目录，而不是文件所在目录
        self.BASE_DIR = os.curdir
        self.PPT_DIR = pjoin(self.BASE_DIR, "resource")
        self.GEN_PPT_DIR = pjoin(self.BASE_DIR, "output/ppts")
        self.IMAGE_DIR = pjoin(self.BASE_DIR, "output/images")
        for the_dir in [self.PPT_DIR, self.IMAGE_DIR, self.GEN_PPT_DIR]:
            if not os.path.exists(the_dir):
                os.makedirs(the_dir)


pjoin = os.path.join
base_config = Config()

if __name__ == "__main__":
    config = Config()
    print(config.PPT_DIR)
