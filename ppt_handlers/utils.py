import os

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
            print('---***---')

def is_primitive(obj):
    """
    判断对象是否是基本类型。

    参数:
    obj: 要判断的对象

    返回:
    如果对象是基本类型，返回True，否则返回False
    """
    return isinstance(
        obj, (int, float, complex, bool, str, bytes, bytearray, type(None))
    )


def object_to_dict(obj, result=None):
    """
    将对象的非隐藏属性拷贝到一个字典中。

    参数:
    obj: 要拷贝属性的对象

    返回:
    包含对象非隐藏属性的字典
    """
    if result is None:
        result = {}
    for attr in dir(obj):
        try:
            if not attr.startswith("_") and not callable(getattr(obj, attr)):
                attr_value = getattr(obj, attr)
                if is_primitive(attr_value):
                    result[attr] = attr_value
        except Exception as e:
            print(f"Error while processing attribute {attr}: {e}")
    return result


def dict_to_object(dict:dict, obj: object):
    """
    从字典中恢复对象的属性。

    参数:
    d: 包含对象属性的字典
    obj: 要恢复属性的对象

    返回:
    恢复属性后的对象
    """
    for key, value in dict.items():
        setattr(obj, key, value)


class Config:
    def __init__(self):
        # 当前运行目录，而不是文件所在目录
        self.BASE_DIR = os.curdir
        self.PPT_DIR = os.path.join(self.BASE_DIR, "resource")
        self.GEN_PPT_DIR = os.path.join(self.BASE_DIR, "output/ppts")
        self.IMAGE_DIR = os.path.join(self.BASE_DIR, "output/images")
        for the_dir in [self.PPT_DIR, self.IMAGE_DIR, self.GEN_PPT_DIR]:
            if not os.path.exists(the_dir):
                os.makedirs(the_dir)


if __name__ == "__main__":
    config = Config()
    print(config.PPT_DIR)
