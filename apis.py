import inspect
import traceback
from enum import Enum
from types import SimpleNamespace

from presentation import Picture, SlidePage

slide = None


# 最好能返回traceback
class ModelAPI:
    def __init__(self, local_vars: dict[str, list[callable]]):
        self.registered_functions = local_vars

    def get_apis_docs(self, op_types: list[str]):
        return "\n".join([self._api_doc(op_type) for op_type in op_types])

    def _api_doc(self, op_type: str):
        op_funcs = self.registered_functions[op_type.value]
        api_doc = [op_type.value.capitalize() + " APIs:"]
        for func in op_funcs:
            sig = inspect.signature(func)
            params = []
            for name, param in sig.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    params.append(f"{name}: {param.annotation.__name__}")
                else:
                    params.append(f"{name}")
            signature = f"def {func.__name__}({', '.join(params)})"
            doc = inspect.getdoc(func)
            api_doc.append(f"{signature}\n\t{doc}")
        return "\n\n".join(api_doc)

    def execute_apis(self, exec_slide: SlidePage, apis: str):
        global slide
        slide = exec_slide
        lines = apis.strip().split("\n")
        code_traces = []
        for line in lines:
            try:
                eval(line, {"slide": slide}, self.registered_functions)
            except Exception as e:
                error_message = traceback.format_exc()
                code_traces.append(f"Error executing line '{line}': {error_message}")
        assert not code_traces, "\n".join(code_traces)
        return code_traces


# element_id 在前，让gpt添加注释和例子


def delete_text(element_id: str):
    """
    This function deletes the text of the element with the given element_id.
    """
    global slide
    element_id, text_id = element_id.split("_")
    for shape in slide.shapes:
        if shape.element_idx == element_id:
            assert shape.text_frame.is_textframe, "The shape does't have a TextFrame."
            idx = -1
            for i in shape.text_frame.data:
                if i["text"]:
                    idx += 1
                if idx == text_id:
                    shape.text_frame.data.remove(i)


def set_text(element_id: str, text: str):
    """
    This function replaces the text of the element with the given element_id.
    """
    global slide
    element_id, text_id = element_id.split("_")
    for shape in slide.shapes:
        if shape.element_idx == element_id:
            assert shape.text_frame.is_textframe, "The shape does't have a TextFrame."
            idx = -1
            for i in shape.text_frame.data:
                if i["text"]:
                    idx += 1
                if idx == text_id:
                    i["text"] = text


def set_image(element_id: str, image_path: str):
    """
    This function sets the image of the element with the given id.
    """
    global slide
    for shape in slide.shapes:
        if shape.element_idx == element_id:
            assert isinstance(shape, Picture), "The shape is not a Picture."
            shape.img_path = image_path


# shape groupfy 或者也许模型可以自己控制哪些是同一个group的
def clone_shape(element_id: str, new_element_id: str, shape_bounds: dict):
    """
    This function clones the shape with the given id, applying the specified bounding box to the new shape.

    Args:
        element_id (str): The unique identifier of the shape to clone.
        new_element_id (str): The new element id for the cloned shape should be element_id + '-' +the number of cloned times.
        shape_bounds (dict): A dictionary containing the bounding box parameters of the shape, including `left`, `top`, `width`, and `height`.
        eg. `clone_shape('1', '1-1', {'left': 100, 'top': 200, 'width': 300, 'height': 400})`
    """
    global slide


def del_shape(element_id: str):
    """
    This function deletes the shape with the given id,
    除了shape_bounds以外的
    """
    for shape in slide.shapes:
        if shape.element_idx == element_id:
            slide.shapes.remove(shape)


# including shape bounds
def swap_style(source_element_id: str, target_element_id: str):
    """
    This function swaps the style of the source shape with the style of the target shape.
    In certain scenarios, one shape in a list may need to become a focal element (e.g., bold, highlighted with different colors, underlined).
    During setting elements, it is sometimes necessary to modify the focal element based on its content.
    """
    is_textframe = "_" in source_element_id
    assert is_textframe == (
        "_" in target_element_id
    ), "The source and target shapes should be the same type (text or picture)."
    if is_textframe:
        source_element_id, source_text_id = source_element_id.split("_")
        target_element_id, target_text_id = target_element_id.split("_")
    source_shape, target_shape = None, None
    for shape in slide.shapes:
        if shape.element_idx == source_element_id:
            source_shape = shape
        if shape.element_idx == target_element_id:
            target_shape = shape
    assert isinstance(
        source_shape, type(target_shape)
    ), "The source and target shapes should be the same type."
    if not is_textframe:
        source_shape.style, target_shape.style = target_shape.style, source_shape.style
    else:
        pass


# def

model_api = ModelAPI(
    {
        "layout adjust": [clone_shape, del_shape],
        "style adjust": [swap_style],
        "set content": [set_text, set_image],
    }
)


class API_TYPES(Enum):
    LAYOUT_ADJUST = "layout adjust"
    STYLE_ADJUST = "style adjust"
    SET_CONTENT = "set content"


if __name__ == "__main__":
    print(model_api.get_apis_docs([API_TYPES.SET_CONTENT, API_TYPES.LAYOUT_ADJUST]))
