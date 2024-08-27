import inspect
import traceback
from copy import deepcopy
from enum import Enum

from jinja2 import Template

from llms import agent_model
from presentation import Picture, SlidePage

slide: SlidePage = None
template_slides: list[SlidePage] = []
# 在重新绑定变量名与全局变量时需要写global声明，否则不需要


class HistoryMark(Enum):
    API_CALL_ERROR = "api_call_error"
    API_CALL_CORRECT = "api_call_correct"


class ModelAPI:

    def __init__(self, local_vars: dict[str, list[callable]], terminate_times: int):
        self.registered_functions = local_vars
        self.local_vars = {func.__name__: func for func in sum(local_vars.values(), [])}
        self.correct_template = Template(open("prompts/agent/code_feedback.txt").read())
        self.terminate_times = terminate_times
        self.history = []

    def get_apis_docs(self, op_types: list[str]):
        return "\n".join([self._api_doc(op_type) for op_type in op_types])

    def _api_doc(self, op_type: str):
        op_funcs = self.registered_functions[op_type]
        api_doc = [op_type.name + " API Docs:"]
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

    def execute_apis(self, prompt: str, apis: str):
        global slide, template_slides
        lines = apis.strip().split("\n")
        code_traces = []
        err_time = 0
        line_idx = 0
        backup_state = (deepcopy(slide), deepcopy(template_slides))
        self.history.append([HistoryMark.API_CALL_ERROR, prompt, apis])
        while line_idx < len(lines):
            line = lines[line_idx]
            if line.startswith("#") or line.startswith("`") or not line.strip():
                continue
            try:
                eval(line)
                line_idx += 1
                self.history[-1][0] = HistoryMark.API_CALL_CORRECT
            except Exception as e:
                err_time += 1
                if err_time > self.terminate_times:
                    break
                error_message = str(e) + traceback.format_exc()
                code_traces.append(error_message)
                api_lines = (
                    "\n".join(lines[:line_idx])
                    + f"\n--> Error Line: {line}\n"
                    + "\n".join(lines[line_idx:])
                )
                slide, template_slides = deepcopy(backup_state)
                prompt = self.correct_template.render(
                    previous_task_prompt=prompt,
                    error_message=error_message,
                    faulty_api_sequence=api_lines,
                )
                lines = agent_model(prompt).strip().split("\n")
                line_idx = 0
                self.history.append([HistoryMark.API_CALL_ERROR, prompt, apis])

        assert err_time < self.terminate_times, "\n".join(code_traces)
        return code_traces


def find_shape(element_id: str):
    for shape in slide.shapes:
        if shape.element_idx == element_id:
            return shape


# element_id 在前，让gpt添加注释和例子
def delete_text(element_id: str):
    """
    This function deletes the text of the element with the given element_id.
    """
    element_id, text_id = element_id.split("_")
    shape = find_shape(element_id)
    assert shape.text_frame.is_textframe, "The shape does't have a TextFrame."
    idx = -1
    for i in shape.text_frame.data:
        if i["text"]:
            idx += 1
        if idx == text_id:
            # TODO change for build finished
            shape.text_frame.data.remove(i)
            return
    raise ValueError("The element_id is invalid.")


def replace_text(element_id: str, text: str):
    """
    This function replaces the text of the element with the given element_id.
    """
    assert "_" in element_id, "The element_id of a text element should contain a `_` "
    element_id, text_id = list(map(int, element_id.split("_")))
    shape = find_shape(element_id)
    assert shape.text_frame.is_textframe, "The shape does't have a TextFrame."
    for para in shape.text_frame.data:
        if para["idx"] == text_id:
            if len(para["text"].splitlines()) != len(text.splitlines()):
                raise ValueError(
                    "The new text should have the same number of lines as the old text."
                )
            para["text"] = text
            return
    raise ValueError("The element_id is invalid.")


# 这里需要调整aspect ratio
# 对于clone 的shape单独调用build
# jia ge builded? image可以最后再build
def replace_image(element_id: str, image_path: str):
    """
    This function sets the image of the element with the given id.
    """
    shape = find_shape(element_id)
    if not isinstance(shape, Picture):
        raise ValueError("The shape is not a Picture.")
    shape.img_path = image_path


def select_template(template_idx):
    global slide
    slide = deepcopy(template_slides[template_idx])


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
    shape = find_shape(element_id)
    shape.shape_idx = new_element_id
    shape.style["shape_bounds"] = shape_bounds
    slide.shapes.append(shape)


def del_shape(element_id: str):
    """
    This function deletes the shape with the given id,
    """
    for shape in slide.shapes:
        if shape.element_idx == element_id:
            slide.shapes.remove(shape)


def del_image(element_id: str):
    """
    This function deletes the image of the element with the given id.
    """
    shape = find_shape(element_id)
    if not isinstance(shape, Picture):
        raise ValueError("The shape is not a Picture.")
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
    source_shape, target_shape = find_shape(source_element_id), find_shape(
        target_element_id
    )
    assert isinstance(
        source_shape, type(target_shape)
    ), "The source and target shapes should be the same type."
    # TODO 这里需要调整
    if not is_textframe:
        source_shape.style, target_shape.style = target_shape.style, source_shape.style
    else:
        source_shape.text_frame.data, target_shape.text_frame.data = (
            target_shape.text_frame.data,
            source_shape.text_frame.data,
        )


class API_TYPES(Enum):
    LAYOUT_ADJUST = "layout adjust"
    STYLE_ADJUST = "style adjust"
    SET_CONTENT = "set content"


model_api = ModelAPI(
    {
        # group shape
        API_TYPES.LAYOUT_ADJUST: [clone_shape, del_shape, select_template],
        API_TYPES.STYLE_ADJUST: [swap_style],
        API_TYPES.SET_CONTENT: [replace_text, replace_image, del_image],
    },
    5,
)


if __name__ == "__main__":
    print(
        model_api.get_apis_docs(
            [API_TYPES.SET_CONTENT, API_TYPES.LAYOUT_ADJUST, API_TYPES.STYLE_ADJUST]
        )
    )
