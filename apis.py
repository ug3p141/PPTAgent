import inspect
import string
import traceback
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial

from jinja2 import Template
from pptx.dml.color import RGBColor
from pptx.oxml import parse_xml
from pptx.shapes.base import BaseShape
from pptx.text.text import _Paragraph, _Run
from pptx.util import Pt

from llms import agent_model
from presentation import GroupShape, Picture, SlidePage

slide: SlidePage = None
image_stats: dict[str, str] = None
image_usage: dict[str, int] = None
template_slides: list[SlidePage] = []
# 在重新绑定变量名与全局变量时需要写global声明，否则不需要


@dataclass
class HistoryMark:
    API_CALL_ERROR = "api_call_error"
    API_CALL_CORRECT = "api_call_correct"
    CODE_RUN_ERROR = "code_run_error"
    CODE_RUN_CORRECT = "code_run_correct"


class CodeExecutor:

    def __init__(self, local_vars: dict[str, list[callable]], terminate_times: int = 1):
        self.registered_functions = local_vars
        self.local_vars = {func.__name__: func for func in sum(local_vars.values(), [])}
        self.correct_template = Template(open("prompts/agent/code_feedback.txt").read())
        self.terminate_times = terminate_times
        self.api_history = []
        self.code_history = []

    def get_apis_docs(self, op_types: list[str], need_doc: bool = False):
        return "\n".join(
            [
                self._func_doc(self.registered_functions[op_type], need_doc)
                for op_type in op_types
            ]
        )

    def _func_doc(self, funcs: list[callable], need_doc: bool):
        api_doc = []
        for func in funcs:
            sig = inspect.signature(func)
            params = []
            for name, param in sig.parameters.items():
                param_str = name
                if param.annotation != inspect.Parameter.empty:
                    param_str += f": {param.annotation.__name__}"
                if param.default != inspect.Parameter.empty:
                    param_str += f" = {repr(param.default)}"
                params.append(param_str)
            signature = f"def {func.__name__}({', '.join(params)})"
            if not need_doc:
                api_doc.append(signature)
                continue
            doc = inspect.getdoc(func)
            api_doc.append(f"{signature}\n\t{doc}")
        return "\n".join(api_doc)

    def execute_apis(self, prompt: str, apis: str):
        global slide, template_slides
        lines = apis.strip().split("\n")
        err_time = 0
        line_idx = 0
        code_start = False
        found_code = False
        backup_state = (deepcopy(slide), deepcopy(template_slides))
        self.api_history.append([HistoryMark.API_CALL_ERROR, prompt, apis])
        while line_idx < len(lines):
            line = lines[line_idx]
            line_idx += 1

            if line == "```python":
                code_start = True
                found_code = True
                continue
            elif line == "```" or not code_start:
                code_start = False
                continue
            elif not line or line[0] not in string.ascii_lowercase:  # not a code line
                continue
            self.code_history.append([HistoryMark.CODE_RUN_ERROR, line, None])
            try:
                eval(line)
                self.code_history[-1][0] = HistoryMark.CODE_RUN_CORRECT
            except Exception as e:
                err_time += 1
                if err_time > self.terminate_times:
                    break
                trace_msg = traceback.format_exc()
                error_message = (
                    str(e)
                    + ":\n"
                    + trace_msg[trace_msg.find("line 1, in <module>") + 20 :]
                )
                self.code_history[-1][-1] = error_message
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
                self.api_history.append([HistoryMark.API_CALL_ERROR, prompt, apis])
        if not found_code:
            raise ValueError("No code block found in the api call.")
        if err_time < self.terminate_times:
            self.api_history[-1][0] = HistoryMark.API_CALL_CORRECT
        else:
            self.api_history[-1][0] = HistoryMark.API_CALL_ERROR
            raise ValueError("The api call is not correct.")


# TODO 之后可以吧orig text的判断给remove掉，因为可能反复修改
def runs_merge(paragraph: _Paragraph):
    runs = paragraph.runs
    if len(runs) == 0:
        runs = [
            _Run(r, paragraph)
            for r in parse_xml(paragraph._element.xml.replace("fld", "r")).r_lst
        ]
    if len(runs) == 1:
        return runs[0]
    run = max(runs, key=lambda x: len(x.text))
    run.text = paragraph.text
    # remove other run
    for run in runs:
        if run != run:
            run._element.getparent().remove(run._element)
    return run


def get_textframe(textframe_id: str):
    if "_" not in textframe_id:
        raise ValueError("The element_id of a text element should contain a `_`")
    element_id, text_id = textframe_id.split("_")
    element_id, text_id = int(element_id), int(text_id)
    shape = slide.shapes[element_id]
    if not shape.text_frame.is_textframe or text_id >= len(shape.text_frame.data):
        raise ValueError(f"Incorrect textframe ID: {textframe_id}.")
    return shape, text_id


def del_para(text: str, text_shape: BaseShape):
    for para in text_shape.text_frame.paragraphs:
        if para.text == text:
            para._element.getparent().remove(para._element)
            if len(text_shape.text_frame.paragraphs) == 0:
                text_shape.element.getparent().remove(text_shape.element)
            return
    raise ValueError(f"Incorrect shape: {text_shape}.")


def replace_para(orig_text: str, new_text: str, text_shape: BaseShape):
    for para in text_shape.text_frame.paragraphs:
        if para.text == orig_text:
            run = runs_merge(para)
            run.text = new_text
            return
    raise ValueError(f"Incorrect shape: {text_shape}.")


def del_textframe(textframe_id: str):
    """Delete the textframe with the given id."""
    shape, text_id = get_textframe(textframe_id)
    shape.closures.append(partial(del_para, shape.text_frame.data[text_id]["text"]))


def replace_text(textframe_id: str, text: str):
    """Replace the text of the textframe with the given id."""
    shape, text_id = get_textframe(textframe_id)

    shape.closures.append(
        partial(replace_para, shape.text_frame.data[text_id]["text"], text)
    )


def set_font_style(
    textframe_id: str,
    bold: bool = None,
    italic: bool = None,
    underline: bool = None,
    font_size: int = None,
    font_color: str = None,
):
    """
    Set the font style of a text frame, set the font color in Hexadecimal Color Notation.
    Example:
    >>> set_font_style("1_1", bold=True, font_size=24, font_color="FF0000")
    """
    shape, text_id = get_textframe(textframe_id)

    def set_font(text_shape: BaseShape):
        find = False
        if not text_shape.has_text_frame:
            raise ValueError(f"The element is not a text frame: {textframe_id}.")
        for para in shape.text_frame.paragraphs:
            if para.text == shape.text_frame.data[text_id]["text"]:
                find = True
                break
        if not find:
            raise ValueError(f"Incorrect element id: {textframe_id}.")
        run = runs_merge(para)
        if bold is not None:
            run.font.bold = bold
        if italic is not None:
            run.font.italic = italic
        if underline is not None:
            run.font.underline = underline
        if font_size is not None:
            run.font.size = Pt(font_size)
        if font_color is not None:
            run.font.color.rgb = RGBColor.from_string(font_color)

    shape.closures.append(set_font)


def adjust_element_geometry(
    element_id: str, left: int, top: int, width: int, height: int
):
    """
    Set the position and size of a element.

    Parameters:
    element_id (str, required): The ID of the element.
    left (int, required): The left position of the element.
    top (int, required): The top position of the element.
    width (int, required): The width of the element.
    height (int, required): The height of the element.

    Example:
    >>> set_shape_position("1", 100, 150, 200, 300)
    """
    shape = slide.shapes[int(element_id)]
    shape.left = Pt(left)
    shape.top = Pt(top)
    shape.width = Pt(width)
    shape.height = Pt(height)

    def set_geometry(shape: BaseShape):
        shape.left = left
        shape.top = top
        shape.width = width
        shape.height = height

    shape.closures.append(set_geometry)


def replace_image(figure_id: str, image_path: str):
    """Replace the image of the element with the given id."""
    # if image_path not in image_stats:
    #     raise ValueError(f"The image path is not in the image stats: {image_path}.")
    shape = slide.shapes[int(figure_id)]
    if not isinstance(shape, Picture):
        raise ValueError("The element is not a Picture.")
    shape.img_path = image_path
    shape.caption = image_stats[image_path]
    image_usage[image_path] += 1


def del_element_byid(element_id: str):
    """Delete the element with the given id"""
    shape = slide.shapes[int(element_id)]

    def del_shape(shape: BaseShape):
        shape.element.getparent().remove(shape.element)

    shape.closures.append(del_shape)


class API_TYPES(Enum):
    TUNING = "style adjust"
    TEXT_EDITING = "text editing"
    IMAGE_EDITING = "image editing"

    @classmethod
    def all_types(cls):
        return [i for i in API_TYPES]


code_executor = CodeExecutor(
    {
        API_TYPES.TUNING: [
            set_font_style,
            adjust_element_geometry,
            del_element_byid,
        ],
        API_TYPES.TEXT_EDITING: [
            replace_text,
            del_textframe,
        ],
        API_TYPES.IMAGE_EDITING: [
            replace_image,
            del_element_byid,
        ],
    },
)

if __name__ == "__main__":
    print(code_executor.get_apis_docs([API_TYPES.TUNING]))
