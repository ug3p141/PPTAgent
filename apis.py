import inspect
import os
import string
import traceback
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

import llms
from presentation import Picture, SlidePage


@dataclass
class HistoryMark:
    API_CALL_ERROR = "api_call_error"
    API_CALL_CORRECT = "api_call_correct"
    CODE_RUN_ERROR = "code_run_error"
    CODE_RUN_CORRECT = "code_run_correct"


class CodeExecutor:

    def __init__(self, local_vars: dict[str, list[callable]], retry_times: int = 1):
        self.api_history = []
        self.code_history = []
        self.registered_functions = local_vars
        self.retry_times = retry_times
        self.local_vars = {func.__name__: func for func in sum(local_vars.values(), [])}
        self.correct_template = Template(open("prompts/agent/code_feedback.txt").read())

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
                if name == "slide":
                    continue
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

    def execute_apis(self, prompt: str, apis: str, edit_slide: SlidePage):
        lines = apis.strip().split("\n")
        err_time = 0
        line_idx = 0
        code_start = False
        found_code = False
        backup_state = deepcopy(edit_slide)
        self.api_history.append(
            [HistoryMark.API_CALL_ERROR, edit_slide.slide_idx, prompt, apis]
        )
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
            elif not line or line[0] not in string.ascii_lowercase:
                continue
            self.code_history.append([HistoryMark.CODE_RUN_ERROR, line, None])
            try:
                func = line.split("(")[0]
                if func.startswith("def"):
                    raise ValueError("The function definition should not be called.")
                if func not in self.local_vars:
                    raise ValueError(f"The function {func} is not defined.")
                partial_func = partial(self.local_vars[func], edit_slide)
                eval(line, {}, {func: partial_func})
                self.code_history[-1][0] = HistoryMark.CODE_RUN_CORRECT
            except:
                err_time += 1
                if err_time > self.retry_times:
                    break
                trace_msg = traceback.format_exc()
                trace_spliter = trace_msg.find("in <module>\n ")
                if trace_spliter == -1:
                    print("No trace spliter found in the error message.")
                    exit(-1)
                error_message = trace_msg[trace_spliter + len("in <module>\n ") :]
                self.code_history[-1][-1] = error_message
                api_lines = (
                    "\n".join(lines[: line_idx - 1])
                    + f"\n--> Error Line: {line}\n"
                    + "\n".join(lines[line_idx:])
                )
                edit_slide = deepcopy(backup_state)
                prompt = self.correct_template.render(
                    previous_task_prompt=prompt,
                    error_message=error_message,
                    faulty_api_sequence=api_lines,
                )
                lines = llms.agent_model(prompt).strip().split("\n")
                line_idx = 0
                self.api_history.append([HistoryMark.API_CALL_ERROR, prompt, apis])
        if not found_code:
            self.api_history[-1][0] = HistoryMark.API_CALL_ERROR
            raise ValueError("No code block found in the api call.")
        if err_time > self.retry_times:
            self.api_history[-1][0] = HistoryMark.API_CALL_ERROR
            raise ValueError("The api call failed too many times.")
        else:
            self.api_history[-1][0] = HistoryMark.API_CALL_CORRECT

    def reset(self):
        self.api_history = []
        self.code_history = []


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

    for run in runs:
        if run != run:
            run._element.getparent().remove(run._element)
    return run


def get_textframe(slide: SlidePage, textframe_id: str):
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


def del_textframe(slide: SlidePage, textframe_id: str):
    """Delete the textframe with the given id."""
    shape, text_id = get_textframe(slide, textframe_id)
    if textframe_id in shape.closures:
        raise ValueError(
            f"The textframe {textframe_id} has been edited, your should not delete it."
        )
    shape.closures[textframe_id] = partial(
        del_para, shape.text_frame.data[text_id]["text"]
    )


def replace_text(slide: SlidePage, textframe_id: str, text: str):
    """Replace the text of the textframe with the given id."""
    shape, text_id = get_textframe(slide, textframe_id)

    if textframe_id in shape.closures:
        raise ValueError(
            f"The textframe {textframe_id} has been edited, your should not edit it again."
        )
    shape.closures[textframe_id] = partial(
        replace_para, shape.text_frame.data[text_id]["text"], text
    )


def set_font_style(
    slide: SlidePage,
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
    shape, text_id = get_textframe(slide, textframe_id)

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

    shape.closures[textframe_id] = set_font


def adjust_element_geometry(
    slide: SlidePage, element_id: str, left: int, top: int, width: int, height: int
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

    shape.closures[element_id] = set_geometry


def replace_image(slide: SlidePage, figure_id: str, image_path: str):
    """Replace the image of the element with the given id."""
    if not os.path.exists(image_path):
        raise ValueError(f"The image {image_path} does not exist.")
    shape = slide.shapes[int(figure_id)]
    if not isinstance(shape, Picture):
        raise ValueError("The element is not a Picture.")
    if figure_id in shape.closures:
        raise ValueError(
            f"The element {figure_id} has been edited, your should not edit it again."
        )
    shape.closures[figure_id] = lambda x: None
    shape.img_path = image_path


def del_element_byid(slide: SlidePage, element_id: str):
    """Delete the element with the given id"""
    if "_" in element_id:
        raise ValueError(
            "Only the element_id of a textframe can contain a `_`, not an element."
        )
    shape = slide.shapes[int(element_id)]

    def del_shape(shape: BaseShape):
        shape.element.getparent().remove(shape.element)

    if shape.text_frame.is_textframe:
        for i in range(len(shape.text_frame.data)):
            if f"{element_id}_{i}" in shape.closures:
                raise ValueError(
                    f"The element {element_id} has been edited, your should not delete it."
                )
            shape.closures[f"{element_id}_{i}"] = lambda x: None
    if element_id in shape.closures:
        raise ValueError(
            f"The element {element_id} has been deleted, your should not delete it again."
        )
    shape.closures[element_id] = del_shape


class API_TYPES(Enum):
    TUNING = "style adjust"
    TEXT_EDITING = "text editing"
    IMAGE_EDITING = "image editing"

    @classmethod
    def all_types(cls):
        return [i for i in API_TYPES]


def get_code_executor(retry_times: int = 1):
    return CodeExecutor(
        {
            API_TYPES.TEXT_EDITING: [
                replace_text,
                del_textframe,
            ],
            API_TYPES.IMAGE_EDITING: [
                replace_image,
                del_element_byid,
            ],
        },
        retry_times=retry_times,
    )