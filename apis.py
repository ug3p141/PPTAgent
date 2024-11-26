import inspect
import os
import re
import traceback
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Literal

import PIL
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.shapes.base import BaseShape
from pptx.util import Pt

from presentation import Closure, Picture, SlidePage
from utils import runs_merge


@dataclass
class HistoryMark:
    API_CALL_ERROR = "api_call_error"
    API_CALL_CORRECT = "api_call_correct"
    COMMAND_CORRECT = "command_correct"
    COMMAND_ERROR = "command_error"
    CODE_RUN_ERROR = "code_run_error"
    CODE_RUN_CORRECT = "code_run_correct"


class CodeExecutor:

    def __init__(self, retry_times: int):
        self.api_history = []
        self.command_history = []
        self.code_history = []
        self.retry_times = retry_times
        self.registered_functions = API_TYPES.all_funcs()
        self.command_regex = re.compile(r"^#.+(text|image).+quantity_change.+$")
        self.function_regex = re.compile(r"^[a-z]+_[a-z]+\(.+\)")

    def get_apis_docs(self, funcs: list[callable], show_example: bool = True):
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
            if not show_example:
                api_doc.append(signature)
                continue
            doc = inspect.getdoc(func)
            if doc is not None:
                signature += f"\n\t{doc}"
            api_doc.append(signature)
        return "\n".join(api_doc)

    def execute_actions(self, actions: str, edit_slide: SlidePage):
        found_code = False
        api_calls = actions.strip().split("\n")
        self.api_history.append(
            [HistoryMark.API_CALL_ERROR, edit_slide.slide_idx, actions]
        )
        for line_idx, line in enumerate(api_calls):
            try:
                if line_idx == len(api_calls) - 1 and not found_code:
                    raise PermissionError(
                        "No code block found in the output, please output the api calls without any prefix."
                    )
                if line.startswith("def"):
                    raise PermissionError("The function definition were not allowed.")
                if self.command_regex.match(line):
                    if len(self.command_history) != 0:
                        self.command_history[-1][0] = HistoryMark.COMMAND_CORRECT
                    self.command_history.append([HistoryMark.COMMAND_ERROR, line, None])
                    continue
                if not self.function_regex.match(line):
                    continue
                found_code = True
                func = line.split("(")[0]
                if func not in self.registered_functions:
                    raise NameError(f"The function {func} is not defined.")
                # only one of clone and del can be used in a row
                if func.startswith("clone") or func.startswith("del"):
                    tag = func.split("_")[0]
                    if (
                        self.command_history[-1][-1] == None
                        or self.command_history[-1][-1] == tag
                    ):
                        self.command_history[-1][-1] = tag
                    else:
                        raise ValueError(
                            "Invalid command: Both 'clone_paragraph' and 'del_span'/'del_image' are used within a single command. "
                            "Each command must only perform one of these operations based on the quantity_change."
                        )
                self.code_history.append([HistoryMark.CODE_RUN_ERROR, line, None])
                partial_func = partial(self.registered_functions[func], edit_slide)
                eval(line, {}, {func: partial_func})
                self.code_history[-1][0] = HistoryMark.CODE_RUN_CORRECT
            except:
                trace_msg = traceback.format_exc()
                if found_code:
                    self.code_history[-1][-1] = trace_msg
                api_lines = (
                    "\n".join(api_calls[: line_idx - 1])
                    + f"\n--> Error Line: {line}\n"
                    + "\n".join(api_calls[line_idx + 1 :])
                )
                return api_lines, trace_msg
        self.command_history[-1][0] = HistoryMark.COMMAND_CORRECT
        self.api_history[-1][0] = HistoryMark.API_CALL_CORRECT


# supporting functions
def element_index(slide: SlidePage, element_id: int):
    for shape in slide:
        if shape.shape_idx == element_id:
            return shape
    raise ValueError(f"Cannot find element {element_id}, is it deleted or not exist?")


def replace_run(paragraph_id: int, run_id: int, new_text: str, shape: BaseShape):
    para = shape.text_frame.paragraphs[paragraph_id]
    run = runs_merge(para)[run_id]
    run.text = new_text


def clone_para(paragraph_id: int, shape: BaseShape):
    para = shape.text_frame.paragraphs[paragraph_id]
    shape.text_frame.paragraphs[-1]._element.addnext(parse_xml(para._element.xml))


def del_run(paragraph_id: int, run_id: int, shape: BaseShape):
    para = shape.text_frame.paragraphs[paragraph_id]
    run = para.runs[run_id]
    run._r.getparent().remove(run._r)
    if len(para.runs) == 0:
        para._element.getparent().remove(para._element)


HORIZONTAL_ALIGN = {
    "left": PP_ALIGN.LEFT,
    "center": PP_ALIGN.CENTER,
    "right": PP_ALIGN.RIGHT,
}
VERTICAL_ALIGN = {
    "top": MSO_ANCHOR.TOP,
    "center": MSO_ANCHOR.MIDDLE,
    "bottom": MSO_ANCHOR.BOTTOM,
}


def set_font(
    bold: bool,
    font_size_delta: int,
    text_shape: BaseShape,
    horizontal_align: Literal["left", "center", "right"] = None,
    vertical_align: Literal["top", "center", "bottom"] = None,
):
    horizontal_align = HORIZONTAL_ALIGN.get(horizontal_align, None)
    vertical_align = VERTICAL_ALIGN.get(vertical_align, None)
    for paragraph in text_shape.text_frame.paragraphs:
        for run in paragraph.runs:
            if bold is not None:
                run.font.bold = bold
            if font_size_delta is not None:
                run.font.size = Pt(run.font.size.pt + font_size_delta)
            if horizontal_align is not None:
                paragraph.alignment = horizontal_align
            if vertical_align is not None:
                paragraph.vertical_anchor = vertical_align


def set_size(width: int, height: int, left: int, top: int, shape: BaseShape):
    if width is not None:
        shape.width = Pt(width)
    if height is not None:
        shape.height = Pt(height)
    if left is not None:
        shape.left = Pt(left)
    if top is not None:
        shape.top = Pt(top)


# api functions
def del_span(slide: SlidePage, div_id: int, paragraph_id: int, span_id: int):
    shape = element_index(slide, div_id)
    assert (
        shape.text_frame.is_textframe
    ), "The element does not have a text frame, please check the element id and type of element."
    for paragraph in shape.text_frame.paragraphs:
        if paragraph.idx == paragraph_id:
            para = paragraph
            break
    else:
        raise IndexError(
            f"Cannot find the paragraph {paragraph_id} of the element {div_id},"
            "may refer to a non-existed paragraph or you haven't cloned enough paragraphs beforehand."
        )
    for run in para.runs:
        if run.idx == span_id:
            para.runs.remove(run)
            shape._closures["delete"].append(
                Closure(
                    partial(del_run, para.real_idx, span_id), para.real_idx, span_id
                )
            )
            return
    raise IndexError(
        f"Cannot find the span {span_id} in the paragraph {paragraph_id} of the element {div_id}, may refer to a non-existent span."
    )


def del_image(slide: SlidePage, figure_id: int):
    shape = element_index(slide, figure_id)
    assert isinstance(shape, Picture), "The element is not a Picture."
    slide.shapes.remove(shape)


def replace_span(
    slide: SlidePage, div_id: int, paragraph_id: int, span_id: int, text: str
):
    shape = element_index(slide, div_id)
    assert (
        shape.text_frame.is_textframe
    ), "The element does not have a text frame, please check the element id and type of element."
    for paragraph in shape.text_frame.paragraphs:
        if paragraph.idx == paragraph_id:
            para = paragraph
            break
    else:
        raise IndexError(
            f"Cannot find the paragraph {paragraph_id} of the element {div_id},"
            "may refer to a non-existed paragraph or you haven't cloned enough paragraphs beforehand."
        )
    for run in para.runs:
        if run.idx == span_id:
            run.text = text
            shape._closures["replace"].append(
                Closure(
                    partial(replace_run, paragraph_id, span_id, text),
                    paragraph_id,
                    span_id,
                )
            )
            return
    raise IndexError(
        f"Cannot find the span {span_id} in the paragraph {paragraph_id} of the element {div_id},"
        "Please:"
        "1. check if you refer to a non-existed span."
        "2. check if you already deleted it, ensure to remove span elements from the end of the paragraph first"
        "3. consider merging adjacent replace_span operations."
    )


def replace_image(slide: SlidePage, img_id: int, image_path: str):
    if not os.path.exists(image_path):
        raise ValueError(
            f"The image {image_path} does not exist, consider use del_image if image_path in the given command is faked"
        )
    shape = element_index(slide, img_id)
    assert isinstance(shape, Picture), "The element is not a Picture."
    img_size = PIL.Image.open(image_path).size
    r = min(shape.width / img_size[0], shape.height / img_size[1])
    new_width = int(img_size[0] * r)
    new_height = int(img_size[1] * r)
    shape.width = Pt(new_width)
    shape.height = Pt(new_height)
    shape.img_path = image_path


def clone_paragraph(slide: SlidePage, div_id: int, paragraph_id: int):
    # The cloned paragraph will have a paragraph_id one greater than the current maximum in the parent element.
    shape = element_index(slide, div_id)
    assert (
        shape.text_frame.is_textframe
    ), "The element does not have a text frame, please check the element id and type of element."
    max_idx = max([para.idx for para in shape.text_frame.paragraphs])
    for para in shape.text_frame.paragraphs:
        if para.idx != paragraph_id:
            continue
        shape.text_frame.paragraphs.append(deepcopy(para))
        shape.text_frame.paragraphs[-1].idx = max_idx + 1
        shape.text_frame.paragraphs[-1].real_idx = len(shape.text_frame.paragraphs) - 1
        shape._closures["clone"].append(
            Closure(
                partial(clone_para, paragraph_id),
                shape.text_frame.paragraphs[-1].real_idx,
            )
        )
        return
    raise IndexError(
        f"Cannot find the paragraph {paragraph_id} of the element {div_id}, may refer to a non-existed paragraph."
    )


# 现在我们先一起调整目标部分：
# 1. 使用set_element_layout 避免元素重叠和大面积空白
# 2. 使用set_font 的font_size_delta缩放字体，以保证文本内容与元素的边界匹配
# 3.
def set_font_style(
    slide: SlidePage,
    element_id: int,
    bold: bool = None,
    font_size_delta: int = None,
    horizontal_align: Literal["left", "center", "right"] = None,
    vertical_align: Literal["top", "center", "bottom"] = None,
):
    """
    Set the font style of an element.
    Args:
        bold: Whether to set the font to bold.
        font_size_delta: The delta of the font size, the unit is pt, positive for larger, negative for smaller. The range is [-8, 8].
        horizontal_align: The horizontal alignment of the text, can be "left", "center" or "right".
        vertical_align: The vertical alignment of the text, can be "top", "center" or "bottom".
    Example:
    >>> set_font_style("1_1", bold=True, font_size_delta=44)
    >>> set_font_style("1_1", horizontal_align="center")
    >>> set_font_style("1_1", vertical_align="bottom")
    """
    shape = element_index(slide, element_id)
    assert (
        abs(font_size_delta) < 8
    ), "The font size delta is too large, please check the font size delta."
    assert (
        shape.text_frame.is_textframe
    ), "The element does not have a text frame, please check the element id and type of element."
    shape._closures["style"].append(
        Closure(
            partial(
                set_font,
                bold,
                font_size_delta,
                horizontal_align,
                vertical_align,
            ),
        )
    )


def set_element_layout(
    slide: SlidePage,
    element_id: int,
    width: int = None,
    height: int = None,
    left: int = None,
    top: int = None,
):
    """
    Set the size or geometry of a shape, the unit is pt.
    Example:
    >>> set_element_size("1_1", width=32, height=32)
    >>> set_element_size("1_1", left=100, top=100)
    """
    shape = element_index(slide, element_id)
    shape._closures["style"].append(
        Closure(partial(set_size, width, height, left, top))
    )


class API_TYPES(Enum):
    Agent = [
        del_span,
        del_image,
        clone_paragraph,
        replace_span,
        replace_image,
    ]

    Typographer = [
        set_font_style,
        set_element_layout,
    ]

    @classmethod
    def all_funcs(cls) -> dict[str, callable]:
        funcs = {}
        for attr in dir(cls):
            if attr.startswith("__"):
                continue
            funcs |= {func.__name__: func for func in getattr(cls, attr).value}
        return funcs


if __name__ == "__main__":
    print(CodeExecutor(0).get_apis_docs(API_TYPES.Agent.value))
