import inspect
import os
import re
import traceback
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Literal

from pptx.dml.color import RGBColor
from pptx.shapes.base import BaseShape
from pptx.slide import Slide
from pptx.text.text import _Paragraph
from pptx.util import Pt

from llms import Role
from presentation import Picture, SlidePage
from utils import get_font_style, runs_merge


@dataclass
class HistoryMark:
    API_CALL_ERROR = "api_call_error"
    API_CALL_CORRECT = "api_call_correct"
    CODE_RUN_ERROR = "code_run_error"
    CODE_RUN_CORRECT = "code_run_correct"


# todo sort 一下，根据操作类型
class CodeExecutor:

    def __init__(self, coder: Role, retry_times: int):
        self.api_history = []
        self.code_history = []
        self.coder = coder
        if coder is None:
            retry_times = 0
        self.retry_times = retry_times
        self.registered_functions = API_TYPES.all_funcs()
        self.function_regex = re.compile(r"^[a-z]+_[a-z]+\(.+\)$")

    def get_apis_docs(self, funcs: list[callable], show_example: bool = False):
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
            api_doc.append(f"{signature}\n\t{doc}")
        return "\n".join(api_doc)

    def execute_actions(self, actions: str, edit_slide: SlidePage, error_time: int = 0):
        found_code = False
        api_calls = actions.strip().split("\n")
        backup_state = deepcopy(edit_slide)
        self.api_history.append(
            [HistoryMark.API_CALL_ERROR, edit_slide.slide_idx, actions]
        )
        for line_idx, line in enumerate(api_calls):
            try:
                if line_idx == len(api_calls) - 1 and not found_code:
                    raise ValueError(
                        "No code block found in the output, wrap your code with ```python```"
                    )
                if line.startswith("def"):
                    raise ValueError("The function definition should not be output.")
                if not self.function_regex.match(line):
                    continue
                found_code = True
                func = line.split("(")[0]
                if func not in self.registered_functions:
                    raise ValueError(f"The function {func} is not defined.")
                self.code_history.append([HistoryMark.CODE_RUN_ERROR, line, None])
                partial_func = partial(self.registered_functions[func], edit_slide)
                eval(line, {}, {func: partial_func})
                self.code_history[-1][0] = HistoryMark.CODE_RUN_CORRECT
            except:
                error_time += 1
                trace_msg = traceback.format_exc()
                if found_code:
                    self.code_history[-1][-1] = trace_msg
                if error_time > self.retry_times:
                    return None
                api_lines = (
                    "\n".join(api_calls[: line_idx - 1])
                    + f"\n--> Error Line: {line}\n"
                    + "\n".join(api_calls[line_idx:])
                )
                actions = self.coder(
                    error_message=trace_msg,
                    faulty_api_sequence=api_lines,
                )
                return self.execute_actions(actions, backup_state, error_time)
        self.api_history[-1][0] = HistoryMark.API_CALL_CORRECT
        return edit_slide


# supporting functions
def element_index(slide: SlidePage, element_id: str):
    element_id = int(element_id)
    for i in slide.shapes:
        if i.shape_idx == element_id:
            return i
    raise ValueError(f"Cannot find element {element_id}, is it deleted or not exist?")


def textframe_index(slide: SlidePage, element_id: str, textframe_id: str):
    shape = element_index(slide, element_id)
    if not shape.text_frame.is_textframe:
        raise ValueError(f"The element {element_id} doesn't have a text frame.")
    for para in shape.text_frame.paragraphs:
        for run in para.runs:
            if run.idx == int(textframe_id):
                return shape, para, run
    raise ValueError(f"Incorrect textframe ID: {textframe_id}.")


def replace_para(orig_text: str, new_text: str, shape: BaseShape):
    for para in shape.text_frame.paragraphs:
        for r in runs_merge(para):
            if r.text == orig_text:
                r.text = new_text
                return
    raise ValueError(f"Cannot find para {orig_text}.")


def del_para(text: str, shape: BaseShape):
    for para in shape.text_frame.paragraphs:
        for run in runs_merge(para):
            if run.text == text:
                para._element.getparent().remove(para._element)
                if len(runs_merge(para)) == 0:
                    para._element.getparent().remove(para._element)
                if len(shape.text_frame.paragraphs) == 0:
                    shape._element.getparent().remove(shape._element)
                return
    raise ValueError(f"Cannot find para {text}.")


# api functions
def del_textframe(slide: SlidePage, element_id: str, textframe_id: str):
    shape, para, run = textframe_index(slide, element_id, textframe_id)
    shape.closures[element_id + "-" + textframe_id] = partial(del_para, run.text)


def del_picture(slide: SlidePage, element_id: str):
    shape = element_index(slide, element_id)
    if not isinstance(shape, Picture):
        raise ValueError("The element is not a Picture.")
    slide.shapes.remove(shape)


def replace_text(slide: SlidePage, element_id: str, textframe_id: str, text: str):
    shape, para, run = textframe_index(slide, element_id, textframe_id)
    shape.closures[element_id + "-" + textframe_id] = partial(
        replace_para, run.text, text
    )


def replace_image(slide: SlidePage, element_id: str, image_path: str):
    if not os.path.exists(image_path):
        raise ValueError(f"The image {image_path} does not exist.")
    shape = element_index(slide, element_id)
    if not isinstance(shape, Picture):
        raise ValueError("The element is not a Picture.")
    shape.img_path = image_path


def clone_textframe(slide: SlidePage, element_id: str, textframe_id: str):
    """Clone a textframe, new textframe_id will be the textframe_id + 1."""
    shape, para, run = textframe_index(slide, element_id, textframe_id)
    para._element.getparent().insert(para._element.getprevious(), para._element)
    # todo implement clone


# 理论上来说只可能是image 或者textframe
# ? 当且仅当一个text_frame都没有的时候，自动删除整个element
# ? text_frame的命名方式和new_element冲突了


# delete 应该不需要，从最后一个开始删除就好了
def spacing_elements(
    slide: SlidePage,
    element_ids: list[str],
    spacing_type: Literal["horizontal", "vertical"],
):
    """Arrange the spacing between elements."""
    top = min(slide.shapes[i].top for i in element_ids)
    left = min(slide.shapes[i].left for i in element_ids)
    bottom = max(slide.shapes[i].bottom for i in element_ids)
    right = max(slide.shapes[i].right for i in element_ids)
    sum_width = sum(slide.shapes[i].width for i in element_ids)
    sum_height = sum(slide.shapes[i].height for i in element_ids)
    if spacing_type == "horizontal":
        # 每两个元素之间要是相同的spacing
        el_spacing = (right - left - sum_width) / (len(element_ids) - 1)
        for rel_idx, shape_id in enumerate(element_ids):
            shape = slide.shapes[shape_id]
            shape.left = left + rel_idx * el_spacing
    elif spacing_type == "vertical":
        el_spacing = (bottom - top - sum_height) / (len(element_ids) - 1)
        for rel_idx, shape_id in enumerate(element_ids):
            shape = slide.shapes[shape_id]
            shape.top = top + rel_idx * el_spacing
    else:
        raise ValueError(
            f"Incorrect spacing type, should be `horizontal` or `vertical`."
        )


def set_font_style(
    slide: SlidePage,
    element_id: str,
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
    shape, para, run = textframe_index(slide, element_id, textframe_id)
    paratext = run.text

    def set_font(text_shape: BaseShape):
        find = False
        if not text_shape.has_text_frame:
            raise ValueError(f"The element is not a text frame: {textframe_id}.")
        for para in shape.text_frame.paragraphs:
            if para.text == paratext:
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


def set_geometry(slide: SlidePage, element_id: str, left: int = None, top: int = None):
    shape = slide.shapes[int(element_id)]
    shape.left = Pt(left)
    shape.top = Pt(top)


def set_size(slide: SlidePage, element_id: str, width: int, height: int):
    shape = slide.shapes[int(element_id)]
    shape.width = Pt(width)
    shape.height = Pt(height)

    def set_geometry(shape: BaseShape):
        shape.width = width
        shape.height = height

    shape.closures[element_id] = set_geometry


class API_TYPES(Enum):
    Agent = [
        replace_text,
        del_textframe,
        replace_image,
        del_picture,
    ]
    Coder = [
        del_textframe,
        del_picture,
        clone_textframe,
        replace_text,
        replace_image,
    ]
    Typographer = [
        set_font_style,
        # set_geometry,
        spacing_elements,
    ]

    # return all functions in the enum
    @classmethod
    def all_funcs(cls) -> dict[str, callable]:
        funcs = {}
        for attr in dir(cls):
            if attr.startswith("__"):
                continue
            funcs |= {func.__name__: func for func in getattr(cls, attr).value}
        return funcs


# 这是基于slide 元素， pptx shape 而不是presentation的操作
# 操作样式, 不操作文本内容, 单位是一个slide
# 似乎还是应该以shape为基本单位而不是slide，不然可能导致不应该被变大变小的元素也被变了
# 感觉也没事
# run的切割问题
class StyleOperator:
    def __init__(self, slide: SlidePage, pptxslide: Slide):
        self.slide = slide
        self.pptxslide = pptxslide
        cascading_styles: list[str, (list[_Paragraph], dict)]

    def set_property(self, style_class: str, property: str, value: Any):
        for para in self.cascading_styles[style_class][0]:
            assert len(para.runs) == 1
            setattr(para.runs[0].font, property, value)

    # 这里的k可以结合之前 text induct的结果
    def get_all_css(self):
        css_list = []
        for k, v in self.cascading_styles:
            style_str = get_font_style(v[1])
            css_list.append(f".{k} {{ {style_str}; }}")
        return css_list


if __name__ == "__main__":
    print(CodeExecutor(None, 0).get_apis_docs(API_TYPES.Coder.value))
