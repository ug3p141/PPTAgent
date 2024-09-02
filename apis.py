import inspect
import traceback
from copy import deepcopy
from enum import Enum

from jinja2 import Template
from pptx.dml.color import RGBColor
from pptx.oxml import parse_xml
from pptx.text.text import _Paragraph, _Run
from pptx.util import Pt

from llms import agent_model
from presentation import GroupShape, Picture, SlidePage

slide: SlidePage = None
template_slides: list[SlidePage] = []
# 在重新绑定变量名与全局变量时需要写global声明，否则不需要


class HistoryMark(Enum):
    API_CALL_ERROR = "api_call_error"
    API_CALL_CORRECT = "api_call_correct"
    CODE_RUN_ERROR = "code_run_error"
    CODE_RUN_CORRECT = "code_run_correct"


class ModelAPI:

    def __init__(self, local_vars: dict[str, list[callable]], terminate_times: int = 5):
        self.registered_functions = local_vars
        self.local_vars = {func.__name__: func for func in sum(local_vars.values(), [])}
        self.correct_template = Template(open("prompts/agent/code_feedback.txt").read())
        self.terminate_times = terminate_times
        self.api_history = []
        self.code_history = []

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
        self.api_history.append([HistoryMark.API_CALL_ERROR, prompt, apis])
        code_start = False
        while line_idx < len(lines):
            line = lines[line_idx]
            line_idx += 1
            if line.startswith("<code>"):
                code_start = True
                continue
            if line.startswith("</code>"):
                code_start = False
            if not code_start:
                continue
            self.code_history.append([HistoryMark.CODE_RUN_ERROR, line])
            try:
                eval(line)
                self.code_history[-1][0] = HistoryMark.CODE_RUN_CORRECT
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
                self.api_history.append([HistoryMark.API_CALL_ERROR, prompt, apis])

        if err_time < self.terminate_times:
            self.api_history[-1][0] = HistoryMark.API_CALL_CORRECT
        else:
            self.api_history[-1][0] = HistoryMark.API_CALL_ERROR
            raise ValueError("The api call is not correct.")
        return code_traces


def runs_merge(paragraph: _Paragraph):
    runs = paragraph.runs
    if len(runs) == 0:
        runs = [
            _Run(r, paragraph)
            for r in parse_xml(paragraph._element.xml.replace("fld", "r")).r_lst
        ]
    run = max(runs, key=lambda x: len(x.text))
    run.text = paragraph.text
    # remove other run
    for run in runs:
        if run != run:
            run._element.getparent().remove(run._element)
    return run


def get_textframe(textframe_id: str):
    assert "_" in textframe_id, "The element_id of a text element should contain a `_` "
    element_id, text_id = textframe_id.split("_")
    shape = slide[element_id]
    assert (
        shape.text_frame.is_textframe and len(shape.text_frame.data) > text_id
    ), f"Incorrect element id: {element_id}."
    return shape, text_id


def delete_text(textframe_id: str):
    """Delete the text of the element with the given element_id."""
    shape, text_id = get_textframe(textframe_id)

    def del_para(shape):
        para = shape.paragraphs[text_id]._p
        para.getparent().remove(para)

    shape.closures.append(del_para)


def replace_text(textframe_id: str, text: str):
    """Replaces the text of the element with the given element_id."""
    shape, text_id = get_textframe(textframe_id)

    def replace_para(shape):
        run = runs_merge(shape.paragraphs[text_id])
        run.text = text

    shape.closures.append(replace_para)


def set_font_style(
    textframe_id: str,
    bold: bool = None,
    italic: bool = None,
    underline: bool = None,
    font_size: int = None,
    font_color: str = None,
):
    """
    Set the font style of a text frame.

    Parameters:
    textframe_id (str, required): The ID of the text frame.
    bold (bool, optional): Whether the text should be bold. Defaults to None.
    italic (bool, optional): Whether the text should be italic. Defaults to None.
    underline (bool, optional): Whether the text should be underlined. Defaults to None.
    font_size (int, optional): The font size. Defaults to None.
    font_color (str, optional): The font color in RGB format (e.g., 'FF0000' for red). Defaults to None.

    Example:
    >>> set_font_style("1_1", bold=True, font_size=24, font_color="FF0000")
    """
    shape, text_id = get_textframe(textframe_id)

    def set_font(shape):
        run = runs_merge(shape.paragraphs[text_id])
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
    shape = slide[element_id]

    def set_geometry(shape):
        shape.left = left
        shape.top = top
        shape.width = width
        shape.height = height

    shape.closures.append(set_geometry)


def replace_image(figure_id: str, image_path: str):
    """Replace the image of the element with the given id."""
    shape = slide[figure_id]
    if not isinstance(shape, Picture):
        raise ValueError("The element is not a Picture.")
    shape.img_path = image_path


def select_template(template_idx: int):
    """Select the template slide with the specified index."""
    global slide
    slide = deepcopy(template_slides[template_idx])


def del_element_byid(element_id: str):
    """Delete the element with the given id"""
    for shape in slide.shapes:
        if shape.element_idx == element_id:
            slide.shapes.remove(shape)
            return
        if isinstance(shape, GroupShape):
            for sub_shape in shape.shapes:
                if sub_shape.element_idx == element_id:
                    shape.shapes.remove(sub_shape)
                    return


class API_TYPES(Enum):
    STYLE_ADJUST = "style adjust"
    TEXT_EDITING = "text editing"
    IMAGE_EDITING = "image editing"
    LAYOUT_ADJUST = "layout adjust"


model_api = ModelAPI(
    {
        API_TYPES.LAYOUT_ADJUST: [select_template, del_element_byid],
        API_TYPES.STYLE_ADJUST: [
            set_font_style,
            adjust_element_geometry,
            del_element_byid,
        ],
        API_TYPES.TEXT_EDITING: [replace_text],
        API_TYPES.IMAGE_EDITING: [replace_image],
    },
)

if __name__ == "__main__":
    print(model_api.get_apis_docs([API_TYPES.SLIDE_GENERATE, API_TYPES.STYLE_ADJUST]))
