from collections import Counter
import os
import re
from pptx.oxml import parse_xml
from pptx import Presentation as PPTXPre
from pptx.slide import Slide
from pptx.shapes.autoshape import Shape as PPTXAutoShape
from pptx.shapes.picture import Picture
from pptx.shapes.base import BaseShape
from pptx.shapes.group import GroupShape
from pptx.chart.chart import Chart
from pptx.table import Table
from pptx.text.text import _Run, TextFrame
from pptx.shapes.placeholder import SlidePlaceholder
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Cm
from regex import P
from typing_extensions import Dict, Type
from utils import Config, dict_to_object, xml_print, object_to_dict  # noqa
from pptx.dml.color import RGBColor

base_config = Config()
pjoin = os.path.join


# 三种级别的text 可以保存的属性
# textframe: shape bounds
# paragraph: space, alignment, level, font
# run: font, hyperlink, text
# 以paragraphs 为单位处理文本框
class TextFrame:

    def __init__(
        self,
        is_textframe: bool,
        style: Dict,
        data: Dict,
    ):
        self.style = style
        self.data = data

    @classmethod
    def from_shape(cls, shape: BaseShape):
        if not shape.has_text_frame:
            return cls(False, {}, [])
        shape = shape.text_frame
        style = object_to_dict(shape)
        style.pop('text')
        data = []
        # 获取文本框内容
        for _, paragraph in enumerate(shape.paragraphs):
            text = (
                paragraph.text
            )  # 有些shape的text为空但也有用（例如一些矩形框），所以不对text做判断
            if not text:
                continue
            runs = paragraph.runs
            if len(runs) == 0:  # 成功解析
                runs = [
                    _Run(r, paragraph)
                    for r in parse_xml(paragraph._element.xml.replace("fld", "r")).r_lst
                ]
            # 只有fill属性懒得解析了
            content = {
                "level": paragraph.level,
                "font": object_to_dict(
                    Counter([run.font for run in runs]).most_common(1)[0][0]
                ),
            }
            if content["font"]["name"] == "+mj-ea":
                content["font"]["name"] = "宋体"

            # element_style = {}
            # 获取Run对象的字体样式 由于runs的信息负责，我认为此处应该使用markdown进行解析，放置过多复杂信息
            # assert len(paragraph.runs) == 1
            # for run in paragraph.runs:
            #     font = run.font
            #     font_style = object_to_dict(font)
            #     try:
            #         font_style['color'] = font.color.rgb
            #     except:
            #         font_style["font_color"] = None
            # content["element_style"] = element_style

            content["text"] = text  # 稍后做markdown处理
            data.append(content)
        return cls(True, style, data)

    def rebuild(self, shape: BaseShape):
        tf = shape.add_textbox(*self.style["shape_bounds"])
        for para in self.data:
            p = tf.add_paragraph()
            p.level = para["level"]
            p.text = para["text"]

            font_color = para["font"]["color"]
            if font_color is not None:
                p.font.fill.color = RGBColor(*font_color)
            dict_to_object(para["font"], p.font)


# TODO 判断rebuild后的属性是否与解析前的一直

class UnsupportedShape():
    @classmethod
    def from_shape(cls, shape_idx: int, shape: BaseShape, style: Dict, text_frame: TextFrame):
        print(f"Unsupport Shape: {shape.shape_type}")

class ShapeElement:
    def __init__(
        self,
        shape_idx: int,
        style: Dict,
        data: Dict,
        text_frame: TextFrame,
    ):
        self.shape_idx = shape_idx
        self.style = style
        self.data = data
        self.text_frame = text_frame

    @classmethod
    def from_shape(cls, shape_idx: int, shape: BaseShape):
        style = {
            "shape_bouds": {
                "width": shape.width.cm,
                "height": shape.height.cm,
                "left": shape.left.cm,
                "top": shape.top.cm,
            },
            "shape_type": shape.shape_type,
            "rotation": shape.rotation,
        }
        text_frame = TextFrame.from_shape(shape)
        return SHAPECAST.get(shape.shape_type, UnsupportedShape).from_shape(
            shape_idx, shape, style, text_frame
        )

    def get_style(self):
        return self.style

    def get_data(self):
        return self.data

    def rebuild(self, slide):
        pass


class AutoShape(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        shape_idx: int,
        shape: PPTXAutoShape,
        style: Dict,
        text_frame: TextFrame,
    ):
        data = shape.auto_shape_type
        return cls(shape_idx, style, data, text_frame)

    def rebuild(self, slide: Slide):
        shape = slide.shapes.add_shape(self.data, *self.style["shape_bounds"])
        self.text_frame.rebuild(shape)


class Placeholder(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        shape_idx: int,
        shape: SlidePlaceholder,
        style: Dict,
        text_frame: TextFrame,
    ):
        
        style |= {
            "placeholder_type" : shape.placeholder_format.type,
        }
        if not shape.has_text_frame or  shape.has_chart or shape.has_table:
            print(f"Unsupported Shape: {shape}")
        data = []
        return cls(shape_idx, style, data, text_frame)

    def rebuild(self, slide: Slide):
        pass
        #self.text_frame.rebuild(shape)


class Picture(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        shape_idx: int,
        shape: Picture,
        style: Dict,
        text_frame: TextFrame,
    ):
        img_name = re.sub(r"[\/\0]", "_", shape.name)
        img_path = pjoin(
            base_config.IMAGE_DIR,
            f"{img_name}.{shape.image.ext}",
        )
        style["img_style"] = {
            "crop_bottom": shape.crop_bottom,
            "crop_top": shape.crop_top,
            "crop_left": shape.crop_left,
            "crop_right": shape.crop_right,
            "auto_shape_type": shape.auto_shape_type,
        }
        with open(img_path, "wb") as f:
            f.write(shape.image.blob)
        return cls(shape_idx, style, img_path, text_frame)

    def rebuild(self, slide: Slide):
        shape = slide.shapes.add_picture(
            pjoin(base_config.IMAGE_DIR, self.data),
            *self.style["style"]["shape_bounds"],
        )
        dict_to_object(self.style, shape)
        self.text_frame.rebuild(shape)


class GroupShape(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        shape_idx: int,
        shape: GroupShape,
        style: Dict,
        text_frame: TextFrame,
    ):
        instance = cls(shape_idx, style, data, text_frame)
        instance.shape_list = [sub_shape for sub_shape in shape.shapes]
        return instance

    def rebuild(self, slide: Slide):

        self.text_frame.rebuild(shape)


class Chart(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        shape_idx: int,
        shape: Chart,
        style: Dict,
        text_frame: TextFrame,
    ):
        chart = shape.chart
        chart_data = []
        for series in chart.series:
            chart_item_data = {"name": str(series.name)}
            data_list = []
            for point in series.values:
                data_list.append(point)
            chart_item_data["data_list"] = data_list
            chart_data.append(chart_item_data)
        return cls(shape_idx, style, chart_data, text_frame)

    def rebuild(self, slide: Slide):

        self.text_frame.rebuild(shape)


class Table(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        shape_idx: int,
        shape: Table,
        style: Dict,
        text_frame: TextFrame,
    ):
        table = shape.table
        row_list = list()
        for row in table.rows:
            cell_list = list()
            for cell in row.cells:
                cell_data = cell.text_frame.text
                cell_list.append(cell_data)
            row_list.append(cell_list)
        return cls(shape_idx, style, row_list, text_frame)

    def rebuild(self, slide: Slide):
        self.text_frame.rebuild(shape)


# template包含哪些东西，如何定义一个template
# 1. template pics：例如banner、background pic、logo，此部分只需要手动识别插入以及rebuild
# 2. cloneable shapes：例如listed text, parallel text等，此部分需要手动安排布局
class SlidePage:
    def __init__(
        self,
        slide: Slide,
        slide_idx: int,
        shapes: list[ShapeElement],
        slide_layout_name: str,
        slide_title: str,
    ):
        self.slide_idx = slide_idx
        self.slide = slide
        self.shapes = shapes
        self.slide_layout_name = slide_layout_name
        self.slide_title = slide_title

    @classmethod
    def from_slide(cls, slide: Slide, slide_idx: int):
        shapes = [
            ShapeElement.from_shape(i, shape) for i, shape in enumerate(slide.shapes)
        ]
        slide_layout_name = slide.slide_layout.name if slide.slide_layout else None
        slide_title = slide.shapes.title.text if slide.shapes.title else None
        return cls(slide, slide_idx, shapes, slide_layout_name, slide_title)

    def rebuild(self, prs: PPTXPre, slide_layout):
        slide = prs.slides.add_slide(slide_layout)
        for shape in self.shapes:
            shape.rebuild(slide)
        if self.slide_title:
            slide.shapes.title.text = self.slide_title


class Presentation:
    def __init__(
        self,
        slides: list[SlidePage],
        slide_width: float,
        slide_height: float,
        num_pages: int,
        author: str,
    ) -> None:
        self.slides = slides
        self.slide_width = slide_width
        self.slide_height = slide_height
        self.num_pages = num_pages
        self.author = author

    @classmethod
    def from_file(cls, file_path):
        prs = PPTXPre(file_path)
        slides = [SlidePage.from_slide(slide, i) for i, slide in enumerate(prs.slides)]
        slide_width = prs.slide_width.cm
        slide_height = prs.slide_height.cm
        core_properties = prs.core_properties
        num_pages = len(slides)
        author = core_properties.author
        return cls(slides, slide_width, slide_height, num_pages, author)

    def save(self, file_path):
        prs = PPTXPre()
        prs.core_properties.author = "OminiPreGen" + self.author
        prs.slide_width = Cm(self.slide_width)
        prs.slide_height = Cm(self.slide_height)
        blank_slide_layout = prs.slide_layouts[6]
        for slide in self.slides:
            slide.rebuild(prs, blank_slide_layout)
        self.prs.save(file_path)


SHAPECAST: Dict[int, Type[ShapeElement]] = {
    MSO_SHAPE_TYPE.AUTO_SHAPE: AutoShape,
    MSO_SHAPE_TYPE.PLACEHOLDER: Placeholder,
    MSO_SHAPE_TYPE.PICTURE: Picture,
    MSO_SHAPE_TYPE.GROUP: GroupShape,
    MSO_SHAPE_TYPE.CHART: Chart,
}


if __name__ == "__main__":
    Presentation.from_file(
        pjoin(
            base_config.PPT_DIR,
            "LLM-Map-2-预训练数据构建-tutorial-2023-7-19-final.pptx",
        )
    ).save(pjoin(base_config.GEN_PPT_DIR, "test.pptx"))
