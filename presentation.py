from pptx import Presentation as PPTXPre
from pptx.chart.chart import Chart as PPTXChart
from pptx.dml.color import RGBColor
from pptx.dml.fill import _NoFill, _NoneFill
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.oxml import parse_xml
from pptx.shapes.autoshape import Shape as PPTXAutoShape
from pptx.shapes.base import BaseShape
from pptx.shapes.connector import Connector as PPTXConnector
from pptx.shapes.graphfrm import GraphicFrame
from pptx.shapes.group import GroupShape as PPTXGroupShape
from pptx.shapes.picture import Picture as PPTXPicture
from pptx.shapes.placeholder import PlaceholderPicture, SlidePlaceholder
from pptx.slide import Slide as PPTXSlide
from pptx.table import Table
from pptx.text.text import _Run
from rich import print

from utils import (
    IMAGE_EXTENSIONS,
    app_config,
    apply_fill,
    dict_to_object,
    extract_fill,
    filename_normalize,
    get_text_inlinestyle,
    object_to_dict,
    parse_groupshape,
    pjoin,
    replace_xml_node,
)

HASH_IMAGE = dict()


class TextFrame:
    def __init__(
        self,
        is_textframe: bool,
        style: dict,
        data: list[dict],
        father_idx: int,
        text: str = "",
    ):
        self.is_textframe = is_textframe
        self.style = style
        self.data = data
        self.father_idx = father_idx
        self.text = text
        # TODO induct texttag from template

    @classmethod
    def from_shape(cls, shape: BaseShape, father_idx: int):
        if not shape.has_text_frame:
            return cls(False, None, None, None, None)
        shape = shape.text_frame
        style = object_to_dict(shape, exclude=["text"])
        data = []
        for idx, paragraph in enumerate(shape.paragraphs):
            runs = paragraph.runs
            if len(runs) == 0:
                runs = [
                    _Run(r, paragraph)
                    for r in parse_xml(paragraph._element.xml.replace("fld", "r")).r_lst
                ]
            content = object_to_dict(paragraph, exclude=["runs"])
            if len(runs) == 0:
                data.append(content)
                continue
            runs_fonts = {run.font: 0 for run in runs}
            for run in runs:
                try:
                    runs_fonts[run.font] += len(run.text)
                except:
                    pass
            content["font"] = object_to_dict(max(runs_fonts, key=runs_fonts.get))
            if content["font"]["name"] == "+mj-ea":
                content["font"]["name"] = "宋体"
            content["idx"] = idx
            data.append(content)
        return cls(True, style, data, father_idx, shape.text)

    def build(self, shape: BaseShape):
        if not self.is_textframe:
            return
        tf = (
            shape.text_frame
            if shape.has_text_frame
            else shape.add_textbox(**self.style["shape_bounds"])
        )
        for pid, para in enumerate(self.data):
            font = para["font"] if "font" in para else None
            p = tf.paragraphs[0]
            if pid != 0:
                p = tf.add_paragraph()
            dict_to_object(para, p, exclude=["runs", "text", "font"])
            if font is None:
                continue
            run = p.add_run()
            if font["color"] is not None:
                run.font.fill.solid()
                run.font.fill.fore_color.rgb = RGBColor.from_string(font["color"])
            run.text = para["text"]
            dict_to_object(font, run.font, exclude=["color"])
        dict_to_object(self.style, tf)

    def to_html(self) -> str:
        repr_list = []
        pre_bullet = None
        bullets = []
        for para in self.data:
            if (para["bullet"] != pre_bullet or para == self.data[-1]) and len(
                bullets
            ) != 0:
                repr_list.extend(["<ul>"] + bullets + ["</ul>"])
                bullets.clear()
            if para["bullet"] is None and para["text"]:
                repr_list.append(
                    f"<p id='{self.father_idx}_{para['idx']}' {get_text_inlinestyle(para)}>{para['text']}</p>"
                )
            elif para["text"]:
                bullets.append(
                    f"<li id='{self.father_idx}_{para['idx']}' {get_text_inlinestyle(para)}>{para['text']}</li>"
                )
            pre_bullet = para["bullet"]
        return "\n".join(repr_list)

    def __repr__(self):
        return f"TextFrame: {self.text}"

    def __len__(self):
        if not self.is_textframe:
            return 0
        return len(self.text)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, TextFrame):
            return False
        return (
            self.is_textframe == __value.is_textframe
            and self.style == __value.style
            and self.data == __value.data
            and self.father_idx == __value.father_idx
            and self.text == __value.text
        )


class ShapeElement:
    def __init__(
        self,
        slide_idx: int,
        shape_idx: int,
        style: dict,
        data: dict,
        text_frame: TextFrame,
    ):
        self.slide_idx = slide_idx
        self.shape_idx = shape_idx
        self.style = style
        self.data = data
        self.text_frame = text_frame

    @classmethod
    def from_shape(cls, slide_idx: int, shape_idx: int, shape: BaseShape):
        line = None
        if "line" in dir(shape) and shape.line._ln is not None:
            line = {
                "fill": extract_fill(shape.line),
                "width": shape.line.width,
                "dash_style": shape.line.dash_style,
            }
        fill = extract_fill(shape)
        style = {
            "shape_bounds": {
                "width": shape.width,
                "height": shape.height,
                "left": shape.left,
                "top": shape.top,
            },
            "shape_type": str(shape.shape_type),
            "rotation": shape.rotation,
            "fill": fill,
            "line": line,
            # "shadow": shape.shadow._element.xml,
        }
        text_frame = TextFrame.from_shape(shape, shape_idx)
        obj = SHAPECAST.get(shape.shape_type, UnsupportedShape).from_shape(
            slide_idx, shape_idx, shape, style, text_frame
        )
        obj.xml = shape._element.xml
        # ? mask to enable pickling
        # obj.from_shape = shape
        return obj

    def build(self, slide, shape):
        self.text_frame.build(shape)
        apply_fill(shape, self.style["fill"])
        # if
        # replace_xml_node(shape.shadow._element, self.style['shadow'])
        if self.style["line"] is not None:
            apply_fill(shape.line, self.style["line"]["fill"])
            dict_to_object(self.style["line"], shape.line, exclude=["fill"])

        dict_to_object(self.style["shape_bounds"], shape)
        if "rotation" in dir(shape):
            shape.rotation = self.style["rotation"]
        return shape

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, type(self)):
            return False
        return (
            self.shape_idx == __value.shape_idx
            and self.style == __value.style
            and self.data == __value.data
            and self.text_frame == __value.text_frame
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: shape {self.shape_idx} of slide {self.slide_idx}"

    def to_html(self) -> str:
        return ""

    def __str__(self) -> str:
        return ""

    @property
    def inline_fill(self):
        fill_dict = self.style["fill"]
        if fill_dict is None or "type" not in fill_dict:
            return ""
        if "color" in fill_dict:
            return f"background-color: #{fill_dict['color']};"
        return f"background-color:{fill_dict['type']};"

    @property
    def inline_border(self):
        line_dict = self.style["line"]
        if line_dict is None or "type" not in line_dict["fill"]:
            return ""
        border_style = f"; border-width: {line_dict['width'].pt}pt; border-style: {line_dict['dash_style']};"
        fill_dict = line_dict["fill"]
        if "color" in fill_dict:
            return f"border-color: #{fill_dict['color']}" + border_style
        return f"border-color: #{fill_dict['type']}" + border_style

    @property
    def inline_box(self):
        return f"left: {self.left}pt; top: {self.top}pt; width: {self.width}pt; height: {self.height}pt; "

    @property
    def inline_style(self):
        return 'style="' + self.inline_box + self.inline_border + self.inline_fill + '"'

    @property
    def left(self):
        return self.style["shape_bounds"]["left"].pt

    @property
    def top(self):
        return self.style["shape_bounds"]["top"].pt

    @property
    def width(self):
        return self.style["shape_bounds"]["width"].pt

    @property
    def height(self):
        return self.style["shape_bounds"]["height"].pt

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def area(self):
        return self.width * self.height

    def __or__(self, other):
        left = min(self.left, other.left)
        top = min(self.top, other.top)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)
        width = right - left
        height = bottom - top
        return width * height

    def __and__(self, other):
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if left < right and top < bottom:
            width = right - left
            height = bottom - top
            return width * height
        else:
            return 0


class UnsupportedShape(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: BaseShape,
        style: dict,
        text_frame: TextFrame,
    ):
        # freeform 类型
        print(
            f"Unsupport shape at slide {slide_idx} shape {shape_idx}: {shape.shape_type}"
        )
        assert shape.shape_type == MSO_SHAPE_TYPE.FREEFORM or not any(
            [shape.has_chart, shape.has_table, shape.has_text_frame]
        )
        obj = cls(slide_idx, shape_idx, style, str(shape.shape_type), text_frame)
        return obj

    def build(self, slide: PPTXSlide):
        pass

    def to_html(self) -> str:
        return f"<{self.data}>"

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, UnsupportedShape):
            return False
        return True


class AutoShape(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: PPTXAutoShape,
        style: dict,
        text_frame: TextFrame,
    ):
        data = {
            "auto_shape_type": shape.auto_shape_type.real,
            "svg_tag": str(shape.auto_shape_type).split()[0].lower(),
            "is_nofill": isinstance(shape.fill._fill, (_NoneFill, _NoFill)),
            "is_line_nofill": isinstance(shape.line.fill._fill, (_NoneFill, _NoFill)),
        }
        return cls(slide_idx, shape_idx, style, data, text_frame)

    def build(self, slide: PPTXSlide):
        shape = slide.shapes.add_shape(
            self.data["auto_shape_type"], **self.style["shape_bounds"]
        )
        if self.data["is_nofill"]:
            shape.fill.background()
            self.style["fill"] = None
            # for para in self.text_frame.data:
            #     if "font" in para and 'color' not in para['font']:
            #         para["font"]["color"] = "000000"
        if self.data["is_line_nofill"]:
            shape.line.fill.background()
            self.style["line"] = None
        shape = super().build(slide, shape)

    def to_html(self) -> str:
        text = self.text_frame.to_html()
        if text:
            text = f"<p>{text}</p>"
        return f"<{self.data['svg_tag']} id='{self.shape_idx}'>\n{text}\n</{self.data['svg_tag']}>\n"


class TextBox(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: TextFrame,
        style: dict,
        text_frame: TextFrame,
    ):
        return cls(slide_idx, shape_idx, style, None, text_frame)

    def build(self, slide: PPTXSlide):
        shape = slide.shapes.add_textbox(**self.style["shape_bounds"])
        return super().build(slide, shape)

    def to_html(self) -> str:
        return f"<text id='{self.shape_idx}' {self.inline_style}>\n{self.text_frame.to_html()}\n</text>"


class Placeholder(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: SlidePlaceholder,
        style: dict,
        text_frame: TextFrame,
    ):
        assert (
            sum(
                [
                    shape.has_text_frame,
                    shape.has_chart,
                    shape.has_table,
                    isinstance(shape, PlaceholderPicture),
                ]
            )
            == 1
        ), "placeholder should have only one type"
        if isinstance(shape, PlaceholderPicture):
            data = Picture.from_shape(slide_idx, shape_idx, shape, style, text_frame)
        elif shape.has_text_frame:
            data = TextBox.from_shape(slide_idx, shape_idx, shape, style, text_frame)
        elif shape.has_chart:
            data = Chart.from_shape(slide_idx, shape_idx, shape, style, text_frame)
        elif shape.has_table:
            data = Table.from_shape(slide_idx, shape_idx, shape, style, text_frame)
        return data
        # ph = cls(slide_idx, shape_idx, style, data, text_frame)
        # ph.ph_index = shape.placeholder_format.type.value
        # return ph

    def build(self, slide: PPTXSlide):
        pass
        # if isinstance(self.data, TextBox):
        #     super().build(slide, slide.shapes[-1])


# 缩放高度，缩放宽度
# placeholder的格式获取不到
class Picture(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: PPTXPicture,
        style: dict,
        text_frame: TextFrame,
    ):
        if shape.image.ext not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported image type {shape.image.ext}")
        img_hash = shape.image.sha1
        if img_hash in HASH_IMAGE:
            img_path = HASH_IMAGE[img_hash]
        else:
            img_name = filename_normalize(shape.name)
            img_path = pjoin(
                app_config.IMAGE_DIR,
                f"{slide_idx}_{img_name}.{shape.image.ext}",
            )
            HASH_IMAGE[img_hash] = img_path
            with open(img_path, "wb") as f:
                f.write(shape.image.blob)
        style["img_style"] = {
            "crop_bottom": shape.crop_bottom,
            "crop_top": shape.crop_top,
            "crop_left": shape.crop_left,
            "crop_right": shape.crop_right,
        }
        picture = cls(
            slide_idx,
            shape_idx,
            style,
            [img_path, shape.name, ""],
            text_frame,
        )
        picture.is_background = False
        return picture

    def build(self, slide: PPTXSlide):
        shape = slide.shapes.add_picture(
            self.img_path,
            **self.style["shape_bounds"],
        )
        shape.name = self.data[1]
        dict_to_object(self.style["img_style"], shape.image)
        return super().build(slide, shape)

    @property
    def img_path(self):
        return self.data[0]

    @img_path.setter
    def img_path(self, img_path: str):
        self.data[0] = img_path

    @property
    def caption(self):
        return self.data[2]

    @caption.setter
    def caption(self, caption: str):
        self.data[2] = caption

    def to_html(self) -> str:
        if self.is_background:
            return ""
        return (
            f"<figure id='{self.shape_idx}' {self.inline_style}>\n<shape alt='{self.data[1]}' />\n"
            + (f"\n<figcaption>{self.caption}</figcaption>" if self.caption else "")
            + "\n</figure>"
        )


class GroupShape(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: PPTXGroupShape,
        style: dict,
        text_frame: TextFrame,
    ):
        data = [
            ShapeElement.from_shape(slide_idx, f"{shape_idx}.{i}", sub_shape)
            for i, sub_shape in enumerate(shape.shapes)
        ]
        for idx, shape_bounds in enumerate(parse_groupshape(shape)):
            data[idx].style["shape_bounds"] = shape_bounds
        return cls(slide_idx, shape_idx, style, data, text_frame)

    def build(self, slide: PPTXSlide):
        shapes = []
        for sub_shape in self.data:
            shapes.append(sub_shape.build(slide))
        shape = slide.shapes.add_group_shape(shapes)
        return super().build(slide, shape)

    # groupshape clone
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, GroupShape) or len(self.data) != len(__value.data):
            return False
        for shape1, shape2 in zip(self.data, __value.data):
            if isinstance(shape1, type(shape2)):
                return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self.data}"

    def to_html(self) -> str:
        return (
            f"<div class='{self.group_label}' id='{self.shape_idx}'>\n"
            + "\n".join([shape.to_html() for shape in self.data])
            + "\n</div>\n"
        )


class Chart(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: PPTXChart,
        style: dict,
        text_frame: TextFrame,
    ):
        chart = shape.chart
        style["chart_style"] = object_to_dict(chart) | {
            "chart_title": (
                chart.chart_title.text_frame.text
                if chart.chart_title.has_text_frame
                else None
            ),
        }
        chart_data = []
        for series in chart.series:
            chart_item_data = {"name": str(series.name)}
            data_list = []
            for point in series.values:
                data_list.append(point)
            chart_item_data["data_list"] = data_list
            chart_data.append(chart_item_data)
        obj = cls(slide_idx, shape_idx, style, chart_data, text_frame)
        setattr(obj, "shape", shape)
        return obj

    def build(self, slide: PPTXSlide):
        pass
        # graphic_frame = slide.shapes.add_chart(
        #     self.style["chart_style"]["chart_type"],

        # shape
        # super().build(slide, self.shape)


# 目前的问题，rows和cols的宽度或高度获取不到
# chart 和table的caption 通过对html的llm解析来获取
class Table(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: Table,
        style: dict,
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
        style["table_style"] = object_to_dict(shape.table)

        obj = cls(slide_idx, shape_idx, style, row_list, text_frame)
        setattr(obj, "shape", shape)
        return obj

    def build(self, slide: PPTXSlide):
        graphic_frame = slide.shapes.add_table(
            len(self.data), len(self.data[0]), **self.style["shape_bounds"]
        )

        dict_to_object(self.style["table_style"], graphic_frame.table)
        self.text_frame.build(graphic_frame)


# TODO  GraphicFrame
# class GraphicFrame(ShapeElement):
#     @classmethod
#     def from_shape(
#         cls,
#         slide_idx: int,
#         shape_idx: int,
#         shape: GraphicFrame,
#         style: dict,
#         text_frame: TextFrame,
#     ):
#         pass

#     def build(self, slide, shape):
#         return super().build(slide, shape)


# unable to get the connector type
class Connector(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: PPTXConnector,
        style: dict,
        text_frame: TextFrame,
    ):
        # data = [
        #     shape.connector_type,
        #     shape.begin_x,
        #     shape.begin_y,
        #     shape.end_x,
        #     shape.end_y,
        # ]
        return cls(slide_idx, shape_idx, style, None, text_frame)

    def build(self, slide: PPTXSlide):
        return
        shape = slide.shapes.add_connector(*self.data)
        return super().build(slide, shape)


# function_args = json.loads(response_message.tool_calls[0].function.arguments)
class SlidePage:
    def __init__(
        self,
        shapes: list[ShapeElement],
        slide_idx: int,
        slide_notes: str,
        slide_layout_name: str,
        slide_title: str,
        slide_width: int,
        slide_height: int,
    ):
        self.shapes = shapes
        self.slide_idx = slide_idx
        self.slide_notes = slide_notes
        self.slide_layout_name = slide_layout_name
        self.slide_title = slide_title
        self.slide_width = slide_width
        self.slide_height = slide_height
        groups_shapes_labels = []
        for shape in shapes:
            if isinstance(shape, GroupShape):
                for group_shape in groups_shapes_labels:
                    if group_shape == shape:
                        shape.group_label = group_shape.group_label
                        continue
                    groups_shapes_labels.append(shape)
                    shape.group_label = f"group_{len(groups_shapes_labels)}"

    @classmethod
    def from_slide(
        cls,
        slide: PPTXSlide,
        slide_idx: int,
        slide_width: int,
        slide_height: int,
    ):
        shapes = [
            ShapeElement.from_shape(slide_idx, i, shape)
            for i, shape in enumerate(slide.shapes)
        ]
        slide_layout_name = slide.slide_layout.name if slide.slide_layout else None
        slide_title = slide.shapes.title.text if slide.shapes.title else None
        slide_notes = (
            slide.notes_slide.notes_text_frame.text
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame
            else None
        )
        return cls(
            shapes,
            slide_idx,
            slide_notes,
            slide_layout_name,
            slide_title,
            slide_width,
            slide_height,
        )

    def build(self, prs: PPTXPre, slide_layout):
        slide: PPTXSlide = prs.slides.add_slide(slide_layout)
        placeholders = {
            ph.placeholder_format.type.value: ph for ph in slide.placeholders
        }
        for shape in self.shapes:
            if isinstance(shape, Picture):
                shape.build(slide)
            else:
                slide.shapes._spTree.insert_element_before(
                    parse_xml(shape.xml), "p:extLst"
                )
        for ph in placeholders.values():
            ph.element.getparent().remove(ph.element)
        return slide

    def get_content_types(self, shapes: list[ShapeElement] = None):
        content_types = set()
        if shapes is None:
            shapes = self.shapes
        for shape in shapes:
            if isinstance(shape, Chart):
                content_types.add("Chart")
            elif isinstance(shape, Table):
                content_types.add("Table")
            elif isinstance(shape, Picture) and not shape.is_background:
                content_types.add("Picture")
            elif isinstance(shape, GroupShape):
                content_types.union(self.get_content_types(shape.data))
        return sorted(list(content_types))

    def to_html(self) -> str:
        return "".join(
            [
                "<!DOCTYPE html>\n<html>\n",
                (f"<title>{self.slide_title}</title>\n" if self.slide_title else ""),
                f'<body style="width:{self.slide_width}pt; height:{self.slide_height}pt;">\n'
                + "\n".join(shape.to_html() for shape in self.shapes),
                "</body>\n</html>\n",
            ]
        )

    def to_text(self) -> str:
        return "\n".join(
            [
                shape.text_frame.text
                for shape in self.shapes
                if shape.text_frame.is_textframe
            ]
            + [shape.caption for shape in self.shapes if isinstance(shape, Picture)]
        )

    @property
    def text_length(self):
        return sum([len(shape.text_frame) for shape in self.shapes])

    def __len__(self):
        return len(self.shapes)

    def __eq__(self, __value) -> bool:
        if not isinstance(__value, SlidePage):
            return False
        return (
            self.shapes == __value.shapes
            and self.slide_layout_name == __value.slide_layout_name
        )


class Presentation:
    def __init__(
        self,
        slides: list[SlidePage],
        error_history: list[str],
        slide_width: float,
        slide_height: float,
        file_path: str,
        num_pages: int,
    ) -> None:
        self.slides = slides
        self.error_history = error_history
        self.slide_width = slide_width
        self.slide_height = slide_height
        self.num_pages = num_pages
        self.source_file = file_path

    @classmethod
    def from_file(cls, file_path):
        prs = PPTXPre(file_path)
        slide_width = prs.slide_width
        slide_height = prs.slide_height
        slides = []
        error_history = []
        for i, slide in enumerate(prs.slides):  # ? page range
            try:
                slides.append(
                    SlidePage.from_slide(slide, i + 1, slide_width.pt, slide_height.pt)
                )
            except Exception as e:
                error_history.append(f"Error in slide {i+1}: {e}")
                if app_config.DEBUG:
                    print(f"Error in slide {i+1}: {e}")

        num_pages = len(slides)
        return cls(
            slides, error_history, slide_width, slide_height, file_path, num_pages
        )

    # 把 table 和 chart 转换成picture
    def save(self, file_path, layout_only=False):
        self.prs = PPTXPre(self.source_file)
        self.prs.core_properties.last_modified_by = "OminiPreGen"
        self.prs.core_properties.author += "OminiPreGen with: "
        while len(self.prs.slides) != 0:
            rId = self.prs.slides._sldIdLst[0].rId
            self.prs.part.drop_rel(rId)
            del self.prs.slides._sldIdLst[0]
        layout_mapping = {layout.name: layout for layout in self.prs.slide_layouts}
        for slide in self.slides:
            if layout_only:
                self.clear_images(slide.shapes)
            pptx_slide = slide.build(self.prs, layout_mapping[slide.slide_layout_name])
            if layout_only:
                self.clear_text(pptx_slide.shapes)
        self.prs.save(file_path)

    def clear_images(self, shapes: list[ShapeElement]):
        for idx, shape in enumerate(shapes):
            if isinstance(shape, GroupShape):
                self.clear_images(shape.data)
            elif isinstance(shape, Picture) and not shape.is_background:
                shape.img_path = "resource/pic_placeholder.png"
            elif isinstance(shape, (Chart, Table)):
                shapes[idx] = Picture(
                    shape.slide_idx,
                    shape.shape_idx,
                    {"img_style": {}} | shape.style,
                    ["resource/pic_placeholder.png", f"chart_{shape.shape_idx}", ""],
                    shape.text_frame,
                )
                shapes[idx].is_background = False

    def clear_text(self, shapes: list[BaseShape]):
        for shape in shapes:
            if isinstance(shape, PPTXGroupShape):
                self.clear_text(shape.shapes)
            elif shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        run.text = "a" * len(run.text)

    def to_html(self, pages=None) -> str:
        if pages is None:
            pages = range(self.num_pages)
        return "\n".join(
            [
                f"Slide Page {slide_idx}\n" + slide.to_html()
                for slide_idx, slide in enumerate(self.slides)
                if slide_idx in pages
            ]
        )

    def __len__(self):
        return len(self.slides)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Presentation):
            return False
        return (
            self.slides == __value.slides
            and self.slide_width == __value.slide_width
            and self.slide_height == __value.slide_height
            and self.num_pages == __value.num_pages
        )


SHAPECAST: dict[int, ShapeElement] = {
    MSO_SHAPE_TYPE.AUTO_SHAPE: AutoShape,
    MSO_SHAPE_TYPE.PLACEHOLDER: Placeholder,
    MSO_SHAPE_TYPE.PICTURE: Picture,
    MSO_SHAPE_TYPE.GROUP: GroupShape,
    MSO_SHAPE_TYPE.CHART: Chart,
    MSO_SHAPE_TYPE.TABLE: Table,
    MSO_SHAPE_TYPE.TEXT_BOX: TextBox,
    MSO_SHAPE_TYPE.LINE: Connector,
}


# 使用生成后的pptx作为测试模板，这样可以保证内容一致性
if __name__ == "__main__":
    Presentation.from_file(app_config.TEST_PPT).save(
        pjoin(app_config.RUN_DIR, "test.pptx")
    )
