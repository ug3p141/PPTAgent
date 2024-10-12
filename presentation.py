import hashlib
from pptx import Presentation as PPTXPre
from pptx.chart.chart import Chart as PPTXChart
from pptx.dml.color import RGBColor
from pptx.dml.fill import _NoFill, _NoneFill
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.oxml import parse_xml
from pptx.shapes.autoshape import Shape as PPTXAutoShape
from pptx.shapes.base import BaseShape
from pptx.shapes.connector import Connector as PPTXConnector
from pptx.shapes.group import GroupShape as PPTXGroupShape
from pptx.shapes.picture import Picture as PPTXPicture
from pptx.shapes.placeholder import PlaceholderPicture, SlidePlaceholder
from pptx.slide import Slide as PPTXSlide
from pptx.table import Table as PPTXTable
from pptx.text.text import _Run
from rich import print

from utils import (
    IMAGE_EXTENSIONS,
    Config,
    apply_fill,
    dict_to_object,
    extract_fill,
    get_text_inlinestyle,
    get_text_pptcstyle,
    object_to_dict,
    parse_groupshape,
    pjoin,
    pexists,
)

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

    @classmethod
    def from_shape(cls, shape: BaseShape, father_idx: int):
        if not shape.has_text_frame:
            return cls(False, None, None, None, None)
        shape = shape.text_frame
        style = object_to_dict(shape, exclude=["text"])
        data = []
        idx = 0
        for paragraph in shape.paragraphs:
            runs = paragraph.runs
            if len(runs) == 0:
                runs = [
                    _Run(r, paragraph)
                    for r in parse_xml(paragraph._element.xml.replace("fld", "r")).r_lst
                ]
            content = object_to_dict(paragraph, exclude=["runs"])
            content["idx"] = idx
            if len(runs) == 0:
                content["idx"] = -1
                data.append(content)
                continue
            idx += 1
            runs_fonts = {run.font: 0 for run in runs}
            for run in runs:
                try:
                    runs_fonts[run.font] += len(run.text)
                except:
                    pass
            content["font"] = object_to_dict(max(runs_fonts, key=runs_fonts.get))
            if content["font"]["name"] == "+mj-ea":
                content["font"]["name"] = "宋体"
            data.append(content)
        return cls(True, style, data, father_idx, shape.text)

    def to_html(self, stylish: bool) -> str:
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
                    f"<p id='{self.father_idx}_{para['idx']}' {get_text_inlinestyle(para, stylish)}>{para['text']}</p>"
                )
            elif para["text"]:
                bullets.append(
                    f"<li id='{self.father_idx}_{para['idx']}' {get_text_inlinestyle(para, stylish)}>{para['text']}</li>"
                )
            pre_bullet = para["bullet"]
        return "\n".join(repr_list)

    def __repr__(self):
        return f"TextFrame: {self.text}"

    def __len__(self):
        if not self.is_textframe:
            return 0
        return len(self.text)

class ShapeElement:
    def __init__(
        self,
        slide_idx: int,
        shape_idx: int,
        style: dict,
        data: dict,
        text_frame: TextFrame,
        slide_area: float,
    ):
        self.slide_idx = slide_idx
        self.shape_idx = shape_idx
        self.style = style
        self.data = data
        self.text_frame = text_frame
        self.stylish = False
        self.closures = {}
        self.slide_area = slide_area

    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: BaseShape,
        config: Config,
        slide_area: float,
    ):
        if shape_idx > 100 and isinstance(shape, PPTXGroupShape):
            raise ValueError(f"nested group shapes are not allowed")
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
            "shape_type": str(shape.shape_type).split("(")[0].lower(),
            "rotation": shape.rotation,
            "fill": fill,
            "line": line,
        }
        text_frame = TextFrame.from_shape(shape, shape_idx)
        obj = SHAPECAST.get(shape.shape_type, UnsupportedShape).from_shape(
            slide_idx, shape_idx, shape, style, text_frame, config, slide_area
        )
        obj.xml = shape._element.xml
        return obj

    def build(self, slide, shape):
        apply_fill(shape, self.style["fill"])
        if self.style["line"] is not None:
            apply_fill(shape.line, self.style["line"]["fill"])
            dict_to_object(self.style["line"], shape.line, exclude=["fill"])

        dict_to_object(self.style["shape_bounds"], shape)
        if "rotation" in dir(shape):
            shape.rotation = self.style["rotation"]
        return shape

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: shape {self.shape_idx} of slide {self.slide_idx}"

    def to_html(self, stylish) -> str:
        self.stylish = stylish
        return ""

    @property
    def left(self):
        return self.style["shape_bounds"]["left"].pt

    @left.setter
    def left(self, value):
        self.style["shape_bounds"]["left"] = value

    @property
    def top(self):
        return self.style["shape_bounds"]["top"].pt

    @top.setter
    def top(self, value):
        self.style["shape_bounds"]["top"] = value

    @property
    def width(self):
        return self.style["shape_bounds"]["width"].pt

    @width.setter
    def width(self, value):
        self.style["shape_bounds"]["width"] = value

    @property
    def height(self):
        return self.style["shape_bounds"]["height"].pt

    @height.setter
    def height(self, value):
        self.style["shape_bounds"]["height"] = value

    @property
    def area(self):
        return self.width * self.height

    @property
    def inline_style(self):
        style = f" data-relative-area='{self.area*100/self.slide_area:.2f}%'"
        if self.stylish:
            style += f" style='left: {self.left}pt; top: {self.top}pt; width: {self.width}pt; height: {self.height}pt;'"
        return style

class UnsupportedShape(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: BaseShape,
        *args,
        **kwargs,
    ):
        raise ValueError(f"unsupported shape {shape.shape_type}")


class TextBox(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: TextFrame,
        style: dict,
        text_frame: TextFrame,
        app_config: Config,
        slide_area: float,
    ):
        return cls(slide_idx, shape_idx, style, None, text_frame, slide_area)

    def to_html(self, stylish) -> str:
        self.stylish = stylish
        return f"<div {self.inline_style}>\n{self.text_frame.to_html(stylish)}\n</div>"


class Picture(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: PPTXPicture,
        style: dict,
        text_frame: TextFrame,
        app_config: Config,
        slide_area: float,
    ):
        if shape.image.ext not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported image type {shape.image.ext}")
        img_path = pjoin(
            app_config.IMAGE_DIR,
            f"{shape.image.sha1}.{shape.image.ext}",
        )
        if not pexists(img_path):
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
            slide_area,
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

    def to_html(self, stylish) -> str:
        self.stylish = stylish
        if self.is_background:
            return ""

        return (
            f"<figure  id='{self.shape_idx}' {self.inline_style}>"
            + (f"\n<figcaption>{self.caption}</figcaption>" if self.caption else "")
            + "\n</figure>"
        )


class Placeholder(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: SlidePlaceholder,
        style: dict,
        text_frame: TextFrame,
        app_config: Config,
        slide_area: float,
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
            data = Picture.from_shape(
                slide_idx, shape_idx, shape, style, text_frame, app_config, slide_area
            )
        elif shape.has_text_frame:
            data = TextBox.from_shape(
                slide_idx, shape_idx, shape, style, text_frame, app_config, slide_area
            )
        elif shape.has_chart or shape.has_table:
            data = GraphicalShape.from_shape(
                slide_idx, shape_idx, shape, style, text_frame, app_config, slide_area
            )
        return data

class GroupShape(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: PPTXGroupShape,
        style: dict,
        text_frame: TextFrame,
        app_config: Config,
        slide_area: float,
    ):
        data = [
            ShapeElement.from_shape(
                slide_idx, (shape_idx + 1) * 100 + i, sub_shape, app_config, slide_area
            )
            for i, sub_shape in enumerate(shape.shapes)
        ]
        for idx, shape_bounds in enumerate(parse_groupshape(shape)):
            data[idx].style["shape_bounds"] = shape_bounds
        return cls(slide_idx, shape_idx, style, data, text_frame, slide_area)

    def build(self, slide: PPTXSlide):
        for sub_shape in self.data:
            if isinstance(sub_shape, (Picture, GroupShape)):
                new_shape = sub_shape.build(slide)
            else:
                new_shape = slide.shapes._spTree.insert_element_before(
                    parse_xml(sub_shape.xml), "p:extLst"
                )
            for closure in sub_shape.closures.values():
                closure(new_shape)

    def __iter__(self):
        for shape in self.data:
            if isinstance(shape, GroupShape):
                yield from shape
            else:
                yield shape

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, GroupShape) or len(self.data) != len(__value.data):
            return False
        for shape1, shape2 in zip(self.data, __value.data):
            if isinstance(shape1, type(shape2)):
                return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self.data}"

    def to_html(self, stylish) -> str:
        self.stylish = stylish
        return (
            f"<div class='{self.group_label}' {self.inline_style}>\n"
            + "\n".join([shape.to_html(stylish) for shape in self.data])
            + "\n</div>\n"
        )

class GraphicalShape(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: PPTXChart | PPTXTable,
        style: dict,
        text_frame: TextFrame,
        app_config: Config,
        slide_area: float,
    ):
        return cls(slide_idx, shape_idx, style, None, text_frame, slide_area)

    @property
    def orig_shape(self):
        return self.style["shape_type"].strip()

    def normalize(self):
        shape = Picture(
            self.slide_idx,
            self.shape_idx,
            {"img_style": {}} | self.style,
            [
                "resource/pic_placeholder.png",
                self.orig_shape,
                f"{self.orig_shape}_{self.shape_idx}",
            ],
            self.text_frame,
            self.slide_area,
        )
        shape.style["fill"] = None
        shape.style["line"] = None
        shape.is_background = False
        shape.text_frame.is_textframe = False
        return shape


class FreeShape(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: PPTXAutoShape,
        style: dict,
        text_frame: TextFrame,
        app_config: Config,
        slide_area: float,
    ):
        data = {
            "shape_type": shape.auto_shape_type.real,
            "svg_tag": str(shape.auto_shape_type).split()[0].lower(),
        }
        return cls(slide_idx, shape_idx, style, data, text_frame, slide_area)

    def to_html(self, stylish) -> str:
        self.stylish = stylish
        text = self.text_frame.to_html(stylish)
        return f"<div data-shape-type='{self.data['svg_tag']}' {self.inline_style}>\n{text}\n</div>\n"


class Connector(ShapeElement):
    @classmethod
    def from_shape(
        cls,
        slide_idx: int,
        shape_idx: int,
        shape: PPTXConnector,
        style: dict,
        text_frame: TextFrame,
        app_config: Config,
        slide_area: float,
    ):
        return FreeShape(
            slide_idx,
            shape_idx,
            style,
            {"shape_type": "connector", "svg_tag": "connector"},
            text_frame,
            slide_area,
        )


class SlidePage:
    def __init__(
        self,
        shapes: list[ShapeElement],
        slide_idx: int,
        real_idx: int,
        slide_notes: str,
        slide_layout_name: str,
        slide_title: str,
        slide_width: int,
        slide_height: int,
    ):
        self.shapes = shapes
        self.slide_idx = slide_idx
        self.real_idx = real_idx
        self.slide_notes = slide_notes
        self.slide_layout_name = slide_layout_name
        self.slide_title = slide_title
        self.slide_width = slide_width
        self.slide_height = slide_height
        groups_shapes_labels = []
        for shape in self.shape_filter(GroupShape):
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
        real_idx: int,
        slide_width: int,
        slide_height: int,
        app_config: Config,
    ):
        shapes = [
            ShapeElement.from_shape(
                slide_idx, i, shape, app_config, slide_width * slide_height
            )
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
            real_idx,
            slide_notes,
            slide_layout_name,
            slide_title,
            slide_width,
            slide_height,
        )

    def build(self, slide: PPTXSlide):
        for ph in slide.placeholders:
            ph.element.getparent().remove(ph.element)

        for shape in self.shapes:
            if isinstance(shape, (Picture, GroupShape)):
                shape.build(slide)
            else:
                slide.shapes._spTree.insert_element_before(
                    parse_xml(shape.xml), "p:extLst"
                )
            for closure in shape.closures.values():
                closure(slide.shapes[-1])
        return slide

    def shape_filter(self, shape_type: type, shapes: list[ShapeElement] = None):
        if shapes is None:
            shapes = self.shapes
        for shape in shapes:
            if isinstance(shape, shape_type):
                yield shape
            elif isinstance(shape, GroupShape):
                yield from self.shape_filter(shape_type, shape.data)

    def get_content_types(self, shapes: list[ShapeElement] = None):
        content_types = set()
        if shapes is None:
            shapes = self.shapes
        for shape in shapes:
            if isinstance(shape, GraphicalShape):
                content_types.add(shape.orig_shape)
            elif isinstance(shape, Picture):
                if not shape.img_path == "resource/pic_placeholder.png":
                    content_types.add("picture")
                else:

                    content_types.add(shape.caption)
            elif isinstance(shape, GroupShape):
                content_types.union(self.get_content_types(shape.data))
        return sorted(list(content_types))

    def to_html(self, stylish=False) -> str:
        return "".join(
            [
                "<!DOCTYPE html>\n<html>\n",
                (f"<title>{self.slide_title}</title>\n" if self.slide_title else ""),
                f'<body style="width:{self.slide_width}pt; height:{self.slide_height}pt;">\n',
                "\n".join([shape.to_html(stylish) for shape in self.shapes]),
                "</body>\n</html>\n",
            ]
        )


    def to_text(self) -> str:
        return "\n".join(
            [
                shape.text_frame.text.strip()
                for shape in self.shapes
                if shape.text_frame.is_textframe
            ]
            + [
                "Image: "+shape.caption
                for shape in self.shape_filter(Picture)
                if not shape.is_background
            ]
        )

    def normalize(self, shapes: list[ShapeElement] = None):
        if shapes is None:
            shapes = self.shapes
        for shape_idx, shape in enumerate(shapes):
            if isinstance(shape, GraphicalShape):
                shapes[shape_idx] = shape.normalize()
            elif isinstance(shape, GroupShape):
                self.normalize(shape.data)

    @property
    def text_length(self):
        return sum([len(shape.text_frame) for shape in self.shapes])

    def __iter__(self):
        for shape in self.shapes:
            if isinstance(shape, GroupShape):
                yield from shape
            else:
                yield shape

    def __len__(self):
        return len(self.shapes)


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
        self.prs = PPTXPre(self.source_file)
        self.layout_mapping = {layout.name: layout for layout in self.prs.slide_layouts}
        self.prs.core_properties.last_modified_by = "PPTAgent"

    @classmethod
    def from_file(cls, file_path: str, app_config: Config):
        prs = PPTXPre(file_path)
        slide_width = prs.slide_width
        slide_height = prs.slide_height
        slides = []
        error_history = []
        slide_idx = 0
        layouts = [layout.name for layout in prs.slide_layouts]
        for slide in prs.slides:
            if slide._element.get("show") == "0":
                continue

            slide_idx += 1
            try:
                if slide.slide_layout.name not in layouts:
                    raise ValueError(
                        f"slide layout {slide.slide_layout.name} not found"
                    )
                slides.append(
                    SlidePage.from_slide(
                        slide,
                        slide_idx - len(error_history),
                        slide_idx,
                        slide_width.pt,
                        slide_height.pt,
                        app_config,
                    )
                )
            except Exception as e:
                error_history.append((slide_idx, str(e)))
                if app_config.DEBUG:
                    print(f"Warning in slide {slide_idx}: {e}")

        num_pages = len(slides)
        return cls(
            slides, error_history, slide_width, slide_height, file_path, num_pages
        )

    def save(self, file_path, layout_only=False):
        self.clear_slides()
        for slide in self.slides:
            if layout_only:
                self.clear_images(slide.shapes)
            pptx_slide = self.build_slide(slide)
            if layout_only:
                self.clear_text(pptx_slide.shapes)
        self.prs.save(file_path)

    def build_slide(self, slide: SlidePage) -> PPTXSlide:
        return slide.build(
            self.prs.slides.add_slide(self.layout_mapping[slide.slide_layout_name])
        )

    def clear_slides(self):
        while len(self.prs.slides) != 0:
            rId = self.prs.slides._sldIdLst[0].rId
            self.prs.part.drop_rel(rId)
            del self.prs.slides._sldIdLst[0]

    def clear_images(self, shapes: list[ShapeElement]):
        for idx, shape in enumerate(shapes):
            if isinstance(shape, GroupShape):
                self.clear_images(shape.data)
            elif isinstance(shape, Picture) and not shape.is_background:
                shape.img_path = "resource/pic_placeholder.png"
            elif isinstance(shape, GraphicalShape):
                shapes[idx] = shape.normalize()

    def clear_text(self, shapes: list[BaseShape]):
        for shape in shapes:
            if isinstance(shape, PPTXGroupShape):
                self.clear_text(shape.shapes)
            elif shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        run.text = "a" * len(run.text)

    def to_html(self, pages=None, stylish: bool = False, filter: type = None) -> str:
        if pages is None:
            pages = range(self.num_pages)
        return "\n".join(
            [
                f"Slide Page {slide_idx}\n" + slide.to_html(stylish)
                for slide_idx, slide in enumerate(self.slides)
                if slide_idx in pages
            ]
        )

    def normalize(self):
        for slide in self.slides:
            slide.normalize()
        return self

    def __len__(self):
        return len(self.slides)


SHAPECAST: dict[int, ShapeElement] = {
    MSO_SHAPE_TYPE.AUTO_SHAPE: FreeShape,
    MSO_SHAPE_TYPE.PLACEHOLDER: Placeholder,
    MSO_SHAPE_TYPE.PICTURE: Picture,
    MSO_SHAPE_TYPE.GROUP: GroupShape,
    MSO_SHAPE_TYPE.TEXT_BOX: TextBox,
    MSO_SHAPE_TYPE.CHART: GraphicalShape,
    MSO_SHAPE_TYPE.TABLE: GraphicalShape,
    MSO_SHAPE_TYPE.DIAGRAM: GraphicalShape,
    MSO_SHAPE_TYPE.LINE: Connector,
}
