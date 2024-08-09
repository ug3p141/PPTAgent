#
# 添加以下指示：
# - 呈现内容不必完全包含SLIDE_CONTENT中的内容，尽可能少的进行clone或删除操作
# - 在clone shape的时候，要调整shape_bounds以使其不要与其他元素重叠， 且能够fit元素的内容
# - 如果
# - 或者简单一点不能增删只能改
import json

from jinja2 import Template

from apis import API_TYPES, model_api
from llms import long_model
from presentation import Picture, Presentation
from utils import app_config, pexists, pjoin, print


# cluster中各取一个作为slide选择
# 首先设置合适的shape数量，看看需不需要增加或者删除,有无重点元素
# then, 设置内容，包括图片
class PPTAgent:
    def __init__(
        self,
        presentation: Presentation,
        template: dict[str, dict[str, list[str]]],
        markdown_str: str,
        images: list[dict[str, str]],
        num_pages: int,
    ):
        self.api = model_api
        self.presentation = presentation
        self.slides_template = template
        self.markdown_str = markdown_str
        self.num_pages = num_pages
        self.images = images
        output_dir = pjoin(app_config.RUN_DIR, "agent")
        self.refined_doc = pjoin(output_dir, "refined_doc.json")
        self.slides_file = pjoin(output_dir, "slide_contents.json")
        self.outline_file = pjoin(output_dir, "slide_outline.json")
        self.slides_api_file = pjoin(output_dir, "slide_apis.json")
        # img.unset = img.is_background

    def work(self):
        if pexists(self.refined_doc):
            self.doc_json = json.load(open(self.refined_doc, "r"))
        else:
            self.doc_json = self.refine_doc()
            json.dump(
                self.doc_json, open(self.refined_doc, "w"), ensure_ascii=False, indent=4
            )
        if pexists(self.outline_file):
            self.outline = json.load(open(self.outline_file, "r"))
        else:
            self.outline = self.generate_outline()
            json.dump(
                self.outline, open(self.outline_file, "w"), ensure_ascii=False, indent=4
            )
        if pexists(self.slides_file):
            self.slides_contents = json.load(open(self.slides_file, "r"))
        else:
            self.slides_contents = self.generate_slides()
            json.dump(
                self.slides_contents,
                open(self.slides_file, "w"),
                ensure_ascii=False,
                indent=4,
            )
        self.set_content()

    def refine_doc(self):
        template = Template(open("prompts/agent/document_refine.txt").read())
        prompt = template.render(markdown_document=self.markdown_str)
        return long_model(prompt)

    # outline这部还应该要加上图片
    def generate_outline(self):
        template = Template(open("prompts/agent/slide_outline.txt").read())
        prompt = template.render(
            num_pages=self.num_pages,
            functional_pages={
                k: v.get("metadata", None)
                for k, v in self.slides_template["functional"].items()
            },
            json_content=self.doc_json,
            images=self.images,
        )
        return long_model(prompt)

    def generate_slides(self):
        template = Template(open("prompts/agent/slides_maker.txt").read())
        prompt = template.render(
            functional_types=list(self.slides_template["functional"].keys()),
            number_of_slides=self.num_pages,
            markdown_content=self.markdown_str,
            image_list=str(self.images),
        )
        return long_model(prompt)

    def verify(self):
        pass

    def execute(self):
        pass

    def set_content(self):
        template = Template(open("prompts/agent/ppt_agent_content.txt").read())
        slide_apis = []
        for slide in self.slides_contents:
            template_id = self.slides_template[slide["type"]][slide["subtype"]][0]
            slide_template = self.presentation.slides[int(template_id)]
            if slide["type"] == "functional":
                shape_idxs = []
                for shape in slide_template.shapes:
                    if isinstance(shape, Picture) and not shape.is_background:
                        shape_idxs.append(shape.shape_idx)
                    elif shape.text_frame.is_textframe:
                        for paragraph in shape.text_frame.data:
                            if paragraph["text"]:
                                shape_idxs.append(
                                    f"{shape.shape_idx}_{paragraph['idx']}"
                                )
                prompt = template.render(
                    api_documentation=model_api.get_apis_docs(
                        [API_TYPES.LAYOUT_ADJUST, API_TYPES.SET_CONTENT]
                    ),
                    slide_html_code=slide_template.to_html(),
                    element_ids=shape_idxs,
                    slide_content=str(slide),
                )
                slide_apis.append(long_model(prompt))
                # slide_apis = json.load(open(self.slides_api_file))
                model_api.execute_apis(
                    slide_template, slide_apis[-1]["layout_adjustment"]
                )
            else:
                raise NotImplementedError
            json.dump(
                slide_apis,
                open(self.slides_api_file, "w"),
                ensure_ascii=False,
                indent=4,
            )

    def adjust_layout(self):
        pass
