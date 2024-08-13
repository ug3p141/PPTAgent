#
# 添加以下指示：
# - 呈现内容不必完全包含SLIDE_CONTENT中的内容，尽可能少的进行clone或删除操作
# - 在clone shape的时候，要调整shape_bounds以使其不要与其他元素重叠， 且能够fit元素的内容
# - 如果
# - 或者简单一点不能增删只能改
import json
from copy import deepcopy

from jinja2 import Template

import apis
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
        self.presentation = presentation
        self.gen_prs = deepcopy(presentation)
        self.gen_prs.slides = []
        self.slide_templates = template
        self.markdown_str = markdown_str
        self.num_pages = num_pages
        self.images = images
        output_dir = pjoin(app_config.RUN_DIR, "agent")
        self.refined_doc = pjoin(output_dir, "refined_doc.json")
        self.slides_file = pjoin(output_dir, "slide_contents.json")
        self.outline_file = pjoin(output_dir, "slide_outline.json")
        self.slides_api_file = pjoin(output_dir, "slide_apis.json")
        self.gen_prs_file = pjoin(output_dir, "generated_presentation.pptx")

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

        stylized_outline = ""
        for slide_idx, slide_title in enumerate(self.outline):
            slide = self.outline[slide_title]
            stylized_outline += f"Slide {slide_idx+1}: {slide_title}\nDescription: {slide['description']}\nKeyPoints: {slide['metadata_keys']+ slide['subsection_keys']}\n\n"
        self.stylized_outline = stylized_outline
        long_model.set_plain()
        self.generate_slides()

    def refine_doc(self):
        template = Template(open("prompts/agent/document_refine.txt").read())
        prompt = template.render(markdown_document=self.markdown_str)
        return long_model(prompt)

    def generate_outline(self):
        template = Template(open("prompts/agent/slide_outline.txt").read())
        prompt = template.render(
            num_pages=self.num_pages,
            functional_pages={
                k: v.get("metadata", None)
                for k, v in self.slide_templates["functional"].items()
            },
            json_content=self.doc_json,
            images=self.images,
        )
        return long_model(prompt)

    def generate_slides(self):
        content_template = Template(open("prompts/agent/content_replacing.txt").read())
        layout_template = Template(open("prompts/agent/layout_adjust.txt").read())
        slide_apis = []
        generated_slides = []
        # 先把functional的做了，如果templates>1 ，则让模型自己选一个，否则直接用第一个
        # 添加页码和n/m functional，帮助模型选择
        for slide_idx, slide_title in enumerate(self.outline):
            slide = self.outline[slide_title]
            slide_content = f"Slide {slide_idx+1} {slide_title}\nSlide Description: {slide['description']}\nSlide Content: "
            for key in slide["metadata_keys"]:
                slide_content += f"{key}: {self.doc_json['metadata'][key]}\n"
            for key in slide["subsection_keys"]:
                for section in self.doc_json["sections"]:
                    for subsection in section["subsections"]:
                        if key in subsection:
                            slide_content += f"SubSection {key}: {subsection[key]}\n"
            if slide["images"]:
                slide_content += "Images: "
                for image_path in slide["images"]:
                    slide_content += f"{image_path}: {self.images[image_path]}\n"
            if slide_title in self.slide_templates["functional"]:
                template_ids = self.slide_templates["functional"][slide_title]
                apis.template_slides = [
                    self.presentation.slides[int(i)] for i in template_ids["slides"]
                ]
                layout_prompt = layout_template.render(
                    api_documentation=model_api.get_apis_docs(
                        [API_TYPES.LAYOUT_ADJUST]
                    ),
                    template_html_code="\n".join(
                        [
                            f"Template {idx}\n---" + i.to_html()
                            for idx, i in enumerate(apis.template_slides)
                        ]
                    ),
                    slide_outline=self.stylized_outline,
                    slide_content=slide_content,
                    layout_usage_count=0,  # TODO 记录layout的使用次数
                )
                slide_apis.append(long_model(layout_prompt))
                model_api.execute_apis(slide_apis[-1])
                slide_template = apis.slide

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
                content_prompt = content_template.render(
                    api_documentation=model_api.get_apis_docs([API_TYPES.SET_CONTENT]),
                    template_html_code=slide_template.to_html(),
                    slide_outline=self.stylized_outline,
                    slide_content=slide_content,
                    layout_usage_count=0,
                )
                slide_apis.append(long_model(content_prompt))
            model_api.execute_apis(slide_apis[-1])
            generated_slides.append(apis.slide)
            self.gen_prs.slides = generated_slides
            self.gen_prs.save(self.gen_prs_file)
            # TODO MLLM 作为agent对生成内容进行collate chat session
            # TODO 目前存在问题：生成字数不匹配

    def select_template(self):
        pass

    def adjust_layout(self):
        pass


if __name__ == "__main__":
    prs = Presentation(app_config.TEST_PPT)
