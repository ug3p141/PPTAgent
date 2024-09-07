import json
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

from jinja2 import Template
from pptx import Presentation as PPTXPre

import apis
from apis import API_TYPES, model_api
from llms import agent_model
from presentation import Presentation, SlidePage
from utils import app_config, clear_slides, pexists, pjoin, ppt_to_images, print


class PPTAgent:
    def __init__(
        self,
        presentation: Presentation,
        template: dict,
        images: dict[str, str],
        num_slides: int,
        doc_json: dict[str, str],
    ):
        self.presentation = presentation
        self.slide_templates = template
        self.doc_json = doc_json
        self.num_slides = num_slides
        self.images = images

        apis.image_stats = images
        apis.image_usage = defaultdict(int)

        self.gen_prs = deepcopy(presentation)
        self.gen_prs.slides = []
        self.inter_pr = PPTXPre(presentation.source_file)

        output_dir = pjoin(app_config.RUN_DIR, "agent")
        os.makedirs(output_dir, exist_ok=True)
        self.outline_file = pjoin(output_dir, "slide_outline.json")
        self.step_result_file = pjoin(output_dir, "step_result.pkl")
        self.gen_prs_file = pjoin(output_dir, "generated_presentation.pptx")

    def work(self):
        if pexists(self.outline_file):
            self.outline = json.load(open(self.outline_file, "r"))
        else:
            self.outline = json.loads(self.generate_outline())
            json.dump(
                self.outline, open(self.outline_file, "w"), ensure_ascii=False, indent=4
            )

        self.simple_outline = (
            "\n".join(
                [
                    f"Slide {slide_idx+1}: {slide_title}"
                    for slide_idx, slide_title in enumerate(self.outline)
                ]
            )
            + f"\nMetadata: {self.doc_json['metadata']}\nCurrent Time: {datetime.now().strftime('%Y-%m-%d')}\n"
        )
        self.generate_slides()

    def generate_slides(self):
        text_edit_template = Template(open("prompts/agent/text_edit.txt").read())
        vision_edit_template = Template(open("prompts/agent/vision_edit.txt").read())
        generated_slides = []
        for slide_idx, (slide_title, slide) in enumerate(self.outline.items()):
            slide_content = self.get_slide_content(slide_idx, slide_title, slide)
            # layout adjustment
            if slide["layout"] not in self.slide_templates:
                raise ValueError(f"{slide['layout']} not in slide_templates")
            template_ids = self.slide_templates[slide["layout"]]
            apis.template_slides = [
                self.presentation.slides[int(i - 1)] for i in template_ids
            ]
            text_edit_prompt = text_edit_template.render(
                api_documentation=model_api.get_apis_docs(
                    [API_TYPES.LAYOUT_ADJUST, API_TYPES.TEXT_EDITING]
                ),
                template_html_code="\n".join(
                    [
                        f"Template index :{idx+1}" + i.to_html(True) + "----"
                        for idx, i in enumerate(apis.template_slides)
                    ]
                ),
                slide_outline=self.simple_outline,
                slide_content=slide_content,
            )
            # TODO 还是提供图片或者style可能会更好，不然的话，text的位置乱填
            model_api.execute_apis(
                text_edit_prompt,
                apis=agent_model(
                    text_edit_prompt,
                ),
            )
            clear_slides(self.inter_pr)
            apis.slide.build(
                self.inter_pr,
                self.gen_prs.layout_mapping[apis.slide.slide_layout_name],
            )
            self.inter_pr.save(pjoin(app_config.RUN_DIR, "inter_prs.pptx"))
            ppt_to_images(
                pjoin(app_config.RUN_DIR, "inter_prs.pptx"),
                pjoin(app_config.RUN_DIR),
            )
            # get image
            vision_edit_prompt = vision_edit_template.render(
                api_documentation=model_api.get_apis_docs(
                    [API_TYPES.STYLE_ADJUST, API_TYPES.IMAGE_EDITING]
                ),
                slide_html_code=apis.slide.to_html(True),
                slide_outline=self.simple_outline,
                provided_images="\n".join(
                    [
                        f"Image {k} used {apis.image_usage} times, caption: {v}"
                        for k, v in self.images.items()
                    ]
                ),
            )
            model_api.execute_apis(
                vision_edit_prompt,
                agent_model(
                    vision_edit_prompt,
                    pjoin(app_config.RUN_DIR, "slide_0001.jpg"),
                ),
            )
            generated_slides.append(apis.slide)
        self.gen_prs.slides = generated_slides
        self.gen_prs.save(self.gen_prs_file)

    def generate_outline(self):
        template = Template(open("prompts/agent/slide_outline.txt").read())
        prompt = template.render(
            num_slides=self.num_slides,
            layouts=list(self.slide_templates.keys()),
            json_content=self.doc_json,
        )
        return agent_model(prompt)

    def get_slide_content(self, slide_idx: int, slide_title: str, slide: dict):
        slide_content = f"Slide-{slide_idx+1} Title: {slide_title}\nSlide Description: {slide['description']}\n"
        if len(slide["subsection_keys"]) != 0:
            slide_content += "Slide Reference Text: "
            for key in slide["subsection_keys"]:
                for section in self.doc_json["sections"]:
                    for subsection in section["subsections"]:
                        if key in subsection:
                            slide_content += f"SubSection {key}: {subsection[key]}\n"
        return slide_content


if __name__ == "__main__":
    prs = Presentation(app_config.TEST_PPT)
