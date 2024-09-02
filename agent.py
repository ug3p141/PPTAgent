import json
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

from jinja2 import Template

import apis
from apis import API_TYPES, model_api
from llms import agent_model
from presentation import Picture, Presentation, SlidePage
from utils import app_config, pexists, pjoin, ppt_to_images, print


class PPTAgent:
    def __init__(
        self,
        presentation: Presentation,
        template: dict,
        images: list[dict[str, str]],
        num_slides: int,
        doc_json: dict[str, str],
        ppt_image_folder: str,
    ):
        self.presentation = presentation
        self.intermediate_prs = deepcopy(presentation)
        self.gen_prs = deepcopy(presentation)
        self.gen_prs.slides = []
        self.slide_templates = template
        self.doc_json = doc_json
        self.num_slides = num_slides
        self.images = images
        output_dir = pjoin(app_config.RUN_DIR, "agent")
        os.makedirs(output_dir, exist_ok=True)
        self.outline_file = pjoin(output_dir, "slide_outline.json")
        self.step_result_file = pjoin(output_dir, "step_result.json")
        self.gen_prs_file = pjoin(output_dir, "generated_presentation.pptx")
        self.ppt_image_folder = ppt_image_folder
        self.inter_image_folder = pjoin(app_config.RUN_DIR, "intermediate_prs_images")

    def work(self, functional_keys: dict):
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
        self.generate_slides(functional_keys)

    def generate_slides(self, functional_keys: dict, use_cache=True):
        text_edit_template = Template(open("prompts/agent/text_edit.txt").read())
        vision_edit_template = Template(open("prompts/agent/vision_edit.txt").read())
        step_results = {
            "generated_slides": [],
            "image_usage": defaultdict(int),
        }
        if use_cache and pexists(self.step_result_file):
            step_results, model_api.api_history = pickle.load(
                open(self.step_result_file, "rb")
            )
        for slide_idx, (slide_title, slide) in enumerate(self.outline.items()):
            if slide_idx != len(step_results["generated_slides"]):
                continue
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
                        f"Template {idx}\n----\n" + i.to_html()
                        for idx, i in enumerate(apis.template_slides)
                    ]
                ),
                slide_outline=self.simple_outline,
                slide_content=slide_content,
            )
            model_api.execute_apis(text_edit_prompt, apis=agent_model(text_edit_prompt))
            self.intermediate_prs.slides = [apis.slide]
            self.intermediate_prs.save(
                pjoin(app_config.RUN_DIR, "intermediate_prs.pptx")
            )
            ppt_to_images(
                pjoin(app_config.RUN_DIR, "intermediate_prs.pptx"),
                self.inter_image_folder,
            )
            inter_image = pjoin(
                self.inter_image_folder, os.listdir(self.inter_image_folder)[0]
            )
            # get image
            vision_edit_prompt = vision_edit_template.render(
                api_documentation=model_api.get_apis_docs(
                    [API_TYPES.STYLE_ADJUST, API_TYPES.IMAGE_EDITING]
                ),
                slide_html_code=slide_template.to_html(),
                slide_outline=self.simple_outline,
                provided_images="\n".join(
                    [
                        f"Image {k} used {step_results['image_usage'][k]} times, caption: {v}"
                        for k, v in self.images.items()
                    ]
                ),
            )
            model_api.execute_apis(
                vision_edit_prompt,
                agent_model(
                    vision_edit_prompt,
                    inter_image,
                ),
            )
            slide_template: SlidePage = apis.slide

            # result saving
            step_results["generated_slides"].append(apis.slide)
            pickle.dump(
                (step_results, model_api.api_history), open(self.step_result_file, "wb")
            )
            # TODO 只在vision feedback时使用style
            self.gen_prs.slides = step_results["generated_slides"]
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
