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
from utils import app_config, pexists, pjoin, print


class PPTAgent:
    def __init__(
        self,
        presentation: Presentation,
        template: dict,
        text_content: str,
        images: list[dict[str, str]],
        num_slides: int,
    ):
        self.presentation = presentation
        self.gen_prs = deepcopy(presentation)
        self.gen_prs.slides = []
        self.slide_templates = template
        self.text_content = text_content
        self.num_slides = num_slides
        self.images = images
        output_dir = pjoin(app_config.RUN_DIR, "agent")
        os.makedirs(output_dir, exist_ok=True)
        self.refined_doc = pjoin(output_dir, "refined_doc.json")
        self.outline_file = pjoin(output_dir, "slide_outline.json")
        self.step_result_file = pjoin(output_dir, "step_result.json")
        self.gen_prs_file = pjoin(output_dir, "generated_presentation.pptx")

    def work(self, functional_keys: dict):
        if pexists(self.refined_doc):
            self.doc_json = json.load(open(self.refined_doc, "r"))
        else:
            template = Template(open("prompts/agent/document_refine.txt").read())
            prompt = template.render(markdown_document=self.text_content)
            self.doc_json = json.loads(agent_model(prompt))
            json.dump(
                self.doc_json, open(self.refined_doc, "w"), ensure_ascii=False, indent=4
            )

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
                    [API_TYPES.LAYOUT_ADJUST, API_TYPES.IMAGE_EDITING]
                ),
                template_html_code="\n".join(
                    [
                        f"Template {idx}\n----\n" + i.to_html()
                        for idx, i in enumerate(apis.template_slides)
                    ]
                ),
                slide_outline=self.simple_outline,
                slide_content=slide_content,
                provided_images="\n".join(
                    [
                        f"Image {k} used {step_results['image_usage'][k]} times, caption: {v}"
                        for k, v in self.images.items()
                    ]
                ),
            )
            model_api.execute_apis(text_edit_prompt, apis=agent_model(text_edit_prompt))
            # layout_prompt = layout_template.render(
            #     api_documentation=model_api.get_apis_docs([API_TYPES.LAYOUT_ADJUST]),
            #     template_html_code="\n".join(
            #         [
            #             f"Template {idx}\n----\n" + i.to_html()
            #             for idx, i in enumerate(apis.template_slides)
            #         ]
            #     ),
            #     slide_outline=self.simple_outline,
            #     slide_content=slide_content,
            # )
            # model_api.execute_apis(layout_prompt, agent_model(layout_prompt))
            # slide_template = apis.slide

            # # content replacement
            # shape_idxs = self.get_replace_ids(slide_template)
            # content_prompt = content_template.render(
            #     api_documentation=model_api.get_apis_docs([API_TYPES.SET_CONTENT]),
            #     template_html_code=slide_template.to_html(),
            #     slide_outline=self.stylized_outline,
            #     slide_content=slide_content,
            #     element_ids=shape_idxs,
            #     image_usage=step_results["image_usage"],
            # )
            # model_api.execute_apis(content_prompt, agent_model(content_prompt))

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
